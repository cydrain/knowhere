// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef SPARSE_INVERTED_INDEX_H
#define SPARSE_INVERTED_INDEX_H

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "index/sparse/sparse_inverted_index_config.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"

namespace knowhere::sparse {
template <typename T>
class BaseInvertedIndex {
 public:
    virtual ~BaseInvertedIndex() = default;

    virtual Status
    Save(MemoryIOWriter& writer) = 0;

    // supplement_target_filename: when in mmap mode, we need an extra file to store the mmaped index data structure.
    // this file will be created during loading and deleted in the destructor.
    virtual Status
    Load(MemoryIOReader& reader, int map_flags, const std::string& supplement_target_filename) = 0;

    virtual Status
    Train(const SparseRow<T>* data, size_t rows) = 0;

    virtual Status
    Add(const SparseRow<T>* data, size_t rows, int64_t dim) = 0;

    virtual void
    Search(const SparseRow<T>& query, size_t k, float drop_ratio_search, float* distances, label_t* labels,
           size_t refine_factor, const BitsetView& bitset, const DocValueComputer<T>& computer) const = 0;

    virtual std::vector<float>
    GetAllDistances(const SparseRow<T>& query, float drop_ratio_search, const BitsetView& bitset,
                    const DocValueComputer<T>& computer) const = 0;

    virtual float
    GetRawDistance(const label_t vec_id, const SparseRow<T>& query, const DocValueComputer<T>& computer) const = 0;

    virtual expected<DocValueComputer<T>>
    GetDocValueComputer(const SparseInvertedIndexConfig& cfg) const = 0;

    [[nodiscard]] virtual size_t
    size() const = 0;

    [[nodiscard]] virtual size_t
    n_rows() const = 0;

    [[nodiscard]] virtual size_t
    n_cols() const = 0;
};

template <typename DType, typename QType, bool use_wand = false, bool bm25 = false, bool mmapped = false>
class InvertedIndex : public BaseInvertedIndex<DType> {
 public:
    explicit InvertedIndex() {
    }

    ~InvertedIndex() override {
        if constexpr (mmapped) {
            if (map_ != nullptr) {
                auto res = munmap(map_, map_byte_size_);
                if (res != 0) {
                    LOG_KNOWHERE_ERROR_ << "Failed to munmap when deleting sparse InvertedIndex: " << strerror(errno);
                }
                map_ = nullptr;
                map_byte_size_ = 0;
            }
            if (map_fd_ != -1) {
                // closing the file descriptor will also cause the file to be deleted.
                close(map_fd_);
                map_fd_ = -1;
            }
        }
    }

    template <typename U>
    using Vector = std::conditional_t<mmapped, GrowableVectorView<U>, std::vector<U>>;

    void
    SetBM25Params(float k1, float b, float avgdl, float max_score_ratio) {
        bm25_params_ = std::make_unique<BM25Params>(k1, b, avgdl, max_score_ratio);
    }

    expected<DocValueComputer<float>>
    GetDocValueComputer(const SparseInvertedIndexConfig& cfg) const override {
        // if metric_type is set in config, it must match with how the index was built.
        auto metric_type = cfg.metric_type;
        if constexpr (!bm25) {
            if (metric_type.has_value() && !IsMetricType(metric_type.value(), metric::IP)) {
                auto msg =
                    "metric type not match, expected: " + std::string(metric::IP) + ", got: " + metric_type.value();
                return expected<DocValueComputer<float>>::Err(Status::invalid_metric_type, msg);
            }
            return GetDocValueOriginalComputer<float>();
        }
        if (metric_type.has_value() && !IsMetricType(metric_type.value(), metric::BM25)) {
            auto msg =
                "metric type not match, expected: " + std::string(metric::BM25) + ", got: " + metric_type.value();
            return expected<DocValueComputer<float>>::Err(Status::invalid_metric_type, msg);
        }
        // avgdl must be supplied during search
        if (!cfg.bm25_avgdl.has_value()) {
            return expected<DocValueComputer<float>>::Err(Status::invalid_args,
                                                          "avgdl must be supplied during searching");
        }
        auto avgdl = cfg.bm25_avgdl.value();
        if constexpr (use_wand) {
            // wand: search time k1/b must equal load time config.
            if ((cfg.bm25_k1.has_value() && cfg.bm25_k1.value() != bm25_params_->k1) ||
                ((cfg.bm25_b.has_value() && cfg.bm25_b.value() != bm25_params_->b))) {
                return expected<DocValueComputer<float>>::Err(
                    Status::invalid_args, "search time k1/b must equal load time config for WAND index.");
            }
            return GetDocValueBM25Computer<float>(bm25_params_->k1, bm25_params_->b, avgdl);
        } else {
            // inverted index: search time k1/b may override load time config.
            auto k1 = cfg.bm25_k1.has_value() ? cfg.bm25_k1.value() : bm25_params_->k1;
            auto b = cfg.bm25_b.has_value() ? cfg.bm25_b.value() : bm25_params_->b;
            return GetDocValueBM25Computer<float>(k1, b, avgdl);
        }
    }

    Status
    Save(MemoryIOWriter& writer) override {
        /**
         * Layout:
         *
         * 1. size_t rows
         * 2. size_t cols
         * 3. DType value_threshold_ (deprecated)
         * 4. for each row:
         *     1. size_t len
         *     2. for each non-zero value:
         *        1. table_t idx
         *        2. DType val (when QType is different from DType, the QType value of val is stored as a DType with
         *           precision loss)
         *
         * inverted_index_ids_, inverted_index_vals_ and max_score_in_dim_ are
         * not serialized, they will be constructed dynamically during
         * deserialization.
         *
         * Data are densely packed in serialized bytes and no padding is added.
         */
        DType deprecated_value_threshold = 0;
        writeBinaryPOD(writer, n_rows_internal_);
        writeBinaryPOD(writer, max_dim_);
        writeBinaryPOD(writer, deprecated_value_threshold);
        BitsetView bitset(nullptr, 0);

        std::vector<Cursor<BitsetView>> cursors;
        for (size_t i = 0; i < inverted_index_ids_.size(); ++i) {
            cursors.emplace_back(inverted_index_ids_[i], inverted_index_vals_[i], n_rows_internal_, 0, 0, bitset);
        }

        auto dim_map_reverse = std::unordered_map<uint32_t, table_t>();
        for (auto dim_it = dim_map_.begin(); dim_it != dim_map_.end(); ++dim_it) {
            dim_map_reverse[dim_it->second] = dim_it->first;
        }

        for (table_t vec_id = 0; vec_id < n_rows_internal_; ++vec_id) {
            std::vector<std::pair<table_t, DType>> vec_row;
            for (size_t i = 0; i < inverted_index_ids_.size(); ++i) {
                if (cursors[i].cur_vec_id_ == vec_id) {
                    vec_row.emplace_back(dim_map_reverse[i], cursors[i].cur_vec_val());
                    cursors[i].next();
                }
            }

            SparseRow<DType> raw_row(vec_row);
            writeBinaryPOD(writer, raw_row.size());
            if (raw_row.size() == 0) {
                continue;
            }
            writer.write(raw_row.data(), raw_row.size() * SparseRow<DType>::element_size());
        }

        return Status::success;
    }
    Status
    Load(MemoryIOReader& reader, int map_flags, const std::string& supplement_target_filename) override {
        DType deprecated_value_threshold;
        int64_t rows;
        readBinaryPOD(reader, rows);
        // previous versions used the signness of rows to indicate whether to
        // use wand. now we use a template parameter to control this thus simply
        // take the absolute value of rows.
        rows = std::abs(rows);
        readBinaryPOD(reader, max_dim_);
        readBinaryPOD(reader, deprecated_value_threshold);

        if constexpr (mmapped) {
            RETURN_IF_ERROR(PrepareMmap(reader, rows, map_flags, supplement_target_filename));
        } else {
            if constexpr (bm25) {
                bm25_params_->row_sums.reserve(rows);
            }
        }

        for (int64_t i = 0; i < rows; ++i) {
            size_t count;
            readBinaryPOD(reader, count);
            SparseRow<DType> raw_row;
            if constexpr (mmapped) {
                raw_row = std::move(SparseRow<DType>(count, reader.data() + reader.tellg(), false));
                reader.advance(count * SparseRow<DType>::element_size());
            } else {
                raw_row = std::move(SparseRow<DType>(count));
                if (count > 0) {
                    reader.read(raw_row.data(), count * SparseRow<DType>::element_size());
                }
            }
            add_row_to_index(raw_row, i);
        }

        n_rows_internal_ = rows;

        return Status::success;
    }

    // memory in reader must be guaranteed to be valid during the lifetime of this object.
    Status
    PrepareMmap(MemoryIOReader& reader, size_t rows, int map_flags, const std::string& supplement_target_filename) {
        const auto initial_reader_location = reader.tellg();
        const auto nnz = (reader.remaining() - (rows * sizeof(size_t))) / SparseRow<DType>::element_size();

        // count raw vector idx occurrences
        std::unordered_map<table_t, size_t> idx_counts;
        for (size_t i = 0; i < rows; ++i) {
            size_t row_nnz;
            readBinaryPOD(reader, row_nnz);
            if (row_nnz == 0) {
                continue;
            }
            for (size_t j = 0; j < row_nnz; ++j) {
                table_t idx;
                readBinaryPOD(reader, idx);
                idx_counts[idx]++;
                // skip value
                reader.advance(sizeof(DType));
            }
        }
        // reset reader to the beginning
        reader.seekg(initial_reader_location);

        auto inverted_index_ids_byte_size =
            idx_counts.size() * sizeof(typename decltype(inverted_index_ids_)::value_type);
        auto inverted_index_vals_byte_size =
            idx_counts.size() * sizeof(typename decltype(inverted_index_vals_)::value_type);
        auto plists_ids_byte_size = nnz * sizeof(typename decltype(inverted_index_ids_)::value_type::value_type);
        auto plists_vals_byte_size = nnz * sizeof(typename decltype(inverted_index_vals_)::value_type::value_type);
        auto max_score_in_dim_byte_size = idx_counts.size() * sizeof(typename decltype(max_score_in_dim_)::value_type);
        size_t row_sums_byte_size = 0;

        map_byte_size_ =
            inverted_index_ids_byte_size + inverted_index_vals_byte_size + plists_ids_byte_size + plists_vals_byte_size;
        if constexpr (use_wand) {
            map_byte_size_ += max_score_in_dim_byte_size;
        }
        if constexpr (bm25) {
            row_sums_byte_size = rows * sizeof(typename decltype(bm25_params_->row_sums)::value_type);
            map_byte_size_ += row_sums_byte_size;
        }

        std::ofstream temp_file(supplement_target_filename, std::ios::binary | std::ios::trunc);
        if (!temp_file) {
            LOG_KNOWHERE_ERROR_ << "Failed to create mmap file when loading sparse InvertedIndex: " << strerror(errno);
            return Status::disk_file_error;
        }
        temp_file.close();

        std::filesystem::resize_file(supplement_target_filename, map_byte_size_);

        map_fd_ = open(supplement_target_filename.c_str(), O_RDWR);
        if (map_fd_ == -1) {
            LOG_KNOWHERE_ERROR_ << "Failed to open mmap file when loading sparse InvertedIndex: " << strerror(errno);
            return Status::disk_file_error;
        }
        // file will disappear in the filesystem immediately but the actual file will not be deleted
        // until the file descriptor is closed in the destructor.
        std::filesystem::remove(supplement_target_filename);

        // clear MAP_PRIVATE flag: we need to write to this mmapped memory/file,
        // MAP_PRIVATE triggers copy-on-write and uses extra anonymous memory.
        map_flags &= ~MAP_PRIVATE;
        map_flags |= MAP_SHARED;

        map_ = static_cast<char*>(mmap(nullptr, map_byte_size_, PROT_READ | PROT_WRITE, map_flags, map_fd_, 0));
        if (map_ == MAP_FAILED) {
            LOG_KNOWHERE_ERROR_ << "Failed to create mmap when loading sparse InvertedIndex: " << strerror(errno)
                                << ", size: " << map_byte_size_ << " on file: " << supplement_target_filename;
            return Status::disk_file_error;
        }
        if (madvise(map_, map_byte_size_, MADV_RANDOM) != 0) {
            LOG_KNOWHERE_WARNING_ << "Failed to madvise mmap when loading sparse InvertedIndex: " << strerror(errno);
        }

        char* ptr = map_;

        // initialize containers memory.
        inverted_index_ids_.initialize(ptr, inverted_index_ids_byte_size);
        ptr += inverted_index_ids_byte_size;
        inverted_index_vals_.initialize(ptr, inverted_index_vals_byte_size);
        ptr += inverted_index_vals_byte_size;

        if constexpr (use_wand) {
            max_score_in_dim_.initialize(ptr, max_score_in_dim_byte_size);
            ptr += max_score_in_dim_byte_size;
        }

        if constexpr (bm25) {
            bm25_params_->row_sums.initialize(ptr, row_sums_byte_size);
            ptr += row_sums_byte_size;
        }

        for (const auto& [idx, count] : idx_counts) {
            auto& plist_ids = inverted_index_ids_.emplace_back();
            auto plist_ids_byte_size = count * sizeof(typename decltype(inverted_index_ids_)::value_type::value_type);
            plist_ids.initialize(ptr, plist_ids_byte_size);
            ptr += plist_ids_byte_size;
        }
        for (const auto& [idx, count] : idx_counts) {
            auto& plist_vals = inverted_index_vals_.emplace_back();
            auto plist_vals_byte_size = count * sizeof(typename decltype(inverted_index_vals_)::value_type::value_type);
            plist_vals.initialize(ptr, plist_vals_byte_size);
            ptr += plist_vals_byte_size;
        }
        size_t dim_id = 0;
        for (const auto& [idx, count] : idx_counts) {
            dim_map_[idx] = dim_id;
            if constexpr (use_wand) {
                max_score_in_dim_.emplace_back(0.0f);
            }
            ++dim_id;
        }
        // in mmap mode, next_dim_id_ should never be used, but still assigning for consistency.
        next_dim_id_ = dim_id;

        return Status::success;
    }

    // Non zero drop ratio is only supported for static index, i.e. data should
    // include all rows that'll be added to the index.
    Status
    Train(const SparseRow<DType>* data, size_t rows) override {
        if constexpr (mmapped) {
            throw std::invalid_argument("mmapped InvertedIndex does not support Train");
        } else {
            return Status::success;
        }
    }

    Status
    Add(const SparseRow<DType>* data, size_t rows, int64_t dim) override {
        if constexpr (mmapped) {
            throw std::invalid_argument("mmapped InvertedIndex does not support Add");
        } else {
            auto current_rows = n_rows_internal_;
            if ((size_t)dim > max_dim_) {
                max_dim_ = dim;
            }

            if constexpr (bm25) {
                bm25_params_->row_sums.reserve(current_rows + rows);
            }
            for (size_t i = 0; i < rows; ++i) {
                add_row_to_index(data[i], current_rows + i);
            }
            n_rows_internal_ += rows;

            return Status::success;
        }
    }

    void
    Search(const SparseRow<DType>& query, size_t k, float drop_ratio_search, float* distances, label_t* labels,
           size_t refine_factor, const BitsetView& bitset, const DocValueComputer<float>& computer) const override {
        // initially set result distances to NaN and labels to -1
        std::fill(distances, distances + k, std::numeric_limits<float>::quiet_NaN());
        std::fill(labels, labels + k, -1);
        if (query.size() == 0) {
            return;
        }

        std::vector<DType> values(query.size());
        for (size_t i = 0; i < query.size(); ++i) {
            values[i] = std::abs(query[i].val);
        }
        auto q_threshold = get_threshold(values, drop_ratio_search);

        // if no data was dropped during search, no refinement is needed.
        if (drop_ratio_search == 0) {
            refine_factor = 1;
        }
        MaxMinHeap<float> heap(k * refine_factor);
        if constexpr (!use_wand) {
            search_brute_force(query, q_threshold, heap, bitset, computer);
        } else {
            search_wand(query, q_threshold, heap, bitset, computer);
        }

        if (refine_factor == 1) {
            collect_result(heap, distances, labels);
        } else {
            refine_and_collect(query, heap, k, distances, labels, computer);
        }
    }

    // Returned distances are inaccurate based on the drop_ratio.
    std::vector<float>
    GetAllDistances(const SparseRow<DType>& query, float drop_ratio_search, const BitsetView& bitset,
                    const DocValueComputer<float>& computer) const override {
        if (query.size() == 0) {
            return {};
        }
        std::vector<DType> values(query.size());
        for (size_t i = 0; i < query.size(); ++i) {
            values[i] = std::abs(query[i].val);
        }
        auto q_threshold = get_threshold(values, drop_ratio_search);
        auto distances = compute_all_distances(query, q_threshold, computer);
        if (!bitset.empty()) {
            for (size_t i = 0; i < distances.size(); ++i) {
                if (bitset.test(i)) {
                    distances[i] = 0.0f;
                }
            }
        }
        return distances;
    }

    float
    GetRawDistance(const label_t vec_id, const SparseRow<DType>& query,
                   const DocValueComputer<float>& computer) const override {
        float distance = 0.0f;

        for (size_t i = 0; i < query.size(); ++i) {
            auto [idx, val] = query[i];
            auto dim_id = dim_map_.find(idx);
            if (dim_id == dim_map_.end()) {
                continue;
            }
            auto& plist_ids = inverted_index_ids_[dim_id->second];
            auto it = std::lower_bound(plist_ids.begin(), plist_ids.end(), vec_id,
                                       [](const auto& x, table_t y) { return x < y; });
            if (it != plist_ids.end() && *it == vec_id) {
                distance += val * computer(inverted_index_vals_[dim_id->second][it - plist_ids.begin()],
                                           bm25 ? bm25_params_->row_sums.at(vec_id) : 0);
            }
        }

        return distance;
    }

    [[nodiscard]] size_t
    size() const override {
        size_t res = sizeof(*this);
        res += dim_map_.size() *
               (sizeof(typename decltype(dim_map_)::key_type) + sizeof(typename decltype(dim_map_)::mapped_type));

        if constexpr (mmapped) {
            return res + map_byte_size_;
        } else {
            res += sizeof(typename decltype(inverted_index_ids_)::value_type) * inverted_index_ids_.capacity();
            for (size_t i = 0; i < inverted_index_ids_.size(); ++i) {
                res += sizeof(typename decltype(inverted_index_ids_)::value_type::value_type) *
                       inverted_index_ids_[i].capacity();
            }
            res += sizeof(typename decltype(inverted_index_vals_)::value_type) * inverted_index_vals_.capacity();
            for (size_t i = 0; i < inverted_index_vals_.size(); ++i) {
                res += sizeof(typename decltype(inverted_index_vals_)::value_type::value_type) *
                       inverted_index_vals_[i].capacity();
            }
            if constexpr (use_wand) {
                res += sizeof(typename decltype(max_score_in_dim_)::value_type) * max_score_in_dim_.capacity();
            }
            return res;
        }
    }

    [[nodiscard]] size_t
    n_rows() const override {
        return n_rows_internal_;
    }

    [[nodiscard]] size_t
    n_cols() const override {
        return max_dim_;
    }

 private:
    // Given a vector of values, returns the threshold value.
    // All values strictly smaller than the threshold will be ignored.
    // values will be modified in this function.
    inline DType
    get_threshold(std::vector<DType>& values, float drop_ratio) const {
        // drop_ratio is in [0, 1) thus drop_count is guaranteed to be less
        // than values.size().
        auto drop_count = static_cast<size_t>(drop_ratio * values.size());
        if (drop_count == 0) {
            return 0;
        }
        auto pos = values.begin() + drop_count;
        std::nth_element(values.begin(), pos, values.end());
        return *pos;
    }

    std::vector<float>
    compute_all_distances(const SparseRow<DType>& q_vec, DType q_threshold,
                          const DocValueComputer<float>& computer) const {
        std::vector<float> scores(n_rows_internal_, 0.0f);
        for (size_t idx = 0; idx < q_vec.size(); ++idx) {
            auto [i, v] = q_vec[idx];
            if (v < q_threshold || i >= max_dim_) {
                continue;
            }
            auto dim_id = dim_map_.find(i);
            if (dim_id == dim_map_.end()) {
                continue;
            }
            auto& plist_ids = inverted_index_ids_[dim_id->second];
            auto& plist_vals = inverted_index_vals_[dim_id->second];
            // TODO: improve with SIMD
            for (size_t j = 0; j < plist_ids.size(); ++j) {
                auto doc_id = plist_ids[j];
                auto val = plist_vals[j];
                float val_sum = bm25 ? bm25_params_->row_sums.at(doc_id) : 0;
                scores[doc_id] += v * computer(val, val_sum);
            }
        }
        return scores;
    }

    template <typename DocIdFilter>
    struct Cursor {
     public:
        Cursor(const Vector<table_t>& plist_ids, const Vector<QType>& plist_vals, size_t num_vec, float max_score,
               float q_value, DocIdFilter filter)
            : plist_ids_(plist_ids),
              plist_vals_(plist_vals),
              plist_size_(plist_ids.size()),
              total_num_vec_(num_vec),
              max_score_(max_score),
              q_value_(q_value),
              filter_(filter) {
            skip_filtered_ids();
            update_cur_vec_id();
        }
        Cursor(const Cursor& rhs) = delete;
        Cursor(Cursor&& rhs) noexcept = default;

        void
        next() {
            ++loc_;
            skip_filtered_ids();
            update_cur_vec_id();
        }

        void
        seek(table_t vec_id) {
            while (loc_ < plist_size_ && plist_ids_[loc_] < vec_id) {
                ++loc_;
            }
            skip_filtered_ids();
            update_cur_vec_id();
        }

        QType
        cur_vec_val() const {
            return plist_vals_[loc_];
        }

        const Vector<table_t>& plist_ids_;
        const Vector<QType>& plist_vals_;
        const size_t plist_size_;
        size_t loc_ = 0;
        size_t total_num_vec_ = 0;
        float max_score_ = 0.0f;
        float q_value_ = 0.0f;
        DocIdFilter filter_;
        table_t cur_vec_id_ = 0;

     private:
        inline void
        update_cur_vec_id() {
            cur_vec_id_ = (loc_ >= plist_size_) ? total_num_vec_ : plist_ids_[loc_];
        }

        inline void
        skip_filtered_ids() {
            while (loc_ < plist_size_ && !filter_.empty() && filter_.test(plist_ids_[loc_])) {
                ++loc_;
            }
        }
    };  // struct Cursor

    // find the top-k candidates using brute force search, k as specified by the capacity of the heap.
    // any value in q_vec that is smaller than q_threshold and any value with dimension >= n_cols() will be ignored.
    // TODO: may switch to row-wise brute force if filter rate is high. Benchmark needed.
    template <typename DocIdFilter>
    void
    search_brute_force(const SparseRow<DType>& q_vec, DType q_threshold, MaxMinHeap<float>& heap, DocIdFilter& filter,
                       const DocValueComputer<float>& computer) const {
        auto scores = compute_all_distances(q_vec, q_threshold, computer);
        for (size_t i = 0; i < n_rows_internal_; ++i) {
            if ((filter.empty() || !filter.test(i)) && scores[i] != 0) {
                heap.push(i, scores[i]);
            }
        }
    }

    // any value in q_vec that is smaller than q_threshold will be ignored.
    template <typename DocIdFilter>
    void
    search_wand(const SparseRow<DType>& q_vec, DType q_threshold, MaxMinHeap<float>& heap, DocIdFilter& filter,
                const DocValueComputer<float>& computer) const {
        auto q_dim = q_vec.size();
        std::vector<std::shared_ptr<Cursor<DocIdFilter>>> cursors(q_dim);
        size_t valid_q_dim = 0;
        for (size_t i = 0; i < q_dim; ++i) {
            auto [idx, val] = q_vec[i];
            auto dim_id = dim_map_.find(idx);
            if (dim_id == dim_map_.end() || std::abs(val) < q_threshold) {
                continue;
            }
            auto& plist_ids = inverted_index_ids_[dim_id->second];
            auto& plist_vals = inverted_index_vals_[dim_id->second];
            cursors[valid_q_dim++] = std::make_shared<Cursor<DocIdFilter>>(
                plist_ids, plist_vals, n_rows_internal_, max_score_in_dim_[dim_id->second] * val, val, filter);
        }
        if (valid_q_dim == 0) {
            return;
        }
        cursors.resize(valid_q_dim);
        auto sort_cursors = [&cursors] {
            std::sort(cursors.begin(), cursors.end(), [](auto& x, auto& y) { return x->cur_vec_id_ < y->cur_vec_id_; });
        };
        sort_cursors();
        while (true) {
            float threshold = heap.full() ? heap.top().val : 0;
            float upper_bound = 0;
            size_t pivot;
            bool found_pivot = false;
            for (pivot = 0; pivot < valid_q_dim; ++pivot) {
                if (cursors[pivot]->loc_ >= cursors[pivot]->plist_size_) {
                    break;
                }
                upper_bound += cursors[pivot]->max_score_;
                if (upper_bound > threshold) {
                    found_pivot = true;
                    break;
                }
            }
            if (!found_pivot) {
                break;
            }
            table_t pivot_id = cursors[pivot]->cur_vec_id_;
            if (pivot_id == cursors[0]->cur_vec_id_) {
                float score = 0;
                for (auto& cursor : cursors) {
                    if (cursor->cur_vec_id_ != pivot_id) {
                        break;
                    }
                    float cur_vec_sum = bm25 ? bm25_params_->row_sums.at(cursor->cur_vec_id_) : 0;
                    score += cursor->q_value_ * computer(cursor->cur_vec_val(), cur_vec_sum);
                    cursor->next();
                }
                heap.push(pivot_id, score);
                sort_cursors();
            } else {
                size_t next_list = pivot;
                for (; cursors[next_list]->cur_vec_id_ == pivot_id; --next_list) {
                }
                cursors[next_list]->seek(pivot_id);
                for (size_t i = next_list + 1; i < valid_q_dim; ++i) {
                    if (cursors[i]->cur_vec_id_ >= cursors[i - 1]->cur_vec_id_) {
                        break;
                    }
                    std::swap(cursors[i], cursors[i - 1]);
                }
            }
        }
    }

    void
    refine_and_collect(const SparseRow<DType>& q_vec, MaxMinHeap<float>& inacc_heap, size_t k, float* distances,
                       label_t* labels, const DocValueComputer<float>& computer) const {
        std::vector<table_t> docids;
        MaxMinHeap<float> heap(k);

        docids.reserve(inacc_heap.size());
        while (!inacc_heap.empty()) {
            table_t u = inacc_heap.pop();
            docids.emplace_back(u);
        }

        DocIdFilterByVector filter(std::move(docids));
        if (use_wand) {
            search_wand(q_vec, 0, heap, filter, computer);
        } else {
            search_brute_force(q_vec, 0, heap, filter, computer);
        }
        collect_result(heap, distances, labels);
    }

    template <typename HeapType>
    void
    collect_result(HeapType& heap, float* distances, label_t* labels) const {
        int cnt = heap.size();
        for (auto i = cnt - 1; i >= 0; --i) {
            labels[i] = heap.top().id;
            distances[i] = heap.top().val;
            heap.pop();
        }
    }

    inline void
    add_row_to_index(const SparseRow<DType>& row, table_t vec_id) {
        [[maybe_unused]] float row_sum = 0;
        for (size_t j = 0; j < row.size(); ++j) {
            auto [idx, val] = row[j];
            if constexpr (bm25) {
                row_sum += val;
            }
            // Skip values equals to or close enough to zero(which contributes
            // little to the total IP score).
            if (val == 0) {
                continue;
            }
            auto dim_it = dim_map_.find(idx);
            if (dim_it == dim_map_.end()) {
                if constexpr (mmapped) {
                    throw std::runtime_error("unexpected vector dimension in mmaped InvertedIndex");
                }
                dim_it = dim_map_.insert({idx, next_dim_id_++}).first;
                inverted_index_ids_.emplace_back();
                inverted_index_vals_.emplace_back();
                if constexpr (use_wand) {
                    max_score_in_dim_.emplace_back(0.0f);
                }
            }
            inverted_index_ids_[dim_it->second].emplace_back(vec_id);
            inverted_index_vals_[dim_it->second].emplace_back(get_quant_val(val));
            if constexpr (use_wand) {
                auto score = static_cast<float>(val);
                if constexpr (bm25) {
                    score = bm25_params_->max_score_ratio * bm25_params_->wand_max_score_computer(val, row_sum);
                }
                max_score_in_dim_[dim_it->second] = std::max(max_score_in_dim_[dim_it->second], score);
            }
        }
        if constexpr (bm25) {
            bm25_params_->row_sums.emplace_back(row_sum);
        }
    }

    inline QType
    get_quant_val(DType val) const {
        if constexpr (!std::is_same_v<QType, DType>) {
            const DType max_val = static_cast<DType>(std::numeric_limits<QType>::max());
            if (val >= max_val) {
                return std::numeric_limits<QType>::max();
            } else if (val <= std::numeric_limits<QType>::min()) {
                return std::numeric_limits<QType>::min();
            } else {
                return static_cast<QType>(val);
            }
        } else {
            return val;
        }
    }

    // key is raw sparse vector dim/idx, value is the mapped dim/idx id in the index.
    std::unordered_map<table_t, uint32_t> dim_map_;

    // reserve, [], size, emplace_back
    Vector<Vector<table_t>> inverted_index_ids_;
    Vector<Vector<QType>> inverted_index_vals_;
    Vector<float> max_score_in_dim_;

    size_t n_rows_internal_ = 0;
    size_t max_dim_ = 0;
    uint32_t next_dim_id_ = 0;

    char* map_ = nullptr;
    size_t map_byte_size_ = 0;
    int map_fd_ = -1;

    struct BM25Params {
        float k1;
        float b;
        // row_sums is used to cache the sum of values of each row, which
        // corresponds to the document length of each doc in the BM25 formula.
        Vector<float> row_sums;

        // below are used only for WAND index.
        float max_score_ratio;
        DocValueComputer<float> wand_max_score_computer;

        BM25Params(float k1, float b, float avgdl, float max_score_ratio)
            : k1(k1),
              b(b),
              max_score_ratio(max_score_ratio),
              wand_max_score_computer(GetDocValueBM25Computer<float>(k1, b, avgdl)) {
        }
    };  // struct BM25Params

    std::unique_ptr<BM25Params> bm25_params_;

};  // class InvertedIndex

}  // namespace knowhere::sparse

#endif  // SPARSE_INVERTED_INDEX_H
