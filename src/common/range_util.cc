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

#include "knowhere/range_util.h"

#include <algorithm>
#include <cinttypes>
#include <queue>

#include "knowhere/log.h"

namespace knowhere {

///////////////////////////////////////////////////////////////////////////////
// for HNSW and DiskANN
void
FilterRangeSearchResultForOneNq(std::vector<float>& distances, std::vector<int64_t>& labels, const bool is_ip,
                                const float radius, const float range_filter) {
    KNOWHERE_THROW_IF_NOT_FMT(distances.size() == labels.size(), "distances' size %ld not equal to labels' size %ld",
                              distances.size(), labels.size());
    auto len = distances.size();
    size_t valid_cnt = 0;
    for (size_t i = 0; i < len; i++) {
        auto dist = distances[i];
        auto id = labels[i];
        if (distance_in_range(dist, radius, range_filter, is_ip)) {
            distances[valid_cnt] = dist;
            labels[valid_cnt] = id;
            valid_cnt++;
        }
    }
    if (valid_cnt != distances.size()) {
        distances.resize(valid_cnt);
        labels.resize(valid_cnt);
    }
}

void
GetRangeSearchResult(const std::vector<std::vector<float>>& result_distances,
                     const std::vector<std::vector<int64_t>>& result_labels, const bool is_ip, const int64_t nq,
                     const float radius, const float range_filter, float*& distances, int64_t*& labels, size_t*& lims) {
    KNOWHERE_THROW_IF_NOT_FMT(result_distances.size() == (size_t)nq, "result distances size %ld not equal to %" SCNd64,
                              result_distances.size(), nq);
    KNOWHERE_THROW_IF_NOT_FMT(result_labels.size() == (size_t)nq, "result labels size %ld not equal to %" SCNd64,
                              result_labels.size(), nq);

    lims = new size_t[nq + 1];
    lims[0] = 0;
    // all distances must be in range scope
    for (int64_t i = 0; i < nq; i++) {
        lims[i + 1] = lims[i] + result_distances[i].size();
    }

    size_t total_valid = lims[nq];
    LOG_KNOWHERE_DEBUG_ << "Range search: is_ip " << (is_ip ? "True" : "False") << ", radius " << radius
                        << ", range_filter " << range_filter << ", total result num " << total_valid;

    distances = new float[total_valid];
    labels = new int64_t[total_valid];

    for (auto i = 0; i < nq; i++) {
        std::copy_n(result_distances[i].data(), lims[i + 1] - lims[i], distances + lims[i]);
        std::copy_n(result_labels[i].data(), lims[i + 1] - lims[i], labels + lims[i]);
    }
}

namespace {
using ResultPair = std::pair<float, int64_t>;
}

/* Sort and return TOPK items as final range search result */
DataSetPtr
ReGenRangeSearchResult(DataSetPtr data_set, bool is_ip, int64_t nq, int64_t topk) {
    /**
     * nq: number of queries;
     * lims: the size of lims is nq + 1, lims[i+1] - lims[i] refers to the size of RangeSearch result queries[i]
     *      for example, the nq is 5. In the selected range,
     *      the size of RangeSearch result for each nq is [1, 2, 3, 4, 5],
     *      the lims will be [0, 1, 3, 6, 10, 15];
     * ids: the size of ids is lim[nq],
     *      {
     *        i(0,0), i(0,1), …, i(0,k0-1),
     *        i(1,0), i(1,1), …, i(1,k1-1),
     *        ... ...
     *        i(n-1,0), i(n-1,1), …, i(n-1,kn-1)
     *      }
     *      i(0,0), i(0,1), …, i(0,k0-1) means the ids of RangeSearch result queries[0], k0 equals lim[1] - lim[0];
     * dist: the size of ids is lim[nq],
     *      {
     *        d(0,0), d(0,1), …, d(0,k0-1),
     *        d(1,0), d(1,1), …, d(1,k1-1),
     *        ... ...
     *        d(n-1,0), d(n-1,1), …, d(n-1,kn-1)
     *      }
     *      d(0,0), d(0,1), …, d(0,k0-1) means the distances of RangeSearch result queries[0], k0 equals lim[1] -
     * lim[0];
     */
    auto lims = data_set->GetLims();
    auto id = data_set->GetIds();
    auto dist = data_set->GetDistance();

    // use p_id and p_dist to GenResultDataset after sorted
    auto p_id = new int64_t[topk * nq];
    auto p_dist = new float[topk * nq];
    std::fill_n(p_id, topk * nq, -1);
    std::fill_n(p_dist, topk * nq, std::numeric_limits<float>::max());

    /*
     *   get result for one nq
     *   IP:   1.0        range_filter     radius
     *          |------------+---------------|       min_heap   descending_order
     *                       |___ ___|
     *                           V
     *                          topk
     *
     *   L2:   0.0        range_filter     radius
     *          |------------+---------------|       max_heap   ascending_order
     *                       |___ ___|
     *                           V
     *                          topk
     */
    std::function<bool(const ResultPair&, const ResultPair&)> cmp = std::less<>();
    if (is_ip) {
        cmp = std::greater<>();
    }

    // The subscript of p_id and p_dist
    for (int i = 0; i < nq; i++) {
        std::priority_queue<ResultPair, std::vector<ResultPair>, decltype(cmp)> pq(cmp);
        auto capacity = std::min<int64_t>(lims[i + 1] - lims[i], topk);

        for (size_t j = lims[i]; j < lims[i + 1]; j++) {
            auto curr = ResultPair(dist[j], id[j]);
            if (pq.size() < capacity) {
                pq.push(curr);
            } else if (cmp(curr, pq.top())) {
                pq.pop();
                pq.push(curr);
            }
        }

        for (int j = capacity - 1; j >= 0; j--) {
            auto& node = pq.top();
            p_dist[i * topk + j] = node.first;
            p_id[i * topk + j] = node.second;
            pq.pop();
        }
    }
    return GenResultDataSet(nq, topk, p_id, p_dist);
}

}  // namespace knowhere
