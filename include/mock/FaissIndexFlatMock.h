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

#pragma once

#include "faiss/IndexFlat.h"
#include "faiss/MetricType.h"

namespace faiss {

struct IndexFlatMock : IndexFlat {
    explicit IndexFlatMock(idx_t d, MetricType metric = MetricType::METRIC_L2, bool is_cosine = false)
        : IndexFlat(d, metric, is_cosine){};

    void
    add(idx_t n, const float* x) override{};

    void
    search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
           const SearchParameters* params = nullptr) const override{};

    void
    range_search(idx_t n, const float* x, float radius, RangeSearchResult* result,
                 const SearchParameters* params = nullptr) const override{};

    void
    reconstruct(idx_t key, float* recons) const override{};

    //    void
    //    compute_distance_subset(idx_t n, const float* x, idx_t k, float* distances, const idx_t* labels) const {};
    //
    //    // get pointer to the floating point data
    //    float*
    //    get_xb() {
    //        return (float*)codes.data();
    //    }
    //    const float*
    //    get_xb() const {
    //        return (const float*)codes.data();
    //    }
    //
    //    float*
    //    get_norms() {
    //        return (float*)code_norms.data();
    //    }
    //    const float*
    //    get_norms() const {
    //        return (const float*)code_norms.data();
    //    }
    //
    //    IndexFlatMock() {
    //    }
    //
    //    FlatCodesDistanceComputer*
    //    get_FlatCodesDistanceComputer() const override;
    //
    //    /* The stanadlone codec interface (just memcopies in this case) */
    //    void
    //    sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    //
    //    void
    //    sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
    //
    //    size_t
    //    cal_size() const;
};
}  // namespace faiss
