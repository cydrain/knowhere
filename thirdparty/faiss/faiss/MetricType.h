/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_METRIC_TYPE_H
#define FAISS_METRIC_TYPE_H

#include <faiss/impl/platform_macros.h>

namespace faiss {

/// The metric space for vector comparison for Faiss indices and algorithms.
///
/// Most algorithms support both inner product and L2, with the flat
/// (brute-force) indices supporting additional metric types for vector
/// comparison.
enum MetricType {
    METRIC_INNER_PRODUCT = 0, ///< maximum inner product search
    METRIC_L2 = 1,            ///< squared L2 search
    METRIC_L1 = 2,            ///< L1 (aka cityblock)
    METRIC_Linf = 3,          ///< infinity distance
    METRIC_Lp = 4,            ///< L_p distance, p is given by a faiss::Index
                              /// metric_arg

    // Note: Faiss 1.7.4 defines METRIC_Jaccard=23,
    //   but Knowhere defines one as 5
    METRIC_Jaccard = 5,       ///< defined as: sum_i(min(a_i, b_i)) / sum_i(max(a_i, b_i))
                              ///< where a_i, b_i > 0
    METRIC_Hamming = 7,
    METRIC_Substructure = 8,   ///< Tversky case alpha = 0, beta = 1
    METRIC_Superstructure = 9, ///< Tversky case alpha = 1, beta = 0

    /// some additional metrics defined in scipy.spatial.distance
    METRIC_Canberra = 20,
    METRIC_BrayCurtis = 21,
    METRIC_JensenShannon = 22,
    /// Squared Eucliden distance, ignoring NaNs
    METRIC_NaNEuclidean = 24,
    /// abs(x | y): the distance to a hyperplane
    METRIC_ABS_INNER_PRODUCT = 25,
    METRIC_MinHash_Jaccard = 26,
};

/// all vector indices are this type
using idx_t = int64_t;

/// this function is used to distinguish between min and max indexes since
/// we need to support similarity and dis-similarity metrics in a flexible way
constexpr bool is_similarity_metric(MetricType metric_type) {
    return ((metric_type == METRIC_INNER_PRODUCT) ||
            (metric_type == METRIC_Jaccard));
}

} // namespace faiss

#endif
