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

#include "distances_ref.h"

#include <cmath>

#include "knowhere/operands.h"
#include "xxhash.h"

namespace faiss {

float
fvec_inner_product_ref(const float* x, const float* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        res += x[i] * y[i];
    }
    return res;
}

float
fvec_L2sqr_ref(const float* x, const float* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

float
fvec_L1_ref(const float* x, const float* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += std::fabs(tmp);
    }
    return res;
}

float
fvec_Linf_ref(const float* x, const float* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        res = std::fmax(res, std::fabs(x[i] - y[i]));
    }
    return res;
}

float
fvec_norm_L2sqr_ref(const float* x, size_t d) {
    double res = 0;
    for (size_t i = 0; i < d; i++) {
        res += x[i] * x[i];
    }
    return res;
}

void
fvec_L2sqr_ny_ref(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr_ref(x, y, d);
        y += d;
    }
}

void
fvec_inner_products_ny_ref(float* ip, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        ip[i] = fvec_inner_product_ref(x, y, d);
        y += d;
    }
}

/// compute ny square L2 distance between x and a set of transposed contiguous
/// y vectors. squared lengths of y should be provided as well
void
fvec_L2sqr_ny_transposed_ref(float* __restrict dis, const float* __restrict x, const float* __restrict y,
                             const float* __restrict y_sqlen, size_t d, size_t d_offset, size_t ny) {
    float x_sqlen = 0;
    for (size_t j = 0; j < d; j++) {
        x_sqlen += x[j] * x[j];
    }

    for (size_t i = 0; i < ny; i++) {
        float dp = 0;
        for (size_t j = 0; j < d; j++) {
            dp += x[j] * y[i + j * d_offset];
        }

        dis[i] = x_sqlen + y_sqlen[i] - 2 * dp;
    }
}

/// compute ny square L2 distance between x and a set of contiguous y vectors
/// and return the index of the nearest vector.
/// return 0 if ny == 0.
size_t
fvec_L2sqr_ny_nearest_ref(float* __restrict distances_tmp_buffer, const float* __restrict x, const float* __restrict y,
                          size_t d, size_t ny) {
    fvec_L2sqr_ny_ref(distances_tmp_buffer, x, y, d, ny);

    size_t nearest_idx = 0;
    float min_dis = HUGE_VALF;

    for (size_t i = 0; i < ny; i++) {
        if (distances_tmp_buffer[i] < min_dis) {
            min_dis = distances_tmp_buffer[i];
            nearest_idx = i;
        }
    }

    return nearest_idx;
}

/// compute ny square L2 distance between x and a set of transposed contiguous
/// y vectors and return the index of the nearest vector.
/// squared lengths of y should be provided as well
/// return 0 if ny == 0.
size_t
fvec_L2sqr_ny_nearest_y_transposed_ref(float* __restrict distances_tmp_buffer, const float* __restrict x,
                                       const float* __restrict y, const float* __restrict y_sqlen, size_t d,
                                       size_t d_offset, size_t ny) {
    fvec_L2sqr_ny_transposed_ref(distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);

    size_t nearest_idx = 0;
    float min_dis = HUGE_VALF;

    for (size_t i = 0; i < ny; i++) {
        if (distances_tmp_buffer[i] < min_dis) {
            min_dis = distances_tmp_buffer[i];
            nearest_idx = i;
        }
    }

    return nearest_idx;
}

void
fvec_madd_ref(size_t n, const float* a, float bf, const float* b, float* c) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + bf * b[i];
    }
}

int
fvec_madd_and_argmin_ref(size_t n, const float* a, float bf, const float* b, float* c) {
    float vmin = 1e20;
    int imin = -1;

    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + bf * b[i];
        if (c[i] < vmin) {
            vmin = c[i];
            imin = i;
        }
    }
    return imin;
}

void
fvec_inner_product_batch_4_ref(const float* __restrict x, const float* __restrict y0, const float* __restrict y1,
                               const float* __restrict y2, const float* __restrict y3, const size_t d, float& dis0,
                               float& dis1, float& dis2, float& dis3) {
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    for (size_t i = 0; i < d; ++i) {
        d0 += x[i] * y0[i];
        d1 += x[i] * y1[i];
        d2 += x[i] * y2[i];
        d3 += x[i] * y3[i];
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}

void
fvec_L2sqr_batch_4_ref(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                       const size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    for (size_t i = 0; i < d; ++i) {
        const float q0 = x[i] - y0[i];
        const float q1 = x[i] - y1[i];
        const float q2 = x[i] - y2[i];
        const float q3 = x[i] - y3[i];
        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}

///////////////////////////////////////////////////////////////////////////////
// for hnsw sq, obsolete

int32_t
ivec_inner_product_ref(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    for (size_t i = 0; i < d; i++) {
        res += (int32_t)x[i] * y[i];
    }
    return res;
}

int32_t
ivec_L2sqr_ref(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    for (size_t i = 0; i < d; i++) {
        const int32_t tmp = (int32_t)x[i] - (int32_t)y[i];
        res += tmp * tmp;
    }
    return res;
}

///////////////////////////////////////////////////////////////////////////////
// fp16

float
fp16_vec_inner_product_ref(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        res += (float)x[i] * (float)y[i];
    }
    return res;
}

float
fp16_vec_L2sqr_ref(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        const float tmp = (float)x[i] - (float)y[i];
        res += tmp * tmp;
    }
    return res;
}

float
fp16_vec_norm_L2sqr_ref(const knowhere::fp16* x, size_t d) {
    double res = 0;
    for (size_t i = 0; i < d; i++) {
        res += (float)x[i] * (float)x[i];
    }
    return res;
}

void
fp16_vec_inner_product_batch_4_ref(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                                   const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0,
                                   float& dis1, float& dis2, float& dis3) {
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    for (size_t i = 0; i < d; ++i) {
        auto x_i = (float)x[i];
        d0 += x_i * (float)y0[i];
        d1 += x_i * (float)y1[i];
        d2 += x_i * (float)y2[i];
        d3 += x_i * (float)y3[i];
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}

void
fp16_vec_L2sqr_batch_4_ref(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                           const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0, float& dis1,
                           float& dis2, float& dis3) {
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    for (size_t i = 0; i < d; ++i) {
        auto x_i = (float)x[i];
        const float q0 = x_i - (float)y0[i];
        const float q1 = x_i - (float)y1[i];
        const float q2 = x_i - (float)y2[i];
        const float q3 = x_i - (float)y3[i];
        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}

///////////////////////////////////////////////////////////////////////////////
// bf16

float
bf16_vec_inner_product_ref(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        res += (float)x[i] * (float)y[i];
    }
    return res;
}

float
bf16_vec_L2sqr_ref(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        const float tmp = (float)x[i] - (float)y[i];
        res += tmp * tmp;
    }
    return res;
}

float
bf16_vec_norm_L2sqr_ref(const knowhere::bf16* x, size_t d) {
    double res = 0;
    for (size_t i = 0; i < d; i++) {
        res += (float)x[i] * (float)x[i];
    }
    return res;
}

void
bf16_vec_inner_product_batch_4_ref(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                                   const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
                                   float& dis1, float& dis2, float& dis3) {
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    for (size_t i = 0; i < d; ++i) {
        auto x_i = (float)x[i];
        d0 += x_i * (float)y0[i];
        d1 += x_i * (float)y1[i];
        d2 += x_i * (float)y2[i];
        d3 += x_i * (float)y3[i];
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}

void
bf16_vec_L2sqr_batch_4_ref(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                           const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0, float& dis1,
                           float& dis2, float& dis3) {
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    for (size_t i = 0; i < d; ++i) {
        auto x_i = (float)x[i];
        const float q0 = x_i - (float)y0[i];
        const float q1 = x_i - (float)y1[i];
        const float q2 = x_i - (float)y2[i];
        const float q3 = x_i - (float)y3[i];
        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}

///////////////////////////////////////////////////////////////////////////////
// int8

float
int8_vec_inner_product_ref(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    for (size_t i = 0; i < d; i++) {
        res += (int32_t)x[i] * (int32_t)y[i];
    }
    return (float)res;
}

float
int8_vec_L2sqr_ref(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    for (size_t i = 0; i < d; i++) {
        const int32_t tmp = (int32_t)x[i] - (int32_t)y[i];
        res += tmp * tmp;
    }
    return (float)res;
}

float
int8_vec_norm_L2sqr_ref(const int8_t* x, size_t d) {
    int32_t res = 0;
    for (size_t i = 0; i < d; i++) {
        res += (int32_t)x[i] * (int32_t)x[i];
    }
    return (float)res;
}

void
int8_vec_inner_product_batch_4_ref(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2,
                                   const int8_t* y3, const size_t d, float& dis0, float& dis1, float& dis2,
                                   float& dis3) {
    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    for (size_t i = 0; i < d; ++i) {
        auto x_i = (int32_t)x[i];
        d0 += x_i * (int32_t)y0[i];
        d1 += x_i * (int32_t)y1[i];
        d2 += x_i * (int32_t)y2[i];
        d3 += x_i * (int32_t)y3[i];
    }

    dis0 = (float)d0;
    dis1 = (float)d1;
    dis2 = (float)d2;
    dis3 = (float)d3;
}

void
int8_vec_L2sqr_batch_4_ref(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2, const int8_t* y3,
                           const size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    for (size_t i = 0; i < d; ++i) {
        auto x_i = (int32_t)x[i];
        const int32_t q0 = x_i - (int32_t)y0[i];
        const int32_t q1 = x_i - (int32_t)y1[i];
        const int32_t q2 = x_i - (int32_t)y2[i];
        const int32_t q3 = x_i - (int32_t)y3[i];
        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
    }

    dis0 = (float)d0;
    dis1 = (float)d1;
    dis2 = (float)d2;
    dis3 = (float)d3;
}

///////////////////////////////////////////////////////////////////////////////
// for cardinal

float
fvec_inner_product_bf16_patch_ref(const float* x, const float* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        res += x[i] * bf16_float(y[i]);
    }
    return res;
}

float
fvec_L2sqr_bf16_patch_ref(const float* x, const float* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        const float tmp = x[i] - bf16_float(y[i]);
        res += tmp * tmp;
    }
    return res;
}

void
fvec_inner_product_batch_4_bf16_patch_ref(const float* __restrict x, const float* __restrict y0,
                                          const float* __restrict y1, const float* __restrict y2,
                                          const float* __restrict y3, const size_t d, float& dis0, float& dis1,
                                          float& dis2, float& dis3) {
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    for (size_t i = 0; i < d; ++i) {
        d0 += x[i] * bf16_float(y0[i]);
        d1 += x[i] * bf16_float(y1[i]);
        d2 += x[i] * bf16_float(y2[i]);
        d3 += x[i] * bf16_float(y3[i]);
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}

void
fvec_L2sqr_batch_4_bf16_patch_ref(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                                  const size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    for (size_t i = 0; i < d; ++i) {
        const float q0 = x[i] - bf16_float(y0[i]);
        const float q1 = x[i] - bf16_float(y1[i]);
        const float q2 = x[i] - bf16_float(y2[i]);
        const float q3 = x[i] - bf16_float(y3[i]);
        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}

///////////////////////////////////////////////////////////////////////////////
// rabitq
float
fvec_masked_sum_ref(const float* q, const uint8_t* x, const size_t d) {
    float sum = 0;

    for (size_t i = 0; i < d; i++) {
        // extract i-th bit
        const uint8_t masker = (1 << (i % 8));
        const bool b_bit = ((x[i / 8] & masker) == masker);

        // accumulate dp
        sum += b_bit ? q[i] : 0;
    }

    return sum;
}

int
rabitq_dp_popcnt_ref(const uint8_t* q, const uint8_t* x, const size_t d, const size_t nb) {
    // this is the scheme for popcount
    const size_t di_8b = (d + 7) / 8;
    const size_t di_64b = (di_8b / 8) * 8;

    int dot = 0;
    for (size_t j = 0; j < nb; j++) {
        const uint8_t* q_j = q + j * di_8b;

        // process 64-bit popcounts
        int count_dot = 0;
        for (size_t i = 0; i < di_64b; i += 8) {
            const auto qv = *(const uint64_t*)(q_j + i);
            const auto xv = *(const uint64_t*)(x + i);
            count_dot += __builtin_popcountll(qv & xv);
        }

        // process leftovers
        for (size_t i = di_64b; i < di_8b; i++) {
            const auto qv = *(q_j + i);
            const auto xv = *(x + i);
            count_dot += __builtin_popcount(qv & xv);
        }

        dot += (count_dot << j);
    }

    return dot;
}

///////////////////////////////////////////////////////////////////////////////
// minhash
float
minhash_lsh_hit_ref(const char* x, const char* y, size_t dim, size_t mh_lsh_band) {
    size_t r = dim / mh_lsh_band;
    for (size_t i = 0; i < mh_lsh_band; i++) {
        const char* x_b = x + r * i;
        const char* y_b = y + r * i;
        size_t j = r;
        bool eq = true;
        while (j > 0) {
            if (*(const uint64_t*)(x_b) != *(const uint64_t*)(y_b)) {
                goto next_band;
            }
            j -= 8;
            x_b += 8;
            y_b += 8;
        }
        // the rest 8 element
        switch (j) {
            case 7:
                eq = eq & (x_b[6] == y_b[6]);
            case 6:
                eq = eq & (x_b[5] == y_b[5]);
            case 5:
                eq = eq & (x_b[4] == y_b[4]);
            case 4:
                eq = eq & (x_b[3] == y_b[3]);
            case 3:
                eq = eq & (x_b[2] == y_b[2]);
            case 2:
                eq = eq & (x_b[1] == y_b[1]);
            case 1:
                eq = eq & (x_b[0] == y_b[0]);
            case 0:;
        }
        if (eq == true) {
            return 1.0;
        }
    next_band:;
    }
    return 0.0;
}

int
u64_binary_search_eq_ref(const uint64_t* data, const size_t size, const uint64_t target) {
    int left = 0;
    int right = static_cast<int>(size) - 1;
    int result = -1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (data[mid] < target) {
            left = mid + 1;
        } else if (target < data[mid]) {
            right = mid - 1;
        } else {
            result = mid;
            right = mid - 1;
        }
    }
    return result;
}
int
u64_binary_search_ge_ref(const uint64_t* data, const size_t size, const uint64_t target) {
    int left = 0;
    int right = static_cast<int>(size) - 1;
    int result = -1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (data[mid] >= target) {
            result = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return result;
}

uint64_t
calculate_hash_ref(const char* data, size_t size) {
    return XXH3_64bits(data, size);
}

float
u32_jaccard_distance_ref(const char* x, const char* y, size_t element_length, size_t element_size) {
    float res = 0.0;
    auto u32_x = (const uint32_t*)x;
    auto u32_y = (const uint32_t*)y;
    for (size_t i = 0; i < element_length; i++) {
        res += (u32_x[i] == u32_y[i]);
    }
    return res / element_length;
}
void
u32_jaccard_distance_batch_4_ref(const char* x, const char* y0, const char* y1, const char* y2, const char* y3,
                                 size_t element_length, size_t element_size, float& dis0, float& dis1, float& dis2,
                                 float& dis3) {
    dis0 = dis1 = dis2 = dis3 = 0.0;
    auto u32_x = (const uint32_t*)x;
    auto u32_y0 = (const uint32_t*)y0;
    auto u32_y1 = (const uint32_t*)y1;
    auto u32_y2 = (const uint32_t*)y2;
    auto u32_y3 = (const uint32_t*)y3;
    for (size_t i = 0; i < element_length; i++) {
        dis0 += (u32_x[i] == u32_y0[i]);
        dis1 += (u32_x[i] == u32_y1[i]);
        dis2 += (u32_x[i] == u32_y2[i]);
        dis3 += (u32_x[i] == u32_y3[i]);
    }
    dis0 /= element_length;
    dis1 /= element_length;
    dis2 /= element_length;
    dis3 /= element_length;
    return;
}
float
u64_jaccard_distance_ref(const char* x, const char* y, size_t element_length, size_t element_size) {
    float res = 0.0;
    auto u64_x = (const uint64_t*)x;
    auto u64_y = (const uint64_t*)y;
    for (size_t i = 0; i < element_length; i++) {
        res += (u64_x[i] == u64_y[i]);
    }
    return res / element_length;
}
void
u64_jaccard_distance_batch_4_ref(const char* x, const char* y0, const char* y1, const char* y2, const char* y3,
                                 size_t element_length, size_t element_size, float& dis0, float& dis1, float& dis2,
                                 float& dis3) {
    dis0 = dis1 = dis2 = dis3 = 0.0;
    auto u64_x = (const uint64_t*)x;
    auto u64_y0 = (const uint64_t*)y0;
    auto u64_y1 = (const uint64_t*)y1;
    auto u64_y2 = (const uint64_t*)y2;
    auto u64_y3 = (const uint64_t*)y3;
    for (size_t i = 0; i < element_length; i++) {
        dis0 += (u64_x[i] == u64_y0[i]);
        dis1 += (u64_x[i] == u64_y1[i]);
        dis2 += (u64_x[i] == u64_y2[i]);
        dis3 += (u64_x[i] == u64_y3[i]);
    }
    dis0 /= element_length;
    dis1 /= element_length;
    dis2 /= element_length;
    dis3 /= element_length;
    return;
}

}  // namespace faiss
