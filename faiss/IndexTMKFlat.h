/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <faiss/Index.h>

namespace faiss {

struct DistanceComputer;

/** Flat storage for precomputed TMK Fourier signatures.
 *
 * The input vector layout is:
 *
 *   [b(period0), b(period1), ..., b(periodF-1),
 *    a(period0), a(period1), ..., a(periodF-1)]
 *
 * where every b/a block has embedding_dim floats. Vectors should be
 * pre-normalized if cosine-scale TMK scores are desired.
 */
struct IndexTMKFlat : Index {
    idx_t embedding_dim = 0;
    std::vector<float> periods;
    std::vector<float> omegas;
    std::vector<float> offsets;
    std::vector<float> cos_table;
    std::vector<float> sin_table;
    std::vector<float> xb;

    IndexTMKFlat();

    IndexTMKFlat(
            idx_t embedding_dim,
            const std::vector<float>& periods,
            int offset_min,
            int offset_max,
            int offset_step);

    void add(idx_t n, const float* x) override;

    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    DistanceComputer* get_distance_computer() const override;

    const float* get_xb() const {
        return xb.data();
    }

    float* get_xb() {
        return xb.data();
    }

    size_t n_periods() const {
        return periods.size();
    }
};

} // namespace faiss
