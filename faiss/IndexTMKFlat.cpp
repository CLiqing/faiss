/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexTMKFlat.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {

namespace {

float dot(const float* x, const float* y, size_t n) {
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++) {
        acc += x[i] * y[i];
    }
    return acc;
}

struct TMKDistanceComputer : DistanceComputer {
    const IndexTMKFlat& index;
    const float* q = nullptr;

    explicit TMKDistanceComputer(const IndexTMKFlat& index) : index(index) {}

    void set_query(const float* x) override {
        q = x;
    }

    float operator()(idx_t i) override {
        FAISS_THROW_IF_NOT(q);
        FAISS_THROW_IF_NOT(i >= 0 && i < index.ntotal);
        return tmk_score(q, index.xb.data() + i * index.d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        FAISS_THROW_IF_NOT(i >= 0 && i < index.ntotal);
        FAISS_THROW_IF_NOT(j >= 0 && j < index.ntotal);
        return tmk_score(
                index.xb.data() + i * index.d, index.xb.data() + j * index.d);
    }

    float tmk_score(const float* x, const float* y) const {
        const size_t f_count = index.n_periods();
        const size_t dim = static_cast<size_t>(index.embedding_dim);
        const size_t a_offset = f_count * dim;
        std::vector<float> real(f_count);
        std::vector<float> imag(f_count);

        for (size_t fi = 0; fi < f_count; fi++) {
            const float* xb = x + fi * dim;
            const float* xa = x + a_offset + fi * dim;
            const float* yb = y + fi * dim;
            const float* ya = y + a_offset + fi * dim;

            real[fi] = dot(xa, ya, dim) + dot(xb, yb, dim);
            imag[fi] = dot(xa, yb, dim) - dot(xb, ya, dim);
        }

        float best = -std::numeric_limits<float>::infinity();

        for (size_t oi = 0; oi < index.offsets.size(); oi++) {
            float scan = 0.0f;
            for (size_t fi = 0; fi < f_count; fi++) {
                const size_t table_idx = fi * index.offsets.size() + oi;
                scan += real[fi] * index.cos_table[table_idx] +
                        imag[fi] * index.sin_table[table_idx];
            }
            best = std::max(best, scan);
        }
        return best;
    }
};

} // namespace

IndexTMKFlat::IndexTMKFlat() : Index(0, METRIC_INNER_PRODUCT) {}

IndexTMKFlat::IndexTMKFlat(
        idx_t embedding_dim_in,
        const std::vector<float>& periods_in,
        int offset_min,
        int offset_max,
        int offset_step)
        : Index(2 * embedding_dim_in * periods_in.size(), METRIC_INNER_PRODUCT),
          embedding_dim(embedding_dim_in),
          periods(periods_in) {
    FAISS_THROW_IF_NOT(embedding_dim > 0);
    FAISS_THROW_IF_NOT(!periods.empty());
    FAISS_THROW_IF_NOT(offset_step > 0);
    FAISS_THROW_IF_NOT(offset_min <= offset_max);

    constexpr float two_pi = 6.28318530717958647692f;
    omegas.reserve(periods.size());
    for (float period : periods) {
        FAISS_THROW_IF_NOT(period >= 0.0f);
        omegas.push_back(period == 0.0f ? 0.0f : two_pi / period);
    }

    for (int offset = offset_min; offset <= offset_max; offset += offset_step) {
        offsets.push_back(static_cast<float>(offset));
    }

    cos_table.resize(periods.size() * offsets.size());
    sin_table.resize(periods.size() * offsets.size());
    for (size_t fi = 0; fi < periods.size(); fi++) {
        for (size_t oi = 0; oi < offsets.size(); oi++) {
            const float angle = omegas[fi] * offsets[oi];
            const size_t table_idx = fi * offsets.size() + oi;
            cos_table[table_idx] = std::cos(angle);
            sin_table[table_idx] = std::sin(angle);
        }
    }

    is_trained = true;
}

void IndexTMKFlat::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(n >= 0);
    if (n == 0) {
        return;
    }
    const size_t old_size = xb.size();
    xb.resize(old_size + static_cast<size_t>(n) * d);
    std::memcpy(xb.data() + old_size, x, sizeof(float) * n * d);
    ntotal += n;
}

void IndexTMKFlat::reset() {
    xb.clear();
    ntotal = 0;
}

void IndexTMKFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* /* params */) const {
    FAISS_THROW_IF_NOT(k > 0);
    TMKDistanceComputer dc(*this);

    for (idx_t qi = 0; qi < n; qi++) {
        dc.set_query(x + qi * d);
        std::priority_queue<
                std::pair<float, idx_t>,
                std::vector<std::pair<float, idx_t>>,
                std::greater<std::pair<float, idx_t>>>
                heap;

        for (idx_t bi = 0; bi < ntotal; bi++) {
            const float score = dc(bi);
            if (static_cast<idx_t>(heap.size()) < k) {
                heap.emplace(score, bi);
            } else if (score > heap.top().first) {
                heap.pop();
                heap.emplace(score, bi);
            }
        }

        const idx_t base = qi * k;
        for (idx_t j = 0; j < k; j++) {
            distances[base + j] = -std::numeric_limits<float>::infinity();
            labels[base + j] = -1;
        }
        idx_t write = static_cast<idx_t>(heap.size());
        while (write > 0) {
            --write;
            distances[base + write] = heap.top().first;
            labels[base + write] = heap.top().second;
            heap.pop();
        }
    }
}

void IndexTMKFlat::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    std::memcpy(recons, xb.data() + key * d, sizeof(float) * d);
}

DistanceComputer* IndexTMKFlat::get_distance_computer() const {
    return new TMKDistanceComputer(*this);
}

} // namespace faiss
