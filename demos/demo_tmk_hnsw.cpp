/*
 * Minimal smoke test for IndexTMKFlat + IndexHNSW.
 *
 * The vectors are precomputed, normalized TMK signatures for the toy example
 * in WorldWill/TMK 相似度 Metric：定义 + 逐步算例.md.
 */

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexTMKFlat.h>

namespace {

std::vector<float> normalized_signature(
        const std::vector<float>& b_p4,
        const std::vector<float>& b_p2,
        const std::vector<float>& a_p4,
        const std::vector<float>& a_p2) {
    std::vector<float> x;
    x.insert(x.end(), b_p4.begin(), b_p4.end());
    x.insert(x.end(), b_p2.begin(), b_p2.end());
    x.insert(x.end(), a_p4.begin(), a_p4.end());
    x.insert(x.end(), a_p2.begin(), a_p2.end());

    float norm2 = 0.0f;
    for (float v : x) {
        norm2 += v * v;
    }
    const float norm = std::sqrt(norm2);
    for (float& v : x) {
        v /= norm;
    }
    return x;
}

} // namespace

int main() {
    const std::vector<float> periods = {4.0f, 2.0f};
    const int embedding_dim = 4;

    const auto q = normalized_signature(
            {1, -1, 1, -1}, {0, -2, 0, 0}, {1, 1, 1, -1}, {0, 0, 0, 0});
    const auto b1 = normalized_signature(
            {1, -1, -1, -1}, {2, 0, 0, -2}, {-1, -1, -1, 1}, {0, 0, 0, 0});
    const auto b2 = normalized_signature(
            {0, -2, 0, 0}, {1, 1, 1, -1}, {1, -1, 1, -1}, {0, 0, 0, 0});

    std::vector<float> xb;
    xb.insert(xb.end(), q.begin(), q.end());
    xb.insert(xb.end(), b1.begin(), b1.end());
    xb.insert(xb.end(), b2.begin(), b2.end());

    faiss::IndexTMKFlat exact(embedding_dim, periods, -2, 2, 1);
    exact.add(3, xb.data());

    std::vector<float> distances(3);
    std::vector<faiss::idx_t> labels(3);
    exact.search(1, q.data(), 3, distances.data(), labels.data());

    std::cout << "Exact TMK search:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  rank " << i << ": label=" << labels[i]
                  << " score=" << distances[i] << std::endl;
    }

    auto storage = new faiss::IndexTMKFlat(embedding_dim, periods, -2, 2, 1);
    faiss::IndexHNSW hnsw(storage, 2);
    hnsw.own_fields = true;
    hnsw.hnsw.efSearch = 16;
    hnsw.add(3, xb.data());
    hnsw.search(1, q.data(), 3, distances.data(), labels.data());

    std::cout << "HNSW TMK search:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  rank " << i << ": label=" << labels[i]
                  << " score=" << distances[i] << std::endl;
    }

    return labels[0] == 0 && labels[1] == 2 ? 0 : 1;
}
