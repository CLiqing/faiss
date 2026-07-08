/*
 * End-to-end TMK HNSW recall smoke test.
 *
 * Inputs are produced by demos/tmk_prepare_from_pdq.py.
 */

#include <faiss/IndexHNSW.h>
#include <faiss/IndexTMKFlat.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); i++) {
        const char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                cur.push_back('"');
                i++;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

std::vector<std::string> read_video_ids(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open videos csv: " + path);
    }
    std::string line;
    std::getline(in, line);
    const auto header = split_csv_line(line);
    auto it = std::find(header.begin(), header.end(), "video_id");
    if (it == header.end()) {
        throw std::runtime_error("videos csv missing video_id column");
    }
    const size_t video_col = std::distance(header.begin(), it);

    std::vector<std::string> ids;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto cols = split_csv_line(line);
        if (video_col >= cols.size()) {
            throw std::runtime_error("malformed videos csv row");
        }
        ids.push_back(cols[video_col]);
    }
    return ids;
}

std::vector<std::unordered_set<faiss::idx_t>> read_gt(
        const std::string& path,
        const std::unordered_map<std::string, faiss::idx_t>& id_to_row,
        size_t n) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open gt csv: " + path);
    }
    std::string line;
    std::getline(in, line);
    const auto header = split_csv_line(line);
    auto a_it = std::find(header.begin(), header.end(), "video_id_a");
    auto b_it = std::find(header.begin(), header.end(), "video_id_b");
    if (a_it == header.end() || b_it == header.end()) {
        throw std::runtime_error(
                "gt csv missing video_id_a/video_id_b columns");
    }
    const size_t a_col = std::distance(header.begin(), a_it);
    const size_t b_col = std::distance(header.begin(), b_it);

    std::vector<std::unordered_set<faiss::idx_t>> gt(n);
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto cols = split_csv_line(line);
        if (a_col >= cols.size() || b_col >= cols.size()) {
            throw std::runtime_error("malformed gt csv row");
        }
        auto ia = id_to_row.find(cols[a_col]);
        auto ib = id_to_row.find(cols[b_col]);
        if (ia == id_to_row.end() || ib == id_to_row.end()) {
            continue;
        }
        gt[ia->second].insert(ib->second);
        gt[ib->second].insert(ia->second);
    }
    return gt;
}

std::vector<float> read_f32(const std::string& path, size_t expected_count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open signature file: " + path);
    }
    std::vector<float> data(expected_count);
    in.read(reinterpret_cast<char*>(data.data()),
            expected_count * sizeof(float));
    if (static_cast<size_t>(in.gcount()) != expected_count * sizeof(float)) {
        throw std::runtime_error("signature file has unexpected size");
    }
    return data;
}

std::vector<float> parse_periods(const std::string& text) {
    std::vector<float> periods;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            periods.push_back(std::stof(item));
        }
    }
    return periods;
}

std::string arg_value(
        int argc,
        char** argv,
        const std::string& name,
        const std::string& def = "") {
    for (int i = 1; i + 1 < argc; i++) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return def;
}

int arg_int(int argc, char** argv, const std::string& name, int def) {
    const std::string v = arg_value(argc, argv, name);
    return v.empty() ? def : std::stoi(v);
}

} // namespace

int main(int argc, char** argv) {
    const std::string signatures_path = arg_value(argc, argv, "--signatures");
    const std::string videos_path = arg_value(argc, argv, "--videos");
    const std::string gt_path = arg_value(argc, argv, "--gt");
    const std::string periods_text = arg_value(
            argc, argv, "--periods", "0,64,128,256,512,1024,2048,4096");
    const int embedding_dim = arg_int(argc, argv, "--embedding-dim", 256);
    const int offset_min = arg_int(argc, argv, "--offset-min", -1200);
    const int offset_max = arg_int(argc, argv, "--offset-max", 1200);
    const int offset_step = arg_int(argc, argv, "--offset-step", 20);
    const int k = arg_int(argc, argv, "--k", 100);
    const int m = arg_int(argc, argv, "--M", 32);
    const int ef_construction = arg_int(argc, argv, "--efConstruction", 200);
    const int ef_search = arg_int(argc, argv, "--efSearch", 128);
    const int max_queries = arg_int(argc, argv, "--max-queries", 0);
    const int max_vectors = arg_int(argc, argv, "--max-vectors", 0);

    if (signatures_path.empty() || videos_path.empty() || gt_path.empty()) {
        std::cerr << "usage: demo_tmk_recall --signatures tmk_flat.f32 "
                  << "--videos tmk_videos.csv --gt gt_pairs.csv [options]\n";
        return 2;
    }

    const auto periods = parse_periods(periods_text);
    auto video_ids = read_video_ids(videos_path);
    if (max_vectors > 0 &&
        video_ids.size() > static_cast<size_t>(max_vectors)) {
        video_ids.resize(max_vectors);
    }
    std::unordered_map<std::string, faiss::idx_t> id_to_row;
    for (size_t i = 0; i < video_ids.size(); i++) {
        id_to_row[video_ids[i]] = static_cast<faiss::idx_t>(i);
    }
    const auto gt = read_gt(gt_path, id_to_row, video_ids.size());
    const size_t dim = 2 * periods.size() * embedding_dim;
    auto signatures = read_f32(signatures_path, video_ids.size() * dim);

    auto storage = new faiss::IndexTMKFlat(
            embedding_dim, periods, offset_min, offset_max, offset_step);
    faiss::IndexHNSW index(storage, m);
    index.own_fields = true;
    index.hnsw.efConstruction = ef_construction;
    index.hnsw.efSearch = ef_search;
    std::cout << "building HNSW: videos=" << video_ids.size() << " dim=" << dim
              << " M=" << m << " efConstruction=" << ef_construction
              << " efSearch=" << ef_search << std::endl;
    const auto build_start = std::chrono::steady_clock::now();
    index.add(video_ids.size(), signatures.data());
    const auto build_end = std::chrono::steady_clock::now();
    std::cout << "build_seconds="
              << std::chrono::duration<double>(build_end - build_start).count()
              << std::endl;

    std::vector<float> distances(k + 1);
    std::vector<faiss::idx_t> labels(k + 1);

    double recall_sum = 0.0;
    size_t query_count = 0;
    size_t hit_queries = 0;
    const auto search_start = std::chrono::steady_clock::now();
    for (size_t qi = 0; qi < video_ids.size(); qi++) {
        if (gt[qi].empty()) {
            continue;
        }
        if (max_queries > 0 &&
            query_count >= static_cast<size_t>(max_queries)) {
            break;
        }
        index.search(
                1,
                signatures.data() + qi * dim,
                k + 1,
                distances.data(),
                labels.data());
        size_t hits = 0;
        for (int rank = 0; rank < k + 1; rank++) {
            const faiss::idx_t label = labels[rank];
            if (label < 0 || static_cast<size_t>(label) == qi) {
                continue;
            }
            if (gt[qi].count(label)) {
                hits++;
            }
        }
        recall_sum += static_cast<double>(hits) / gt[qi].size();
        if (hits > 0) {
            hit_queries++;
        }
        query_count++;
    }
    const auto search_end = std::chrono::steady_clock::now();

    std::cout << "videos=" << video_ids.size() << "\n";
    std::cout << "queries=" << query_count << "\n";
    std::cout << "recall@" << k << "="
              << (query_count ? recall_sum / query_count : 0.0) << "\n";
    std::cout << "query_hit_rate@" << k << "="
              << (query_count ? static_cast<double>(hit_queries) / query_count
                              : 0.0)
              << "\n";
    std::cout
            << "search_seconds="
            << std::chrono::duration<double>(search_end - search_start).count()
            << "\n";
    return 0;
}
