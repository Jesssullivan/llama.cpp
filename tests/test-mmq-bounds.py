#!/usr/bin/env python3
"""Compile actual RDNA4 MMQ tables/helpers without CUDA or HIP headers."""

import argparse
import pathlib
import re
import subprocess


def extract(text, pattern):
    match = re.search(pattern, text, re.S)
    if not match:
        raise RuntimeError(f"MMQ source adapter could not match {pattern}")
    return match.group(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    cuda = root / "ggml/src/ggml-cuda"
    source = (cuda / "mmq.cuh").read_text()
    allocation = (cuda / "mmq.cu").read_text()
    if allocation.count("ggml_cuda_mmq_get_y_padding(src0->type, fallback, cc, prec_src1)") != 2:
        raise RuntimeError("Dense and routed allocations must use the checked padding helper")
    gathers = re.findall(r"ids_dst_shared\[j\] = ([^;]*ids_dst\[[^;]+);", source)
    if len(gathers) != 3:
        raise RuntimeError("Expected all three routed destination-ID gathers")
    if source.count("for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size)") != 2:
        raise RuntimeError("Thread-read oracle needs the two actual tile load loops")

    definitions = "\n".join(extract(source, rf"#define {name}[^\n]*") for name in
                            ("MMQ_ITER_K", "MMQ_ITER_K_FP4", "MMQ_TILE_NE_K", "MMQ_TILE_Y_K"))
    common = (root / "ggml/src/ggml-common.h").read_text()
    definitions += "\n" + "\n".join(extract(common, rf"#define {name}[^\n]*") for name in ("QI8_1", "QR8_1"))
    layout = extract(source, r"enum ggml_cuda_mmq_sram_layout \{.*?\n\};")
    config = extract(source, r"struct ggml_cuda_mmq_config \{.*?\n\};")
    cases = extract(source, r"#define CASE\(.*?(?=\n#include)")
    padding = extract(source, r"static __host__ size_t ggml_cuda_mmq_get_y_padding\(.*?\n\}")
    previous = extract(source, r"static __host__ int ggml_cuda_mmq_get_J_max\(.*?\n\}")
    gather_functions = "\n".join(
        f"int gather_{i}(checked_ids & ids_dst, int col_low, int col_diff, int jt, int J, int j) {{ return {expression}; }}"
        for i, expression in enumerate(gathers)
    )
    harness = r'''
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#define __host__
#define __device__
#define AMD_WMMA_AVAILABLE
inline bool amd_mfma_available(int) { return false; }
inline bool amd_wmma_available(int) { return true; }
inline bool turing_mma_available(int) { return false; }
void require(bool condition) { if (!condition) { std::fputs("MMQ bounds failure\n", stderr); std::abort(); } }
''' + definitions + "\n" + layout + "\n" + config + "\n" + cases + r'''
#include "mmq-config-rdna4.cuh"
#undef CASE
// RDNA4's actual host dispatch ignores cc and precision when selecting its table.
static ggml_cuda_mmq_config ggml_cuda_mmq_get_config(ggml_type type, int J, bool fallback, int, ggml_prec = GGML_PREC_Q8) {
    return ggml_cuda_mmq_get_config_rdna4(type, J, fallback);
}
''' + previous + "\n" + padding + r'''
struct checked_ids {
    int size;
    int reads = 0;
    int operator[](int index) { require(index >= 0 && index < size); ++reads; return index + 1; }
};
''' + gather_functions + r'''
int main() {
    size_t table_checks = 0, shape_checks = 0, dense_checks = 0, old_failures = 0, upstream_failures = 0, dense_failures = 0;
    const int words_per_block = MMQ_TILE_Y_K;
    for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
        for (bool fallback : {false, true}) {
            for (ggml_prec precision : {GGML_PREC_Q8, GGML_PREC_Q4}) {
                const size_t padding_bytes = ggml_cuda_mmq_get_y_padding(ggml_type(type), fallback, 0, precision);
                for (int J = 8; J <= 128; J += 8) {
                    const auto config = ggml_cuda_mmq_get_config(ggml_type(type), J, fallback, 0, precision);
                    if (config.type == GGML_TYPE_COUNT) { continue; }
                    // Enumerate the unmasked kernel loads, including the last thread group.
                    for (int first = 0; first < J * MMQ_TILE_Y_K; first += config.nthreads) {
                        for (int thread = 0; thread < config.nthreads; ++thread) {
                            require(size_t(first + thread + 1) * sizeof(int) <= padding_bytes);
                        }
                    }
                    ++table_checks;
                }
            }
        }
    }
    for (int K : {512, 2048}) {
        for (int T : {5, 8, 15, 16, 31, 32, 33, 64}) {
            for (int experts : {9, 256, 605}) {
                // Every token chooses 0..7, except the last token replaces 7 with the last expert.
                std::vector<int> counts(experts, 0);
                for (int token = 0; token < T; ++token) {
                    for (int id = 0; id < 8; ++id) {
                        ++counts[token == T - 1 && id == 7 ? experts - 1 : id];
                    }
                }
                require(counts.back() == 1);
                const int R = T * 8;
                const int ncols_opt = (R + experts - 1) / experts;
                for (bool fallback : {false, true}) {
                    for (ggml_prec precision : {GGML_PREC_Q8, GGML_PREC_Q4}) {
                        int J = 0, best_tiles = std::numeric_limits<int>::max();
                        ggml_cuda_mmq_config selected = ggml_cuda_mmq_get_config_rdna4(GGML_TYPE_COUNT, 0, fallback);
                        for (int candidate = 8; candidate <= 128 && best_tiles > 1; candidate += 8) {
                            const auto config = ggml_cuda_mmq_get_config(GGML_TYPE_Q4_K, candidate, fallback, 0, precision);
                            if (config.type == GGML_TYPE_COUNT) { continue; }
                            const int tiles = (ncols_opt + candidate - 1) / candidate;
                            if (tiles < best_tiles) { J = candidate; selected = config; best_tiles = tiles; }
                        }
                        require(J != 0);
                        const size_t payload_words = size_t(R) * (K / 128) * words_per_block;
                        const size_t padding_words = ggml_cuda_mmq_get_y_padding(GGML_TYPE_Q4_K, fallback, 0, precision) / sizeof(int);
                        size_t maximum_read = 0;
                        int col_low = 0;
                        for (int count : counts) {
                            for (int jt = 0; jt * J < count; ++jt) {
                                // Enumerate the kernel's two Y loads for every actual Q4_K K iteration.
                                for (int kb0 = 0; kb0 < K / 256; ++kb0) {
                                    for (int half = 0; half < 2; ++half) {
                                        const size_t base = (size_t(kb0 * 2 + half) * R + col_low + jt * J) * words_per_block;
                                        for (int first = 0; first < J * MMQ_TILE_Y_K; first += selected.nthreads) {
                                            for (int thread = 0; thread < selected.nthreads; ++thread) {
                                                maximum_read = std::max(maximum_read, base + first + thread);
                                            }
                                        }
                                    }
                                }
                                for (auto gather : {gather_0, gather_1, gather_2}) {
                                    checked_ids ids{R};
                                    for (int j = 0; j < J; ++j) {
                                        const int result = gather(ids, col_low, count, jt, J, j);
                                        require(result == (j < count - jt * J ? col_low + jt * J + j + 1 : 0));
                                    }
                                    require(ids.reads == std::min(J, count - jt * J));
                                }
                            }
                            col_low += count;
                        }
                        require(col_low == R && maximum_read < payload_words + padding_words);
                        for (int ne11 : {1, 8}) {
                            const size_t old_words = size_t(ggml_cuda_mmq_get_J_max(GGML_TYPE_Q4_K, fallback, 0, ne11)) * words_per_block;
                            old_failures += maximum_read >= payload_words + old_words;
                        }
                        const size_t upstream_words = size_t(ggml_cuda_mmq_get_J_max(GGML_TYPE_Q4_K, fallback, 0, T)) * words_per_block;
                        upstream_failures += maximum_read >= payload_words + upstream_words;
                        ++shape_checks;
                    }
                }
            }
        }
    }
    for (int K : {512, 2048}) {
        for (int T : {9, 10, 12, 15, 16, 31, 32, 33, 64}) {
            for (bool fallback : {false, true}) {
                for (ggml_prec precision : {GGML_PREC_Q8, GGML_PREC_Q4}) {
                    int J = 0, best_tiles = std::numeric_limits<int>::max();
                    ggml_cuda_mmq_config selected = ggml_cuda_mmq_get_config_rdna4(GGML_TYPE_COUNT, 0, fallback);
                    for (int candidate = 8; candidate <= 128 && best_tiles > 1; candidate += 8) {
                        const auto config = ggml_cuda_mmq_get_config(GGML_TYPE_Q4_K, candidate, fallback, 0, precision);
                        if (config.type == GGML_TYPE_COUNT) { continue; }
                        const int tiles = (T + candidate - 1) / candidate;
                        if (tiles < best_tiles) { J = candidate; selected = config; best_tiles = tiles; }
                    }
                    require(J != 0);
                    const size_t payload_words = size_t(T) * (K / 128) * words_per_block;
                    size_t maximum_read = 0;
                    for (int jt = 0; jt * J < T; ++jt) {
                        for (int kb0 = 0; kb0 < K / 256; ++kb0) {
                            for (int half = 0; half < 2; ++half) {
                                const size_t base = (size_t(kb0 * 2 + half) * T + jt * J) * words_per_block;
                                for (int first = 0; first < J * MMQ_TILE_Y_K; first += selected.nthreads) {
                                    for (int thread = 0; thread < selected.nthreads; ++thread) {
                                        maximum_read = std::max(maximum_read, base + first + thread);
                                    }
                                }
                            }
                        }
                    }
                    const size_t padding_words = ggml_cuda_mmq_get_y_padding(GGML_TYPE_Q4_K, fallback, 0, precision) / sizeof(int);
                    require(maximum_read < payload_words + padding_words);
                    const size_t old_words = size_t(ggml_cuda_mmq_get_J_max(GGML_TYPE_Q4_K, fallback, 0, T)) * words_per_block;
                    const bool old_failure = maximum_read >= payload_words + old_words;
                    require(old_failure == (T >= 9 && T <= 15));
                    dense_failures += old_failure;
                    ++dense_checks;
                }
            }
        }
    }
    require(old_failures > 0 && upstream_failures > 0);
    require(dense_failures > 0);
    std::printf("MMQ host bounds PASS: %zu table checks, %zu sparse shapes, %zu old failures, %zu ne12-only failures, %zu dense shapes, %zu old dense failures\n",
                table_checks, shape_checks, old_failures, upstream_failures, dense_checks, dense_failures);
}
'''
    args.build_dir.mkdir(parents=True, exist_ok=True)
    cpp = args.build_dir / "mmq-bounds.cpp"
    binary = args.build_dir / "mmq-bounds"
    cpp.write_text(harness)
    subprocess.run([args.cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I" + str(root / "ggml/include"), "-I" + str(root / "ggml/src"),
                    "-I" + str(cuda), str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
