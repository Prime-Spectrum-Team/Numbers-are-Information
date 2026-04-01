/**
 * C++20 PrimeSpec Benchmark
 * Measures: Cayley lookup, SpectralAddress computation, SA GCD (component-wise min)
 *
 * Compile: g++ -O3 -march=native -std=c++20 -o exp_hw_bench_cpp20 exp_hw_bench_cpp20.cpp
 * Run:     ./exp_hw_bench_cpp20 > ../results/exp_hw_bench_cpp20.json
 */

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

// ─── Classification constants ────────────────────────────────────────────────
static constexpr int N_CLASSES = 7;

// O(1) Cayley table: CAYLEY[c1][c2] = product class
static constexpr uint8_t CAYLEY[N_CLASSES][N_CLASSES] = {
    {0,1,2,3,4,5,6},  // S0
    {1,5,5,5,5,6,6},  // S1
    {2,5,5,5,5,6,6},  // S2
    {3,5,5,5,5,6,6},  // S3
    {4,5,5,5,5,6,6},  // S4
    {5,6,6,6,6,6,6},  // S5
    {6,6,6,6,6,6,6},  // S6
};

// ─── SpectralAddress ─────────────────────────────────────────────────────────
static constexpr int K = 7;
static constexpr uint32_t BASIS[K] = {2, 3, 5, 7, 11, 13, 17};

using SA = std::array<uint8_t, K>;

[[nodiscard]] inline SA spectral_address(uint32_t n) noexcept {
    SA sa{};
    for (int i = 0; i < K; ++i) {
        uint32_t p = BASIS[i];
        while (n % p == 0) {
            ++sa[i];
            n /= p;
        }
    }
    return sa;
}

[[nodiscard]] inline SA sa_gcd(const SA& a, const SA& b) noexcept {
    SA result{};
    for (int i = 0; i < K; ++i)
        result[i] = std::min(a[i], b[i]);
    return result;
}

// ─── Timing helper ───────────────────────────────────────────────────────────
using Clock = std::chrono::high_resolution_clock;
using Seconds = std::chrono::duration<double>;

struct BenchResult {
    std::string name;
    uint64_t    iterations;
    double      elapsed_s;
    double      ops_per_second;
};

// ─── Benchmark 1: Cayley lookup ──────────────────────────────────────────────
BenchResult bench_cayley(uint64_t iters) {
    // Prevent dead-code elimination with volatile accumulator
    volatile uint64_t sink = 0;

    // Precompute random class pairs
    constexpr int PAIR_BUF = 1 << 16;
    std::array<uint8_t, PAIR_BUF> ca{}, cb{};
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, N_CLASSES - 1);
    for (int i = 0; i < PAIR_BUF; ++i) {
        ca[i] = static_cast<uint8_t>(dist(rng));
        cb[i] = static_cast<uint8_t>(dist(rng));
    }

    auto t0 = Clock::now();
    for (uint64_t i = 0; i < iters; ++i) {
        sink += CAYLEY[ca[i & (PAIR_BUF-1)]][cb[i & (PAIR_BUF-1)]];
    }
    double elapsed = Seconds(Clock::now() - t0).count();
    (void)sink;
    return {"cayley_lookup", iters, elapsed, static_cast<double>(iters) / elapsed};
}

// ─── Benchmark 2: SpectralAddress computation ────────────────────────────────
BenchResult bench_sa_compute(uint64_t iters) {
    volatile uint64_t sink = 0;

    // Precompute random B-smooth-ish numbers in [1, 17^4]
    constexpr int BUF = 1 << 16;
    std::array<uint32_t, BUF> ns{};
    std::mt19937 rng(123);
    std::uniform_int_distribution<uint32_t> dist(1, 83521); // 17^4
    for (int i = 0; i < BUF; ++i)
        ns[i] = dist(rng);

    auto t0 = Clock::now();
    for (uint64_t i = 0; i < iters; ++i) {
        SA sa = spectral_address(ns[i & (BUF-1)]);
        sink += sa[0]; // prevent elimination
    }
    double elapsed = Seconds(Clock::now() - t0).count();
    (void)sink;
    return {"sa_computation", iters, elapsed, static_cast<double>(iters) / elapsed};
}

// ─── Benchmark 3: SA GCD (component-wise min) ────────────────────────────────
BenchResult bench_sa_gcd(uint64_t iters) {
    volatile uint64_t sink = 0;

    constexpr int BUF = 1 << 16;
    std::array<SA, BUF> sas{};
    std::mt19937 rng(456);
    std::uniform_int_distribution<uint32_t> dist(1, 83521);
    for (int i = 0; i < BUF; ++i)
        sas[i] = spectral_address(dist(rng));

    auto t0 = Clock::now();
    for (uint64_t i = 0; i < iters; ++i) {
        SA g = sa_gcd(sas[i & (BUF-1)], sas[(i+1) & (BUF-1)]);
        sink += g[0];
    }
    double elapsed = Seconds(Clock::now() - t0).count();
    (void)sink;
    return {"sa_gcd", iters, elapsed, static_cast<double>(iters) / elapsed};
}

// ─── JSON output ─────────────────────────────────────────────────────────────
std::string fmt_double(double v, int precision = 3) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.*f", precision, v);
    return buf;
}

int main() {
    // Warmup
    bench_cayley(1'000'000);
    bench_sa_compute(100'000);

    // Run benchmarks
    auto r1 = bench_cayley(1'000'000'000ULL);
    auto r2 = bench_sa_compute(10'000'000ULL);
    auto r3 = bench_sa_gcd(10'000'000ULL);

    // Collect system info
    // Note: hardware info injected by Python wrapper; here we output raw numbers
    auto to_M = [](double v) { return v / 1e6; };

    // Output JSON to stdout
    std::cout << "{\n";
    std::cout << "  \"experiment\": \"E_cpp20_benchmark\",\n";
    std::cout << "  \"compiler_flags\": \"-O3 -march=native -std=c++20\",\n";
    std::cout << "  \"benchmarks\": {\n";

    // Cayley
    std::cout << "    \"cayley_lookup\": {\n";
    std::cout << "      \"iterations\": " << r1.iterations << ",\n";
    std::cout << "      \"elapsed_seconds\": " << fmt_double(r1.elapsed_s, 6) << ",\n";
    std::cout << "      \"ops_per_second\": " << fmt_double(r1.ops_per_second, 0) << ",\n";
    std::cout << "      \"ops_per_second_M\": " << fmt_double(to_M(r1.ops_per_second), 1) << "\n";
    std::cout << "    },\n";

    // SA compute
    std::cout << "    \"sa_computation\": {\n";
    std::cout << "      \"iterations\": " << r2.iterations << ",\n";
    std::cout << "      \"elapsed_seconds\": " << fmt_double(r2.elapsed_s, 6) << ",\n";
    std::cout << "      \"ops_per_second\": " << fmt_double(r2.ops_per_second, 0) << ",\n";
    std::cout << "      \"ops_per_second_M\": " << fmt_double(to_M(r2.ops_per_second), 1) << ",\n";
    std::cout << "      \"basis\": [2,3,5,7,11,13,17],\n";
    std::cout << "      \"n_range_upper\": 83521\n";
    std::cout << "    },\n";

    // SA GCD
    std::cout << "    \"sa_gcd\": {\n";
    std::cout << "      \"iterations\": " << r3.iterations << ",\n";
    std::cout << "      \"elapsed_seconds\": " << fmt_double(r3.elapsed_s, 6) << ",\n";
    std::cout << "      \"ops_per_second\": " << fmt_double(r3.ops_per_second, 0) << ",\n";
    std::cout << "      \"ops_per_second_M\": " << fmt_double(to_M(r3.ops_per_second), 1) << ",\n";
    std::cout << "      \"description\": \"component-wise min of two SA vectors (O(K))\"\n";
    std::cout << "    }\n";

    std::cout << "  }\n";
    std::cout << "}\n";

    return 0;
}
