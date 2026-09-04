// Minimal test helpers. Each test is a plain executable; the exit code is the failure count.
#pragma once

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

namespace moldcool_test {

inline int failures = 0;

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "  CHECK FAILED: %s   (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++moldcool_test::failures;                                                     \
        }                                                                                  \
    } while (0)

#define CHECK_RANGE(x, lo, hi)                                                                          \
    do {                                                                                                \
        const double _v = (x);                                                                          \
        if (!((_v) >= (lo) && (_v) <= (hi))) {                                                          \
            std::fprintf(stderr, "  CHECK FAILED: %s = %.6g not in [%g, %g]   (%s:%d)\n", #x, _v, double(lo), \
                         double(hi), __FILE__, __LINE__);                                               \
            ++moldcool_test::failures;                                                                  \
        }                                                                                               \
    } while (0)

// Observed order of accuracy from two errors at two resolutions: p = log(e1/e2) / log(h1/h2).
inline double observed_order(double e1, double e2, double h1, double h2) {
    return std::log(e1 / e2) / std::log(h1 / h2);
}

// Results directory passed as argv[1]; empty string disables CSV output.
inline std::string results_dir(int argc, char** argv) { return argc > 1 ? std::string(argv[1]) + "/" : std::string(); }

inline int finish(const char* name) {
    if (failures == 0) std::printf("[PASS] %s\n", name);
    else std::printf("[FAIL] %s: %d check(s) failed\n", name, failures);
    return failures;
}

}  // namespace moldcool_test
