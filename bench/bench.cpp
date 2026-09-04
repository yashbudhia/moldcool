// CPU benchmark: explicit stencil throughput (serial and OpenMP) and one Crank-Nicolson step.
// Prints a single JSON line consumed by bench/check_perf.py.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "moldcool/core.hpp"
#include "moldcool/operator.hpp"
#include "moldcool/stepper.hpp"

using namespace moldcool;
using clk = std::chrono::steady_clock;

static double mlups_explicit(int N, int steps) {
    auto prob = make_unit_square_dirichlet<double>(N, Materials<double>::Unit(), [](double, double) { return 1.0; });
    Operator<double> op(prob);
    const double dt = 0.9 * op.explicit_dt_gershgorin();
    ThetaStepper<double> st(op, 0.0, dt);
    std::vector<double> T = prob.T0;
    st.step(T, 0.0);  // warm-up
    const auto t0 = clk::now();
    for (int s = 0; s < steps; ++s) st.step(T, s * dt);
    const double sec = std::chrono::duration<double>(clk::now() - t0).count();
    return double(op.n_unknown) * steps / sec / 1e6;
}

static double cn_step_ms(int N, int* iters) {
    auto prob = make_unit_square_dirichlet<double>(N, Materials<double>::Unit(), [](double, double) { return 1.0; });
    Operator<double> op(prob);
    const double dt = prob.grid.dx;  // r = N/2, far beyond the explicit limit
    ThetaStepper<double> st(op, 0.5, dt, 1e-10);
    std::vector<double> T = prob.T0;
    st.step(T, 0.0);
    const auto t0 = clk::now();
    st.step(T, dt);
    const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    *iters = st.last_cg().iterations;
    return ms;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 2048;
    const int steps = argc > 2 ? std::atoi(argv[2]) : 50;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    const double serial = mlups_explicit(N, steps);
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    const double par = threads > 1 ? mlups_explicit(N, steps) : serial;
    int iters = 0;
    const double cn = cn_step_ms(1024, &iters);
    std::printf("{\"N\":%d,\"steps\":%d,\"threads\":%d,\"explicit_serial_mlups\":%.2f,\"explicit_omp_mlups\":%.2f,"
                "\"cn_1024_step_ms\":%.2f,\"cn_1024_cg_iters\":%d}\n",
                N, steps, threads, serial, par, cn, iters);
    return 0;
}
