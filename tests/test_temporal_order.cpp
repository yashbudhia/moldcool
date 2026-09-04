// Temporal order of the three theta schemes, isolated from spatial error.
//
// Rod of length 1 with adiabatic ends and initial condition cos(pi x). For the cell-centred
// finite-volume discretisation this is an exact eigenvector, so the semi-discrete solution is
// known in closed form: T_i(t) = cos(pi x_i) exp(-alpha lambda_h t). Comparing each integrator
// with THAT (not with the PDE solution) leaves only the time-integration error.
// Expected: forward Euler 1, backward Euler 1, Crank-Nicolson 2.
#include <cstdio>
#include <vector>

#include "check.hpp"
#include "moldcool/analytic.hpp"
#include "moldcool/core.hpp"
#include "moldcool/operator.hpp"
#include "moldcool/stepper.hpp"

using namespace moldcool;

static double run(int N, double theta, double dt, double t_end) {
    const auto mat = Materials<double>::Unit();
    auto prob = make_rod_neumann_1d<double>(1.0, N, mat, [](double x) { return std::cos(kPi * x); });
    Operator<double> op(prob);
    const int steps = int(std::llround(t_end / dt));
    ThetaStepper<double> st(op, theta, dt, 1e-14, 100000);
    std::vector<double> T = prob.T0;
    double t = 0;
    for (int s = 0; s < steps; ++s) { st.step(T, t); t += dt; }
    const double lam = rod_lambda_h<double>(1.0, prob.grid.dx);
    double err = 0;
    for (int i = 0; i < N; ++i)
        err = std::max(err, std::abs(T[i] - std::cos(kPi * prob.grid.xc(i)) * std::exp(-mat.alpha() * lam * t_end)));
    return err;
}

int main(int argc, char** argv) {
    const std::string out = moldcool_test::results_dir(argc, argv);
    const int N = 64;
    const double t_end = 0.5;
    const double dx = 1.0 / N;
    const double dt_explicit_max = dx * dx / 2.0;  // von Neumann limit for alpha = 1
    struct Scheme { const char* name; double theta; double dt0; double expected; };
    const Scheme schemes[] = {
        {"forward_euler", 0.0, 0.5 * dt_explicit_max, 1.0},
        {"backward_euler", 1.0, 0.05, 1.0},
        {"crank_nicolson", 0.5, 0.05, 2.0},
    };
    std::FILE* csv = out.empty() ? nullptr : std::fopen((out + "temporal_order.csv").c_str(), "w");
    if (csv) std::fprintf(csv, "scheme,dt,err,order\n");
    for (const auto& s : schemes) {
        std::printf("  %s\n", s.name);
        double prev_err = 0, prev_dt = 0;
        for (int k = 0; k < 4; ++k) {
            const double dt = s.dt0 / (1 << k);
            // keep t_end / dt an integer
            const double dt_adj = t_end / std::llround(t_end / dt);
            const double err = run(N, s.theta, dt_adj, t_end);
            const double p = k ? moldcool_test::observed_order(prev_err, err, prev_dt, dt_adj) : 0;
            std::printf("    dt = %.3e   err = %.4e   order = %6.3f\n", dt_adj, err, p);
            if (csv) std::fprintf(csv, "%s,%.10g,%.10e,%.6f\n", s.name, dt_adj, err, p);
            if (k >= 2) CHECK_RANGE(p, s.expected - 0.1, s.expected + 0.1);
            prev_err = err; prev_dt = dt_adj;
        }
    }
    if (csv) std::fclose(csv);
    // Spatial error of the semi-discrete operator itself: lambda_h vs pi^2, second order in dx.
    double prev = 0;
    for (int n : {16, 32, 64, 128}) {
        const double e = std::abs(rod_lambda_h<double>(1.0, 1.0 / n) - kPi * kPi);
        if (prev > 0) CHECK_RANGE(std::log(prev / e) / std::log(2.0), 1.9, 2.1);
        prev = e;
    }
    return moldcool_test::finish("temporal_order");
}
