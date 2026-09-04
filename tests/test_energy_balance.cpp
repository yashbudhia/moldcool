// Discrete energy conservation and floating-point behaviour.
//
// The finite-volume scheme is conservative: internal fluxes cancel pairwise, so for every step
//     H^{n+1} - H^n = dt [ theta Q^{n+1} + (1 - theta) Q^n ]
// exactly in real arithmetic (H = sum M T, Q = heat flow through the walls). Measured in double
// the residual is round-off (explicit) or the CG tolerance (implicit). The same run in float
// shows the residual growing to ~1e-4, and naive vs compensated summation differ visibly.
// Also checks that K is symmetric (the property CG relies on) with random vectors.
#include <cstdio>
#include <random>
#include <vector>

#include "check.hpp"
#include "moldcool/core.hpp"
#include "moldcool/operator.hpp"
#include "moldcool/stepper.hpp"
#include "moldcool/sum.hpp"

using namespace moldcool;

template <class Real>
struct Result { double max_step_rel = 0, cum_rel = 0, naive_vs_kahan = 0, pairwise_vs_kahan = 0; };

template <class Real>
Result<Real> run(Real theta, int steps, std::FILE* csv, const char* label) {
    const auto mat = Materials<Real>::ABS();
    // plate 10 mm x 2 mm with a 1.5 mm x 4 mm rib, dx = 0.1 mm
    auto prob = make_plate_rib_2d<Real>(Real(10e-3), Real(2e-3), Real(1.5e-3), Real(4e-3), Real(0.1e-3), mat, Real(230), Real(50));
    Operator<Real> op(prob);
    const Real dt = (theta == Real(0)) ? Real(0.9) * op.explicit_dt_gershgorin() : Real(0.01);
    ThetaStepper<Real> st(op, theta, dt, Real(1e-12), 20000);
    std::vector<Real> T = prob.T0;
    Real t = 0;
    double H_prev = double(enthalpy(op, T));
    const double H0 = H_prev;
    double Q_prev = double(st.net_power(T, t));
    Result<Real> res;
    double cum = 0;
    for (int s = 1; s <= steps; ++s) {
        st.step(T, t);
        t += dt;
        const double H = double(enthalpy(op, T));
        const double Q = double(st.net_power(T, t));
        const double lhs = H - H_prev;
        const double rhs = double(dt) * (double(theta) * Q + (1.0 - double(theta)) * Q_prev);
        const double rel = std::abs(lhs - rhs) / std::abs(rhs);
        res.max_step_rel = std::max(res.max_step_rel, rel);
        cum += lhs - rhs;
        if (csv && (s % 10 == 0 || s == 1)) std::fprintf(csv, "%s,%d,%.6e,%.6e\n", label, s, rel, std::abs(cum) / std::abs(H - H0));
        H_prev = H; Q_prev = Q;
    }
    res.cum_rel = std::abs(cum) / std::abs(H_prev - H0);
    // Summation sensitivity on the final field.
    std::vector<Real> terms;
    for (int P = 0; P < op.grid.n(); ++P)
        if (op.mask[P] == Cell::Unknown) terms.push_back(op.mass[P] * T[P]);
    const double kahan = double(kahan_sum(terms));
    res.naive_vs_kahan = std::abs(double(naive_sum(terms)) - kahan) / std::abs(kahan);
    res.pairwise_vs_kahan = std::abs(double(pairwise_sum(terms)) - kahan) / std::abs(kahan);
    return res;
}

int main(int argc, char** argv) {
    const std::string out = moldcool_test::results_dir(argc, argv);
    std::FILE* csv = out.empty() ? nullptr : std::fopen((out + "energy_balance.csv").c_str(), "w");
    if (csv) std::fprintf(csv, "case,step,step_rel_residual,cumulative_rel_residual\n");

    // Symmetry of K: x . K y == y . K x
    {
        auto prob = make_plate_rib_2d<double>(10e-3, 2e-3, 1.5e-3, 4e-3, 0.2e-3, Materials<double>::ABS(), 230, 50);
        Operator<double> op(prob);
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> U(-1, 1);
        const int n = op.grid.n();
        std::vector<double> x(n), y(n), Kx(n), Ky(n);
        for (int P = 0; P < n; ++P) { x[P] = U(rng); y[P] = U(rng); }
        op.applyK(x.data(), Kx.data());
        op.applyK(y.data(), Ky.data());
        double xKy = 0, yKx = 0, scale = 0;
        for (int P = 0; P < n; ++P) { xKy += x[P] * Ky[P]; yKx += y[P] * Kx[P]; scale += std::abs(x[P] * Ky[P]); }
        std::printf("  symmetry |x.Ky - y.Kx| / sum|x.Ky| = %.3e\n", std::abs(xKy - yKx) / scale);
        CHECK(std::abs(xKy - yKx) / scale < 1e-13);
    }

    auto dE = run<double>(0.0, 400, csv, "double_explicit");
    auto dC = run<double>(0.5, 400, csv, "double_crank_nicolson");
    auto fE = run<float>(0.0f, 400, csv, "float_explicit");
    std::printf("  double explicit : max per-step rel residual %.3e, cumulative %.3e, naive-vs-Kahan %.3e, pairwise-vs-Kahan %.3e\n", dE.max_step_rel, dE.cum_rel, dE.naive_vs_kahan, dE.pairwise_vs_kahan);
    std::printf("  double CN       : max per-step rel residual %.3e, cumulative %.3e, naive-vs-Kahan %.3e, pairwise-vs-Kahan %.3e\n", dC.max_step_rel, dC.cum_rel, dC.naive_vs_kahan, dC.pairwise_vs_kahan);
    std::printf("  float explicit  : max per-step rel residual %.3e, cumulative %.3e, naive-vs-Kahan %.3e, pairwise-vs-Kahan %.3e\n", fE.max_step_rel, fE.cum_rel, fE.naive_vs_kahan, fE.pairwise_vs_kahan);
    CHECK(fE.pairwise_vs_kahan < fE.naive_vs_kahan);  // pairwise summation beats naive in float
    if (csv) std::fclose(csv);
    if (!out.empty()) {
        std::FILE* s = std::fopen((out + "energy_summary.csv").c_str(), "w");
        std::fprintf(s, "case,max_step_rel_residual,cumulative_rel_residual,naive_vs_kahan\n");
        std::fprintf(s, "double_explicit,%.3e,%.3e,%.3e\n", dE.max_step_rel, dE.cum_rel, dE.naive_vs_kahan);
        std::fprintf(s, "double_crank_nicolson,%.3e,%.3e,%.3e\n", dC.max_step_rel, dC.cum_rel, dC.naive_vs_kahan);
        std::fprintf(s, "float_explicit,%.3e,%.3e,%.3e\n", fE.max_step_rel, fE.cum_rel, fE.naive_vs_kahan);
        std::fclose(s);
    }
    CHECK(dE.max_step_rel < 1e-9);
    CHECK(dE.cum_rel < 1e-10);
    CHECK(dC.max_step_rel < 1e-7);
    CHECK(dC.cum_rel < 1e-8);
    CHECK(fE.max_step_rel > dE.max_step_rel * 1e3);  // float is measurably worse
    CHECK(fE.max_step_rel < 1e-2);
    return moldcool_test::finish("energy_balance");
}
