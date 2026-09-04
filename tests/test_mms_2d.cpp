// Method of manufactured solutions in 2D.
//
// T = exp(-t) sin(pi x) sin(pi y) on the unit square, alpha = 1, source f = (2 pi^2 - 1) T so
// that T satisfies the forced heat equation exactly. Dirichlet data is zero. Time step tied to
// the grid, dt = dx, which is 2N times the explicit limit (the reason to go implicit).
//
// Crank-Nicolson: error vs the exact solution is O(dt^2 + dx^2) = O(dx^2), observed order 2.
// Backward Euler: its O(dt) error and the O(dx^2) spatial error are comparable on these grids,
// so the order against the exact solution sits between 1 and 2 (recorded, not asserted). To
// isolate the temporal term, BE is also compared with CN on the identical grid and time step:
// both share the same spatial error, and the difference is C dt + O(dt^2), observed order 1.
#include <cstdio>
#include <vector>

#include "check.hpp"
#include "moldcool/analytic.hpp"
#include "moldcool/core.hpp"
#include "moldcool/operator.hpp"
#include "moldcool/stepper.hpp"

using namespace moldcool;

static std::vector<double> solve(int N, double theta, double t_end, int* cg_iters, Problem<double>* out_prob) {
    const auto mat = Materials<double>::Unit();
    auto prob = make_unit_square_dirichlet<double>(N, mat, [](double x, double y) { return mms_T<double>(x, y, 0.0); });
    Operator<double> op(prob);
    const double dx = prob.grid.dx;
    const int steps = int(std::llround(t_end / dx));
    const double dt = t_end / steps;
    ThetaStepper<double> st(op, theta, dt, 1e-13, 100000);
    const double alpha = mat.alpha();
    SourceFn<double> f = [alpha](double x, double y, double t) { return mms_source<double>(x, y, t, alpha); };
    std::vector<double> T = prob.T0;
    double t = 0;
    int iters = 0;
    for (int s = 0; s < steps; ++s) { st.step(T, t, &f); t += dt; iters = std::max(iters, st.last_cg().iterations); }
    if (cg_iters) *cg_iters = iters;
    if (out_prob) *out_prob = prob;
    return T;
}

static double linf_vs_exact(const Problem<double>& prob, const std::vector<double>& T, double t_end) {
    double err = 0;
    for (int j = 0; j < prob.grid.ny; ++j)
        for (int i = 0; i < prob.grid.nx; ++i) {
            const int P = prob.grid.idx(i, j);
            if (prob.mask[P] != Cell::Unknown) continue;
            err = std::max(err, std::abs(T[P] - mms_T<double>(prob.grid.xc(i), prob.grid.yc(j), t_end)));
        }
    return err;
}

static double linf_diff(const Problem<double>& prob, const std::vector<double>& A, const std::vector<double>& B) {
    double d = 0;
    for (int P = 0; P < prob.grid.n(); ++P)
        if (prob.mask[P] == Cell::Unknown) d = std::max(d, std::abs(A[P] - B[P]));
    return d;
}

int main(int argc, char** argv) {
    const std::string out = moldcool_test::results_dir(argc, argv);
    // t_end = 0.5 gives 5 to 80 steps across the grids; with fewer steps the O(dt) term of
    // backward Euler is not yet the leading one and the measured order drifts.
    const double t_end = 0.5;
    std::FILE* csv = out.empty() ? nullptr : std::fopen((out + "mms2d_convergence.csv").c_str(), "w");
    if (csv) std::fprintf(csv, "N,dx,err_cn,order_cn,cg_iters_cn,err_be,order_be,be_minus_cn,order_be_minus_cn\n");
    std::printf("     N   CN err      order   CG   BE err      order   |BE-CN|     order\n");
    double pc = 0, pb = 0, pd = 0, e_c = 0, e_b = 0, e_d = 0, h_prev = 0;
    int k = 0;
    // N chosen so that t_end / dx is an integer: dt halves exactly when dx halves.
    for (int N : {10, 20, 40, 80, 160}) {
        int it_c = 0, it_b = 0;
        Problem<double> prob(Grid<double>{});
        const auto Tc = solve(N, 0.5, t_end, &it_c, &prob);
        const auto Tb = solve(N, 1.0, t_end, &it_b, nullptr);
        const double h = 1.0 / N;
        const double ec = linf_vs_exact(prob, Tc, t_end), eb = linf_vs_exact(prob, Tb, t_end), ed = linf_diff(prob, Tb, Tc);
        if (k) {
            pc = moldcool_test::observed_order(e_c, ec, h_prev, h);
            pb = moldcool_test::observed_order(e_b, eb, h_prev, h);
            pd = moldcool_test::observed_order(e_d, ed, h_prev, h);
        }
        std::printf("  %4d   %.3e   %5.3f   %3d  %.3e   %5.3f   %.3e   %5.3f\n", N, ec, pc, it_c, eb, pb, ed, pd);
        if (csv) std::fprintf(csv, "%d,%.10g,%.10e,%.6f,%d,%.10e,%.6f,%.10e,%.6f\n", N, h, ec, pc, it_c, eb, pb, ed, pd);
        if (k >= 2) { CHECK_RANGE(pc, 1.85, 2.15); CHECK_RANGE(pd, 0.9, 1.1); }
        if (k) { CHECK(ec < e_c); CHECK(eb < e_b); CHECK(eb > ec); }
        e_c = ec; e_b = eb; e_d = ed; h_prev = h; ++k;
    }
    if (csv) std::fclose(csv);
    return moldcool_test::finish("mms_2d");
}
