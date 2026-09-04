// Spatial convergence on the 1D slab against the exact series solution.
//
// Unit material (alpha = 1), h = 1, walls at 0, initial temperature 1. Solution compared at
// Fourier number 0.1 at every cell centre, L-inf norm. Three runs per grid:
//   FTCS at fixed r = alpha dt / dx^2 = 0.25   -> dt ~ dx^2, total error O(dx^2)
//   Crank-Nicolson at dt = dx, plain            -> DOES NOT CONVERGE (documented failure)
//   Crank-Nicolson at dt = dx, Rannacher start  -> error O(dt^2 + dx^2) = O(dx^2)
//
// The plain CN run is kept on purpose. The initial condition is discontinuous at the walls
// (T = 1 inside, 0 on the face); with dt = dx the stiff modes see z = dt lambda ~ 4/dx and CN's
// amplification factor (1 - z/2)/(1 + z/2) is close to -1, so they never decay. Four backward-
// Euler half-steps at the start damp them and restore second order (see stepper.hpp).
#include <cstdio>
#include <vector>

#include "check.hpp"
#include "moldcool/analytic.hpp"
#include "moldcool/core.hpp"
#include "moldcool/operator.hpp"
#include "moldcool/stepper.hpp"

using namespace moldcool;

enum class Mode { FTCS, CN_plain, CN_rannacher };

static double run(int N, Mode mode, double t_end) {
    const auto mat = Materials<double>::Unit();
    auto prob = make_slab_1d<double>(1.0, N, mat, 1.0, 0.0);
    Operator<double> op(prob);
    const double dx = prob.grid.dx;
    double dt = (mode == Mode::FTCS) ? 0.25 * dx * dx / mat.alpha() : dx;
    const int steps = int(std::ceil(t_end / dt - 1e-12));
    dt = t_end / steps;  // land exactly on t_end
    std::vector<double> T = prob.T0;
    double t = 0;
    int done = 0;
    if (mode == Mode::CN_rannacher) { t = rannacher_startup(op, T, dt, 0.0, 1e-13); done = 2; }
    ThetaStepper<double> st(op, mode == Mode::FTCS ? 0.0 : 0.5, dt, 1e-13);
    for (int s = done; s < steps; ++s) { st.step(T, t); t += dt; }
    double err = 0;
    for (int i = 0; i < prob.grid.nx; ++i) {
        if (prob.mask[i] != Cell::Unknown) continue;
        const double exact = slab_theta<double>(prob.grid.xc(i), t_end, 1.0, mat.alpha());
        err = std::max(err, std::abs(T[i] - exact));
    }
    return err;
}

int main(int argc, char** argv) {
    const std::string out = moldcool_test::results_dir(argc, argv);
    const int Ns[] = {20, 40, 80, 160, 320};
    const double t_end = 0.1;
    std::vector<double> eF, eP, eR, hs;
    std::printf("  N      dx      FTCS r=0.25   order   CN plain    CN+Rannacher  order\n");
    std::FILE* csv = out.empty() ? nullptr : std::fopen((out + "slab_convergence.csv").c_str(), "w");
    if (csv) std::fprintf(csv, "N,dx,err_ftcs,order_ftcs,err_cn_plain,err_cn_rannacher,order_cn_rannacher\n");
    for (int N : Ns) {
        hs.push_back(1.0 / N);
        eF.push_back(run(N, Mode::FTCS, t_end));
        eP.push_back(run(N, Mode::CN_plain, t_end));
        eR.push_back(run(N, Mode::CN_rannacher, t_end));
        const size_t k = eF.size() - 1;
        const double pF = k ? moldcool_test::observed_order(eF[k - 1], eF[k], hs[k - 1], hs[k]) : 0;
        const double pR = k ? moldcool_test::observed_order(eR[k - 1], eR[k], hs[k - 1], hs[k]) : 0;
        std::printf("%4d  %.5f   %.4e   %6.3f   %.3e   %.4e   %6.3f\n", N, hs[k], eF[k], pF, eP[k], eR[k], pR);
        if (csv) std::fprintf(csv, "%d,%.10g,%.10e,%.6f,%.10e,%.10e,%.6f\n", N, hs[k], eF[k], pF, eP[k], eR[k], pR);
        if (k >= 2) {  // asymptotic range: the last refinements must show second order
            CHECK_RANGE(pF, 1.9, 2.1);
            CHECK_RANGE(pR, 1.9, 2.1);
        }
        if (k) { CHECK(eF[k] < eF[k - 1]); CHECK(eR[k] < eR[k - 1]); }
        CHECK(eP[k] > 0.1);  // the plain CN run must keep failing: this is the documented ringing
    }
    if (csv) std::fclose(csv);
    CHECK(eF.back() < 1e-5);
    // CN with dt = dx runs 320 explicit steps' worth of time per step at N = 320; its O(dt^2)
    // term dominates and the absolute error is ~2e-4 there. The claim under test is the order.
    CHECK(eR.back() < 1e-3);
    return moldcool_test::finish("slab_convergence");
}
