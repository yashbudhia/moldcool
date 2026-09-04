// Validation against the injection-moulding cooling-time formula.
//
// ABS plate, Tmelt 230 C, mould 50 C, ejection when the hottest point reaches 90 C. The handbook
// formula t_c = h^2/(pi^2 alpha) ln((4/pi)(Tmelt - Tmould)/(Teject - Tmould)) is the one-term
// series; the solver must reproduce it for a uniform plate. A ribbed plate must take longer than
// the plate alone (the formula has no way to know about the rib).
#include <cstdio>
#include <vector>

#include "check.hpp"
#include "moldcool/analytic.hpp"
#include "moldcool/core.hpp"
#include "moldcool/operator.hpp"
#include "moldcool/stepper.hpp"

using namespace moldcool;

int main(int argc, char** argv) {
    const std::string out = moldcool_test::results_dir(argc, argv);
    const auto abs_ = Materials<double>::ABS();
    const double Tmelt = 230, Tmould = 50, Teject = 90;
    std::FILE* csv = out.empty() ? nullptr : std::fopen((out + "cooling_slab.csv").c_str(), "w");
    if (csv) std::fprintf(csv, "h_mm,t_formula,t_series,t_ftcs,t_cn,rel_err_ftcs,rel_err_cn\n");
    std::printf("  ABS plate, alpha = %.3e m^2/s\n", abs_.alpha());
    std::printf("  h[mm]   formula[s]   series[s]   FTCS[s]     CN[s]      err_FTCS   err_CN\n");
    for (double h_mm : {1.0, 2.0, 3.0, 4.0}) {
        const double h = h_mm * 1e-3;
        const double tf = cooling_time_formula<double>(h, abs_.alpha(), Tmelt, Tmould, Teject);
        const double ts = cooling_time_series<double>(h, abs_.alpha(), Tmelt, Tmould, Teject);
        const int N = 100;
        double t_ftcs, t_cn;
        {
            auto prob = make_slab_1d<double>(h, N, abs_, Tmelt, Tmould);
            Operator<double> op(prob);
            const double dt = 0.25 * prob.grid.dx * prob.grid.dx / abs_.alpha();
            ThetaStepper<double> st(op, 0.0, dt);
            std::vector<double> T = prob.T0;
            t_ftcs = cooling_time(st, op, prob, T, Teject).first;
        }
        {
            auto prob = make_slab_1d<double>(h, N, abs_, Tmelt, Tmould);
            Operator<double> op(prob);
            const double dt = tf / 2000;
            ThetaStepper<double> st(op, 0.5, dt, 1e-12);
            std::vector<double> T = prob.T0;
            t_cn = cooling_time(st, op, prob, T, Teject).first;
        }
        const double eF = std::abs(t_ftcs - ts) / ts, eC = std::abs(t_cn - ts) / ts;
        std::printf("  %4.1f    %8.3f     %8.3f    %8.3f    %8.3f    %.2e   %.2e\n", h_mm, tf, ts, t_ftcs, t_cn, eF, eC);
        if (csv) std::fprintf(csv, "%.1f,%.6f,%.6f,%.6f,%.6f,%.3e,%.3e\n", h_mm, tf, ts, t_ftcs, t_cn, eF, eC);
        CHECK(std::abs(tf - ts) / ts < 1e-5);  // the one-term formula is that good at this Fo
        CHECK(eF < 5e-3);
        CHECK(eC < 5e-3);
    }
    if (csv) std::fclose(csv);

    // Plate 20 mm x 2 mm alone, then with a 1.5 mm x 6 mm rib, dx = 0.1 mm.
    double t_plate, t_rib;
    for (int with_rib = 0; with_rib < 2; ++with_rib) {
        auto prob = make_plate_rib_2d<double>(20e-3, 2e-3, 1.5e-3, with_rib ? 6e-3 : 0.0, 0.1e-3, abs_, Tmelt, Tmould);
        Operator<double> op(prob);
        ThetaStepper<double> st(op, 0.5, 0.01, 1e-10);
        std::vector<double> T = prob.T0;
        const double tc = cooling_time(st, op, prob, T, Teject).first;
        (with_rib ? t_rib : t_plate) = tc;
    }
    const double tf2 = cooling_time_formula<double>(2e-3, abs_.alpha(), Tmelt, Tmould, Teject);
    std::printf("  2D plate 20x2 mm: %.3f s (formula %.3f s);  with 1.5x6 mm rib: %.3f s  (x%.2f)\n", t_plate, tf2, t_rib, t_rib / t_plate);
    // The side walls are 10 thicknesses away and do not reach the centre in 5 s (Fourier number in
    // x is 1.7e-3), so the 2D plate must reproduce the 1D formula to discretisation accuracy.
    CHECK(std::abs(t_plate - tf2) / tf2 < 1e-2);
    // A 1.5 mm rib on a 2 mm plate lengthens cooling by roughly a quarter (measured 1.25x); the
    // exact factor is a result, not an input, so the check only demands a clear increase.
    CHECK(t_rib > 1.1 * t_plate);
    return moldcool_test::finish("cooling_time");
}
