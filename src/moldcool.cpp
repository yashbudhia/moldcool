// moldcool command line: runs the physical cases and writes CSV/VTK into an output directory.
//
//   moldcool slab      <outdir>   1D ABS plate profiles vs the series solution
//   moldcool rib       <outdir>   plate vs plate+rib in an isothermal mould, fields and history
//   moldcool conjugate <outdir>   plate in a steel block with coolant channels
//   moldcool all       <outdir>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "moldcool/analytic.hpp"
#include "moldcool/core.hpp"
#include "moldcool/io.hpp"
#include "moldcool/operator.hpp"
#include "moldcool/stepper.hpp"

using namespace moldcool;

static const double Tmelt = 230, Tmould = 50, Teject = 90;

static void run_slab(const std::string& out) {
    const auto abs_ = Materials<double>::ABS();
    const double h = 2e-3;
    const int N = 200;
    auto prob = make_slab_1d<double>(h, N, abs_, Tmelt, Tmould);
    Operator<double> op(prob);
    const double dt = 0.001;
    ThetaStepper<double> st(op, 0.5, dt, 1e-12);
    std::vector<double> T = prob.T0;
    const double tf = cooling_time_formula<double>(h, abs_.alpha(), Tmelt, Tmould, Teject);
    std::FILE* f = std::fopen((out + "/slab_profiles.csv").c_str(), "w");
    std::fprintf(f, "t,x_mm,T_solver,T_series\n");
    std::FILE* hist = std::fopen((out + "/slab_history.csv").c_str(), "w");
    std::fprintf(hist, "t,Tmax_solver,Tmid_series\n");
    const double snaps[] = {0.1, 0.5, 1.0, 2.0, tf};
    double t = 0;
    int next = 0;
    while (t < tf + 1e-9) {
        if (next < 5 && t >= snaps[next] - 0.5 * dt) {
            for (int i = 0; i < prob.grid.nx; ++i) {
                if (prob.mask[i] != Cell::Unknown) continue;
                const double x = prob.grid.xc(i);
                const double Ts = Tmould + (Tmelt - Tmould) * slab_theta<double>(x, t, h, abs_.alpha());
                std::fprintf(f, "%.4f,%.5f,%.6f,%.6f\n", t, x * 1e3, T[i], Ts);
            }
            ++next;
        }
        std::fprintf(hist, "%.4f,%.6f,%.6f\n", t, max_unknown(op, T), Tmould + (Tmelt - Tmould) * slab_theta<double>(h / 2, t, h, abs_.alpha()));
        st.step(T, t);
        t += dt;
    }
    std::fclose(f);
    std::fclose(hist);
    std::printf("slab: formula t_c = %.3f s (h = %.1f mm ABS)\n", tf, h * 1e3);
}

static void run_rib(const std::string& out) {
    const auto abs_ = Materials<double>::ABS();
    const double W = 20e-3, tp = 2e-3, tr = 1.5e-3, hr = 6e-3, dx = 0.1e-3, dt = 0.01;
    const double tf = cooling_time_formula<double>(tp, abs_.alpha(), Tmelt, Tmould, Teject);
    std::FILE* hist = std::fopen((out + "/rib_history.csv").c_str(), "w");
    std::fprintf(hist, "case,t,Tmax\n");
    double t_plate = 0, t_rib = 0;
    for (int with_rib = 0; with_rib < 2; ++with_rib) {
        auto prob = make_plate_rib_2d<double>(W, tp, tr, with_rib ? hr : 0.0, dx, abs_, Tmelt, Tmould);
        Operator<double> op(prob);
        ThetaStepper<double> st(op, 0.5, dt, 1e-10);
        std::vector<double> T = prob.T0;
        const char* name = with_rib ? "rib" : "plate";
        double t = 0;
        bool wrote_tf = false;
        double prev = max_unknown(op, T), tc = -1;
        while (tc < 0 || t < 2 * tf) {
            std::fprintf(hist, "%s,%.3f,%.6f\n", name, t, max_unknown(op, T));
            if (!wrote_tf && t >= tf - 0.5 * dt) {
                write_vtk(out + "/" + name + "_at_plate_tc.vtk", prob.grid, prob.mask, T);
                write_field_csv(out + "/" + name + "_at_plate_tc.csv", prob.grid, prob.mask, T);
                wrote_tf = true;
            }
            st.step(T, t);
            t += dt;
            const double cur = max_unknown(op, T);
            if (tc < 0 && cur < Teject) {
                tc = t - dt + dt * (prev - Teject) / (prev - cur);
                write_vtk(out + "/" + name + "_at_eject.vtk", prob.grid, prob.mask, T);
                write_field_csv(out + "/" + name + "_at_eject.csv", prob.grid, prob.mask, T);
            }
            prev = cur;
            if (t > 20 * tf) break;
        }
        (with_rib ? t_rib : t_plate) = tc;
    }
    std::fclose(hist);
    std::FILE* s = std::fopen((out + "/rib_summary.csv").c_str(), "w");
    std::fprintf(s, "case,t_eject_s\nformula_plate,%.4f\nplate_2d,%.4f\nplate_rib_2d,%.4f\n", tf, t_plate, t_rib);
    std::fclose(s);
    std::printf("rib: formula %.3f s, plate 2D %.3f s, plate+rib %.3f s (x%.2f)\n", tf, t_plate, t_rib, t_rib / t_plate);

    // DFM sweep: cooling time against rib thickness / wall thickness. Handbooks advise ribs of
    // 0.5 to 0.6 of the wall to limit sink marks and cycle time; this is what the solver says.
    std::FILE* sw = std::fopen((out + "/rib_sweep.csv").c_str(), "w");
    std::fprintf(sw, "rib_mm,rib_over_wall,t_eject_s,ratio_to_plate\n");
    for (double trib : {0.5e-3, 1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3, 3.0e-3}) {
        auto prob = make_plate_rib_2d<double>(W, tp, trib, hr, dx, abs_, Tmelt, Tmould);
        Operator<double> op(prob);
        ThetaStepper<double> st(op, 0.5, dt, 1e-10);
        std::vector<double> T = prob.T0;
        const double tc = cooling_time(st, op, prob, T, Teject).first;
        std::fprintf(sw, "%.2f,%.3f,%.4f,%.4f\n", trib * 1e3, trib / tp, tc, tc / t_plate);
        std::printf("  rib %.1f mm (%.2f of wall): %.3f s (x%.2f)\n", trib * 1e3, trib / tp, tc, tc / t_plate);
    }
    std::fclose(sw);
}

static void run_conjugate(const std::string& out) {
    const auto abs_ = Materials<double>::ABS();
    const auto steel = Materials<double>::P20();
    const double Wb = 40e-3, Hb = 40e-3, W = 20e-3, tp = 2e-3, dx = 0.2e-3, d_ch = 6e-3, dt = 0.01;
    // Channels 10 mm above and below the cavity mid-plane (1.7 diameters), 16 mm apart.
    std::vector<std::pair<double, double>> ch = {{12e-3, 10e-3}, {28e-3, 10e-3}, {12e-3, 30e-3}, {28e-3, 30e-3}};
    const double Tcool = Tmould, Tsteel0 = Tmould;  // same wall temperature as the isothermal case at t = 0
    auto prob = make_conjugate_2d<double>(Wb, Hb, W, tp, dx, ch, d_ch, abs_, steel, Tmelt, Tsteel0, Tcool);
    Operator<double> op(prob);
    ThetaStepper<double> st(op, 0.5, dt, 1e-10);
    std::vector<double> T = prob.T0;
    const double tf = cooling_time_formula<double>(tp, abs_.alpha(), Tmelt, Tmould, Teject);
    std::FILE* hist = std::fopen((out + "/conjugate_history.csv").c_str(), "w");
    std::fprintf(hist, "t,Tmax_polymer,Tmax_steel\n");
    double t = 0, prev = max_where(op, prob, T, abs_.k), tc = -1;
    while (t < 3 * tf) {
        std::fprintf(hist, "%.3f,%.6f,%.6f\n", t, max_where(op, prob, T, abs_.k), max_where(op, prob, T, steel.k));
        st.step(T, t);
        t += dt;
        const double cur = max_where(op, prob, T, abs_.k);
        if (tc < 0 && cur < Teject) {
            tc = t - dt + dt * (prev - Teject) / (prev - cur);
            write_vtk(out + "/conjugate_at_eject.vtk", prob.grid, prob.mask, T, &prob.k);
            write_field_csv(out + "/conjugate_at_eject.csv", prob.grid, prob.mask, T);
        }
        prev = cur;
    }
    std::fclose(hist);
    std::FILE* s = std::fopen((out + "/conjugate_summary.csv").c_str(), "w");
    std::fprintf(s, "case,t_eject_s\nformula_plate,%.4f\nconjugate_steel_block,%.4f\n", tf, tc);
    std::fclose(s);
    std::printf("conjugate: formula %.3f s, steel block with channels %.3f s (x%.2f)\n", tf, tc, tc / tf);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: moldcool {slab|rib|conjugate|all} <outdir>\n");
        return 2;
    }
    const std::string cmd = argv[1], out = argv[2];
    const bool all = cmd == "all";
    if (all || cmd == "slab") run_slab(out);
    if (all || cmd == "rib") run_rib(out);
    if (all || cmd == "conjugate") run_conjugate(out);
    return 0;
}
