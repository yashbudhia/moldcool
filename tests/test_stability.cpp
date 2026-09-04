// Explicit stability limit, checked on both sides of r = 1/2.
//
// For FTCS the amplification matrix G = I - dt M^{-1} K is symmetric with eigenvalues
// 1 - r lambda dx^2, lambda dx^2 in (0, 4]. The 2-norm of the solution is non-increasing iff
// r <= 1/2. On the slab the largest eigenvalue is within O(1/N^2) of 4/dx^2, so r = 0.51 must
// grow by a factor of about 1.04 per step and r = 0.49 must not grow at all.
// The Gershgorin bound implemented in Operator::explicit_dt_gershgorin is r <= 1/3 here
// (cells next to a wall have a larger diagonal); it is sufficient, not necessary, and the test
// records both numbers.
#include <cmath>
#include <cstdio>
#include <vector>

#include "check.hpp"
#include "moldcool/core.hpp"
#include "moldcool/operator.hpp"
#include "moldcool/stepper.hpp"

using namespace moldcool;

static double norm2(const Operator<double>& op, const std::vector<double>& T) {
    double s = 0;
    for (int P = 0; P < op.grid.n(); ++P)
        if (op.mask[P] == Cell::Unknown) s += T[P] * T[P];
    return std::sqrt(s);
}

int main(int argc, char** argv) {
    const std::string out = moldcool_test::results_dir(argc, argv);
    const int N = 200, steps = 3000;
    const auto mat = Materials<double>::Unit();
    std::FILE* csv = out.empty() ? nullptr : std::fopen((out + "stability.csv").c_str(), "w");
    if (csv) std::fprintf(csv, "r,step,norm_ratio\n");
    double ratio_stable = 0, ratio_unstable = 0;
    for (double r : {0.49, 0.51}) {
        auto prob = make_slab_1d<double>(1.0, N, mat, 1.0, 0.0);
        Operator<double> op(prob);
        const double dx = prob.grid.dx;
        const double dt = r * dx * dx / mat.alpha();
        ThetaStepper<double> st(op, 0.0, dt);
        std::vector<double> T = prob.T0;
        const double n0 = norm2(op, T);
        double t = 0, ratio = 1;
        for (int s = 1; s <= steps; ++s) {
            st.step(T, t);
            t += dt;
            ratio = norm2(op, T) / n0;
            if (csv && (s % 50 == 0 || s == 1)) std::fprintf(csv, "%.2f,%d,%.10e\n", r, s, ratio);
            if (!std::isfinite(ratio)) break;
        }
        std::printf("  r = %.2f : ||T||/||T0|| after %d steps = %.4e\n", r, steps, ratio);
        if (r < 0.5) ratio_stable = ratio; else ratio_unstable = ratio;
        if (r < 0.5) {
            std::printf("  Gershgorin dt = %.4e (r = %.4f), interior dt = %.4e (r = 0.5)\n",
                        op.explicit_dt_gershgorin(), op.explicit_dt_gershgorin() * mat.alpha() / (dx * dx),
                        op.explicit_dt_interior(prob));
            CHECK_RANGE(op.explicit_dt_gershgorin() * mat.alpha() / (dx * dx), 1.0 / 3 - 1e-9, 1.0 / 3 + 1e-9);
            CHECK_RANGE(op.explicit_dt_interior(prob) * mat.alpha() / (dx * dx), 0.5 - 1e-12, 0.5 + 1e-12);
        }
    }
    if (csv) std::fclose(csv);
    CHECK(ratio_stable <= 1.0 + 1e-12);
    CHECK(!std::isfinite(ratio_unstable) || ratio_unstable > 1e6);
    return moldcool_test::finish("stability");
}
