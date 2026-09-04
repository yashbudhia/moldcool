// Cross-check of the hand-written matrix-free CG against Eigen's sparse solvers.
//
// The implicit matrix A = M/dt + theta K is assembled explicitly from the stencil coefficients,
// solved with Eigen's SimplicialLDLT (direct) and Eigen's ConjugateGradient, and compared to
// moldcool's solution of the same right-hand side.
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <cstdio>
#include <vector>

#include "check.hpp"
#include "moldcool/cg.hpp"
#include "moldcool/core.hpp"
#include "moldcool/operator.hpp"

using namespace moldcool;

int main(int, char**) {
    auto prob = make_plate_rib_2d<double>(10e-3, 2e-3, 1.5e-3, 4e-3, 0.1e-3, Materials<double>::ABS(), 230, 50);
    Operator<double> op(prob);
    const double theta = 0.5, dt = 0.05;
    ImplicitSystem<double> sys(op, theta, dt);
    const int n = op.grid.n(), nx = op.grid.nx;

    // Index map: grid cell -> unknown number
    std::vector<int> id(n, -1);
    int m = 0;
    for (int P = 0; P < n; ++P) if (op.mask[P] == Cell::Unknown) id[P] = m++;

    std::vector<Eigen::Triplet<double>> trip;
    for (int P = 0; P < n; ++P) {
        if (id[P] < 0) continue;
        trip.emplace_back(id[P], id[P], op.mass[P] / dt + theta * op.aP(P));
        if (P + 1 < n && id[P + 1] >= 0 && (P % nx) + 1 < nx) trip.emplace_back(id[P], id[P + 1], -theta * op.aE[P]);
        if (P - 1 >= 0 && id[P - 1] >= 0 && (P % nx) > 0)     trip.emplace_back(id[P], id[P - 1], -theta * op.aE[P - 1]);
        if (P + nx < n && id[P + nx] >= 0)                    trip.emplace_back(id[P], id[P + nx], -theta * op.aN[P]);
        if (P - nx >= 0 && id[P - nx] >= 0)                   trip.emplace_back(id[P], id[P - nx], -theta * op.aN[P - nx]);
    }
    Eigen::SparseMatrix<double> A(m, m);
    A.setFromTriplets(trip.begin(), trip.end());

    // Right-hand side: an implicit step from the initial field
    std::vector<double> Kx(n), b(n, 0.0), x(n, 0.0);
    op.applyK(prob.T0.data(), Kx.data());
    Eigen::VectorXd be(m);
    for (int P = 0; P < n; ++P) {
        if (id[P] < 0) continue;
        b[P] = op.mass[P] / dt * prob.T0[P] - (1 - theta) * Kx[P] + op.bfix[P];
        be[id[P]] = b[P];
        x[P] = prob.T0[P];
    }
    CGStats st = sys.solve(b, x, 1e-13, 100000);
    std::printf("  moldcool CG: %d iterations, relative residual %.2e, unknowns %d\n", st.iterations, st.relative_residual, m);
    CHECK(st.converged);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(A);
    CHECK(ldlt.info() == Eigen::Success);
    Eigen::VectorXd xd = ldlt.solve(be);
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> ecg;
    ecg.setTolerance(1e-13);
    ecg.setMaxIterations(100000);
    ecg.compute(A);
    Eigen::VectorXd xc = ecg.solve(be);
    std::printf("  Eigen CG: %ld iterations, error %.2e\n", long(ecg.iterations()), ecg.error());

    double dmax_d = 0, dmax_c = 0, scale = 0;
    for (int P = 0; P < n; ++P) {
        if (id[P] < 0) continue;
        dmax_d = std::max(dmax_d, std::abs(x[P] - xd[id[P]]));
        dmax_c = std::max(dmax_c, std::abs(x[P] - xc[id[P]]));
        scale = std::max(scale, std::abs(xd[id[P]]));
    }
    std::printf("  max |x_moldcool - x_LDLT| / max|x| = %.2e,  vs Eigen CG = %.2e\n", dmax_d / scale, dmax_c / scale);
    CHECK(dmax_d / scale < 1e-9);
    CHECK(dmax_c / scale < 1e-8);
    return moldcool_test::finish("cg_vs_eigen");
}
