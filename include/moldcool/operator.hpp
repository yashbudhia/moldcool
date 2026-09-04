// Discrete conduction operator: cell-centred finite volumes on a uniform Cartesian grid.
//
// For an Unknown cell P with neighbours nb across faces of area A at distance d:
//     rhocp_P V dT_P/dt = sum_faces g_f (T_nb - T_P) + V f_P
// with face conductance g_f = k_f A / d, k_f the harmonic mean of the two cell conductivities
// (exact for a piecewise-constant conductivity with continuous flux). Faces to Fixed cells use
// the cell's own k and half distance d/2, which places the Dirichlet condition on the face.
// Faces to Void cells or outside the grid carry zero flux.
//
// Writing K T = aP T_P - sum_nb a_nb T_nb (the "stiffness" part), the semi-discrete system is
//     M dT/dt = -K T + b + V f
// where M is the diagonal mass matrix rhocp V, b collects conductance * Tfixed over Fixed faces.
// K is symmetric (harmonic mean is symmetric in P and nb) and positive definite whenever at least
// one Fixed face exists, positive semi-definite otherwise. That is what lets CG solve the
// implicit systems.
#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "core.hpp"

namespace moldcool {

template <class Real>
struct Operator {
    Grid<Real> grid;
    std::vector<Cell> mask;
    std::vector<Real> aE;    // conductance to the east neighbour, Unknown-Unknown faces only
    std::vector<Real> aN;    // conductance to the north neighbour, Unknown-Unknown faces only
    std::vector<Real> aUU;   // sum of conductances to Unknown neighbours (all four faces)
    std::vector<Real> gfix;  // sum of conductances to Fixed faces
    std::vector<Real> bfix;  // sum of conductance * Tfixed over Fixed faces
    std::vector<Real> mass;  // rhocp * V
    std::vector<Real> Tfixed;
    int n_unknown = 0;

    explicit Operator(const Problem<Real>& p) : grid(p.grid), mask(p.mask), Tfixed(p.Tfixed) {
        const int n = grid.n();
        aE.assign(n, 0); aN.assign(n, 0); aUU.assign(n, 0); gfix.assign(n, 0); bfix.assign(n, 0);
        mass.assign(n, 0);
        const Real Ax = grid.dy, Ay = grid.dx;  // face areas per unit depth
        for (int j = 0; j < grid.ny; ++j)
            for (int i = 0; i < grid.nx; ++i) {
                const int P = grid.idx(i, j);
                if (mask[P] != Cell::Unknown) continue;
                ++n_unknown;
                mass[P] = p.rhocp[P] * grid.volume();
                const Real kP = p.k[P];
                auto face = [&](int ii, int jj, Real A, Real d, Real* store_sym) {
                    if (ii < 0 || jj < 0 || ii >= grid.nx || jj >= grid.ny) return;  // outside: adiabatic
                    const int Q = grid.idx(ii, jj);
                    if (mask[Q] == Cell::Unknown) {
                        const Real kQ = p.k[Q];
                        const Real kf = Real(2) * kP * kQ / (kP + kQ);
                        const Real g = kf * A / d;
                        aUU[P] += g;
                        if (store_sym) *store_sym = g;
                    } else if (mask[Q] == Cell::Fixed) {
                        const Real g = kP * A / (d / Real(2));
                        gfix[P] += g;
                        bfix[P] += g * p.Tfixed[Q];
                    }
                };
                face(i + 1, j, Ax, grid.dx, &aE[P]);
                face(i - 1, j, Ax, grid.dx, nullptr);
                face(i, j + 1, Ay, grid.dy, &aN[P]);
                face(i, j - 1, Ay, grid.dy, nullptr);
            }
    }

    Real aP(int P) const { return aUU[P] + gfix[P]; }

    // y = K x on Unknown cells (y = 0 elsewhere).
    void applyK(const Real* x, Real* y) const {
        const int nx = grid.nx, ny = grid.ny;
        const int n = nx * ny;
        (void)n;
#pragma omp parallel for schedule(static) if (n >= kOmpMinCells)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int P = j * nx + i;
                if (mask[P] != Cell::Unknown) { y[P] = 0; continue; }
                Real s = (aUU[P] + gfix[P]) * x[P];
                if (i + 1 < nx) s -= aE[P] * x[P + 1];
                if (i > 0)      s -= aE[P - 1] * x[P - 1];
                if (j + 1 < ny) s -= aN[P] * x[P + nx];
                if (j > 0)      s -= aN[P - nx] * x[P - nx];
                y[P] = s;
            }
        }
    }

    // Net heat flow into the Unknown region through Fixed faces, W per unit depth:
    //     Q(T) = sum_P (bfix_P - gfix_P T_P)
    Real boundary_heat_flow(const Real* T) const {
        Real q = 0;
        const int n = grid.n();
        for (int P = 0; P < n; ++P)
            if (mask[P] == Cell::Unknown) q += bfix[P] - gfix[P] * T[P];
        return q;
    }

    // Largest stable explicit time step by the Gershgorin bound on M^{-1} K:
    //     dt <= min_P mass_P / aP_P.
    // For interior cells of a uniform grid this is exactly dx^2 dy^2 / (2 alpha (dx^2 + dy^2)),
    // i.e. r_x + r_y <= 1/2. Cells touching a Fixed face get a smaller local bound (their
    // diagonal is larger); that bound is sufficient, not necessary. See notes/derivations.md.
    Real explicit_dt_gershgorin() const {
        Real dt = std::numeric_limits<Real>::infinity();
        const int n = grid.n();
        for (int P = 0; P < n; ++P)
            if (mask[P] == Cell::Unknown && aP(P) > 0) dt = std::min(dt, mass[P] / aP(P));
        return dt;
    }

    // Interior (von Neumann) limit ignoring boundary rows: dt <= 1 / (2 alpha (1/dx^2 + 1/dy^2)),
    // using the largest alpha present.
    Real explicit_dt_interior(const Problem<Real>& p) const {
        Real amax = 0;
        for (int P = 0; P < grid.n(); ++P)
            if (mask[P] == Cell::Unknown) amax = std::max(amax, p.k[P] / p.rhocp[P]);
        const Real inv = Real(1) / (grid.dx * grid.dx) + (grid.ny > 1 ? Real(1) / (grid.dy * grid.dy) : Real(0));
        return Real(1) / (Real(2) * amax * inv);
    }
};

}  // namespace moldcool
