// Jacobi-preconditioned conjugate gradient, matrix-free.
//
// Solves A x = b for the implicit step, A = M/dt + theta K. A is symmetric positive definite
// (M diagonal positive, K symmetric positive semi-definite), which is exactly the condition CG
// needs. The operator is never assembled: apply() computes A x from the stencil coefficients.
// Non-Unknown entries of every vector are kept at zero and excluded from inner products.
#pragma once

#include <cmath>
#include <vector>

#include "operator.hpp"

namespace moldcool {

struct CGStats {
    int iterations = 0;
    double relative_residual = 0;  // ||b - A x|| / ||b||
    bool converged = false;
};

template <class Real>
class ImplicitSystem {
public:
    ImplicitSystem(const Operator<Real>& op, Real theta, Real dt) : op_(op), theta_(theta), dt_(dt) {
        const int n = op.grid.n();
        diag_.assign(n, Real(1));
        for (int P = 0; P < n; ++P)
            if (op.mask[P] == Cell::Unknown) diag_[P] = op.mass[P] / dt + theta * op.aP(P);
    }

    // y = (M/dt + theta K) x
    void apply(const Real* x, Real* y) const {
        op_.applyK(x, y);
        const int n = op_.grid.n();
#pragma omp parallel for schedule(static) if (n >= kOmpMinCells)
        for (int P = 0; P < n; ++P)
            if (op_.mask[P] == Cell::Unknown) y[P] = op_.mass[P] / dt_ * x[P] + theta_ * y[P];
            else y[P] = 0;
    }

    // Inner product over Unknown cells only.
    Real dot(const Real* a, const Real* b) const {
        const int n = op_.grid.n();
        Real s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static) if (n >= kOmpMinCells)
        for (int P = 0; P < n; ++P)
            if (op_.mask[P] == Cell::Unknown) s += a[P] * b[P];
        return s;
    }

    CGStats solve(const std::vector<Real>& b, std::vector<Real>& x, Real rel_tol, int max_iter) const {
        const int n = op_.grid.n();
        std::vector<Real> r(n, 0), z(n, 0), p(n, 0), q(n, 0);
        apply(x.data(), q.data());
        for (int P = 0; P < n; ++P) r[P] = (op_.mask[P] == Cell::Unknown) ? b[P] - q[P] : Real(0);
        const Real bnorm = std::sqrt(dot(b.data(), b.data()));
        const Real stop = rel_tol * (bnorm > 0 ? bnorm : Real(1));
        CGStats st;
        Real rnorm = std::sqrt(dot(r.data(), r.data()));
        if (rnorm <= stop) { st.converged = true; st.relative_residual = double(rnorm / (bnorm > 0 ? bnorm : 1)); return st; }
        for (int P = 0; P < n; ++P) z[P] = r[P] / diag_[P];
        p = z;
        Real rz = dot(r.data(), z.data());
        for (int it = 1; it <= max_iter; ++it) {
            apply(p.data(), q.data());
            const Real pq = dot(p.data(), q.data());
            const Real alpha = rz / pq;
#pragma omp parallel for schedule(static) if (n >= kOmpMinCells)
            for (int P = 0; P < n; ++P) { x[P] += alpha * p[P]; r[P] -= alpha * q[P]; }
            rnorm = std::sqrt(dot(r.data(), r.data()));
            st.iterations = it;
            if (rnorm <= stop) { st.converged = true; break; }
            for (int P = 0; P < n; ++P) z[P] = r[P] / diag_[P];
            const Real rz_new = dot(r.data(), z.data());
            const Real beta = rz_new / rz;
            rz = rz_new;
#pragma omp parallel for schedule(static) if (n >= kOmpMinCells)
            for (int P = 0; P < n; ++P) p[P] = z[P] + beta * p[P];
        }
        st.relative_residual = double(rnorm / (bnorm > 0 ? bnorm : 1));
        return st;
    }

private:
    const Operator<Real>& op_;
    Real theta_, dt_;
    std::vector<Real> diag_;
};

}  // namespace moldcool
