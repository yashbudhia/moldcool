// Theta-method time integration of  M dT/dt = -K T + b + V f(t).
//
//   (M/dt + theta K) T^{n+1} = (M/dt - (1-theta) K) T^n + theta (b + V f^{n+1}) + (1-theta)(b + V f^n)
//
//   theta = 0    forward Euler (FTCS): explicit, first order in time, stable iff dt <= 2/lambda_max
//   theta = 1/2  Crank-Nicolson: second order in time, unconditionally stable, may ring on
//                discontinuous initial data (amplification factor -> -1 for stiff modes)
//   theta = 1    backward Euler: first order, unconditionally stable, strongly damping
//
// The explicit branch never forms or solves a system. The implicit branch calls CG with the
// previous temperature as the initial guess.
#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "cg.hpp"
#include "operator.hpp"
#include "sum.hpp"

namespace moldcool {

template <class Real>
using SourceFn = std::function<Real(Real x, Real y, Real t)>;

template <class Real>
class ThetaStepper {
public:
    ThetaStepper(const Operator<Real>& op, Real theta, Real dt, Real cg_rel_tol = Real(1e-12), int cg_max_iter = 10000)
        : op_(op), theta_(theta), dt_(dt), cg_tol_(cg_rel_tol), cg_max_(cg_max_iter),
          Kx_(op.grid.n(), Real(0)), rhs_(op.grid.n(), Real(0)) {
        if (theta_ > 0) sys_ = std::make_unique<ImplicitSystem<Real>>(op_, theta_, dt_);
    }

    Real theta() const { return theta_; }
    Real dt() const { return dt_; }
    const CGStats& last_cg() const { return cg_; }

    // Advance T from time t to t + dt. T holds Fixed-cell values in Fixed cells (they are left
    // untouched) and is only modified in Unknown cells.
    void step(std::vector<Real>& T, Real t, const SourceFn<Real>* f = nullptr) {
        const int n = op_.grid.n();
        const Real V = op_.grid.volume();
        op_.applyK(T.data(), Kx_.data());
        // Source at t^n and t^{n+1}, evaluated at cell centres (midpoint rule, second order).
        std::vector<Real> fn, fn1;
        if (f) {
            fn.resize(n); fn1.resize(n);
            for (int j = 0; j < op_.grid.ny; ++j)
                for (int i = 0; i < op_.grid.nx; ++i) {
                    const int P = op_.grid.idx(i, j);
                    if (op_.mask[P] != Cell::Unknown) { fn[P] = fn1[P] = 0; continue; }
                    fn[P] = (*f)(op_.grid.xc(i), op_.grid.yc(j), t);
                    fn1[P] = (*f)(op_.grid.xc(i), op_.grid.yc(j), t + dt_);
                }
        }
        if (theta_ == Real(0)) {
#pragma omp parallel for schedule(static) if (n >= kOmpMinCells)
            for (int P = 0; P < n; ++P) {
                if (op_.mask[P] != Cell::Unknown) continue;
                Real rhs = -Kx_[P] + op_.bfix[P];
                if (f) rhs += V * fn[P];
                T[P] += dt_ / op_.mass[P] * rhs;
            }
            return;
        }
        for (int P = 0; P < n; ++P) {
            if (op_.mask[P] != Cell::Unknown) { rhs_[P] = 0; continue; }
            Real r = op_.mass[P] / dt_ * T[P] - (Real(1) - theta_) * Kx_[P] + op_.bfix[P];
            if (f) r += V * (theta_ * fn1[P] + (Real(1) - theta_) * fn[P]);
            rhs_[P] = r;
        }
        cg_ = sys_->solve(rhs_, T, cg_tol_, cg_max_);
    }

    // Discrete energy bookkeeping for the step just taken (see notes/derivations.md, Energy):
    //     H^{n+1} - H^n = dt [ theta Q^{n+1} + (1-theta) Q^n ]
    // with H = sum M T over Unknown cells and Q = boundary heat flow + source. This identity
    // holds to round-off for explicit steps and to the CG tolerance for implicit ones.
    Real net_power(const std::vector<Real>& T, Real t, const SourceFn<Real>* f = nullptr) const {
        Real q = op_.boundary_heat_flow(T.data());
        if (f) {
            const Real V = op_.grid.volume();
            for (int j = 0; j < op_.grid.ny; ++j)
                for (int i = 0; i < op_.grid.nx; ++i) {
                    const int P = op_.grid.idx(i, j);
                    if (op_.mask[P] == Cell::Unknown) q += V * (*f)(op_.grid.xc(i), op_.grid.yc(j), t);
                }
        }
        return q;
    }

private:
    const Operator<Real>& op_;
    Real theta_, dt_, cg_tol_;
    int cg_max_;
    std::vector<Real> Kx_, rhs_;
    std::unique_ptr<ImplicitSystem<Real>> sys_;
    CGStats cg_;
};

// Rannacher start-up: replace the first two Crank-Nicolson steps by four backward-Euler
// half-steps. CN's amplification factor (1 - z/2)/(1 + z/2) tends to -1 as z = dt lambda grows,
// so a discontinuous initial condition (melt against a cold wall) leaves an undamped, sign-
// alternating oscillation in the stiffest modes; backward Euler damps them by 1/(1 + z/2) per
// step and second-order accuracy is recovered (Rannacher, Numer. Math. 43, 1984, pp. 309-327).
// Returns the time reached (t0 + 2 dt).
template <class Real>
Real rannacher_startup(const Operator<Real>& op, std::vector<Real>& T, Real dt, Real t0, Real cg_tol = Real(1e-12)) {
    ThetaStepper<Real> be(op, Real(1), dt / Real(2), cg_tol);
    Real t = t0;
    for (int s = 0; s < 4; ++s) { be.step(T, t); t += dt / Real(2); }
    return t;
}

// Total enthalpy relative to zero, sum over Unknown cells of mass * T, compensated summation.
template <class Real>
Real enthalpy(const Operator<Real>& op, const std::vector<Real>& T) {
    std::vector<Real> terms;
    terms.reserve(op.n_unknown);
    for (int P = 0; P < op.grid.n(); ++P)
        if (op.mask[P] == Cell::Unknown) terms.push_back(op.mass[P] * T[P]);
    return kahan_sum(terms);
}

// Maximum temperature over Unknown cells (the hottest point sets the ejection time).
template <class Real>
Real max_unknown(const Operator<Real>& op, const std::vector<Real>& T) {
    Real m = -std::numeric_limits<Real>::infinity();
    for (int P = 0; P < op.grid.n(); ++P)
        if (op.mask[P] == Cell::Unknown) m = std::max(m, T[P]);
    return m;
}

// Maximum over cells whose conductivity equals `k_select` (used to pick the polymer in the
// conjugate problem).
template <class Real>
Real max_where(const Operator<Real>& op, const Problem<Real>& p, const std::vector<Real>& T, Real k_select) {
    Real m = -std::numeric_limits<Real>::infinity();
    for (int P = 0; P < op.grid.n(); ++P)
        if (op.mask[P] == Cell::Unknown && p.k[P] == k_select) m = std::max(m, T[P]);
    return m;
}

// Run until the hottest selected cell falls below T_eject; the crossing time is interpolated
// linearly between the last two steps. Returns (t_eject, steps). k_select < 0 means all cells.
template <class Real>
std::pair<Real, int> cooling_time(ThetaStepper<Real>& st, const Operator<Real>& op, const Problem<Real>& p,
                                  std::vector<Real>& T, Real T_eject, Real k_select = Real(-1), Real t_max = Real(1e9)) {
    Real t = 0;
    Real prev = (k_select < 0) ? max_unknown(op, T) : max_where(op, p, T, k_select);
    int steps = 0;
    while (t < t_max) {
        st.step(T, t);
        t += st.dt();
        ++steps;
        const Real cur = (k_select < 0) ? max_unknown(op, T) : max_where(op, p, T, k_select);
        if (cur < T_eject) {
            const Real frac = (prev - T_eject) / (prev - cur);
            return {t - st.dt() + frac * st.dt(), steps};
        }
        prev = cur;
    }
    return {t, steps};
}

}  // namespace moldcool
