// Summation with controlled round-off.
//
// Naive left-to-right summation of n terms has worst-case relative error O(n eps). Kahan
// compensated summation brings it to O(eps) + O(n eps^2), and pairwise summation to O(eps log n).
// The energy-balance check uses these so that the reported residual measures the scheme and the
// linear solver, not the summation.
#pragma once

#include <cstddef>
#include <vector>

namespace moldcool {

template <class Real>
Real naive_sum(const Real* x, std::size_t n) {
    Real s = 0;
    for (std::size_t i = 0; i < n; ++i) s += x[i];
    return s;
}

template <class Real>
Real kahan_sum(const Real* x, std::size_t n) {
    Real s = 0, c = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Real y = x[i] - c;
        const Real t = s + y;
        c = (t - s) - y;
        s = t;
    }
    return s;
}

template <class Real>
Real pairwise_sum(const Real* x, std::size_t n) {
    if (n <= 16) return naive_sum(x, n);
    const std::size_t h = n / 2;
    return pairwise_sum(x, h) + pairwise_sum(x + h, n - h);
}

template <class Real>
Real kahan_sum(const std::vector<Real>& v) { return kahan_sum(v.data(), v.size()); }
template <class Real>
Real pairwise_sum(const std::vector<Real>& v) { return pairwise_sum(v.data(), v.size()); }
template <class Real>
Real naive_sum(const std::vector<Real>& v) { return naive_sum(v.data(), v.size()); }

}  // namespace moldcool
