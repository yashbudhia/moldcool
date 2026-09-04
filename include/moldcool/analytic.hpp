// Analytical references used for verification.
#pragma once

#include <cmath>

namespace moldcool {

constexpr double kPi = 3.14159265358979323846;

// Slab of thickness h, initially at uniform temperature, both faces held at the wall
// temperature from t = 0. Dimensionless temperature theta = (T - Twall)/(T0 - Twall):
//
//     theta(x,t) = sum_{n odd} (4/(n pi)) sin(n pi x / h) exp(-n^2 pi^2 alpha t / h^2)
//
// (separation of variables; Carslaw & Jaeger, Conduction of Heat in Solids, section 3.3).
// The series converges geometrically for t > 0; `nterms` odd terms are summed.
template <class Real>
Real slab_theta(Real x, Real t, Real h, Real alpha, int nterms = 200) {
    double s = 0;
    const double X = double(x) / double(h);
    const double Fo = double(alpha) * double(t) / (double(h) * double(h));  // Fourier number
    for (int m = 0; m < nterms; ++m) {
        const int n = 2 * m + 1;
        const double e = std::exp(-double(n * n) * kPi * kPi * Fo);
        if (e < 1e-300) break;
        s += 4.0 / (n * kPi) * std::sin(n * kPi * X) * e;
    }
    return Real(s);
}

// Cooling time of a plate from the one-term truncation of the series at the mid-plane
// (sin(n pi / 2) = 1 for n = 1):
//
//     t_c = h^2 / (pi^2 alpha) * ln( (4/pi) (Tmelt - Tmould) / (Teject - Tmould) )
//
// This is the formula quoted in injection-moulding handbooks for the cooling stage of the
// cycle (Menges, Michaeli, Mohren, "How to Make Injection Molds", ch. 8; Osswald, Turng,
// Gramann, "Injection Molding Handbook"). It is exact for the mid-plane temperature up to the
// neglected n = 3 term, whose relative size is (1/3) exp(-8 pi^2 Fo).
template <class Real>
Real cooling_time_formula(Real h, Real alpha, Real Tmelt, Real Tmould, Real Teject) {
    const double a = double(h) * double(h) / (kPi * kPi * double(alpha));
    return Real(a * std::log(4.0 / kPi * (double(Tmelt) - double(Tmould)) / (double(Teject) - double(Tmould))));
}

// Exact mid-plane crossing time from the full series, found by bisection on theta_mid(t).
template <class Real>
Real cooling_time_series(Real h, Real alpha, Real Tmelt, Real Tmould, Real Teject) {
    const double target = (double(Teject) - double(Tmould)) / (double(Tmelt) - double(Tmould));
    double lo = 0, hi = 10.0 * double(cooling_time_formula(h, alpha, Tmelt, Tmould, Teject)) + 1e-12;
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (double(slab_theta<Real>(h / Real(2), Real(mid), h, alpha)) > target) lo = mid; else hi = mid;
    }
    return Real(0.5 * (lo + hi));
}

// Method of manufactured solutions on the unit square with alpha = 1:
//     T(x,y,t) = exp(-t) sin(pi x) sin(pi y)
//     f = dT/dt - alpha laplacian(T) = (-1 + 2 pi^2 alpha) T
// T vanishes on the boundary, so the Dirichlet data is zero. Volumetric form used by the
// solver: rhocp dT/dt = div(k grad T) + f, so with k = rhocp = 1 the source is the same.
template <class Real>
Real mms_T(Real x, Real y, Real t) {
    return Real(std::exp(-double(t)) * std::sin(kPi * double(x)) * std::sin(kPi * double(y)));
}
template <class Real>
Real mms_source(Real x, Real y, Real t, Real alpha) {
    return Real((-1.0 + 2.0 * kPi * kPi * double(alpha))) * mms_T<Real>(x, y, t);
}

// Rod with adiabatic ends and initial condition cos(pi x / L): the cell-centred finite-volume
// discretisation has cos(pi x_i / L) as an exact eigenvector with eigenvalue
//     lambda_h = (4 / dx^2) sin^2(pi dx / (2 L))
// (versus the continuous pi^2 / L^2). The semi-discrete solution is therefore exactly
//     T_i(t) = cos(pi x_i / L) exp(-alpha lambda_h t)
// and comparing a time integrator against it isolates the temporal error from the spatial one.
template <class Real>
Real rod_lambda_h(Real L, Real dx) {
    const double s = std::sin(kPi * double(dx) / (2.0 * double(L)));
    return Real(4.0 / (double(dx) * double(dx)) * s * s);
}

}  // namespace moldcool
