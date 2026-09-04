// moldcool: 2D transient heat conduction for injection-mould cooling-time estimation.
//
// Grid, materials and problem description. Everything is templated on the floating-point
// type `Real` so the same code can be run in float and double to expose round-off behaviour.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace moldcool {

// OpenMP is only worth it above this many cells. Below it, the cost of waking a thread team
// several times per CG iteration dominates: on a 16-thread laptop the 1D and small 2D cases ran
// 13x slower with OpenMP than without until this guard was added. Every `#pragma omp` in the
// library carries `if(n >= kOmpMinCells)`.
inline constexpr int kOmpMinCells = 1 << 15;

// Cell classification.
//   Unknown: temperature is a degree of freedom.
//   Fixed:   cell holds a prescribed temperature. The face shared with an Unknown neighbour is
//            a Dirichlet boundary located AT THE FACE (half-cell distance), which keeps the
//            scheme second order. Used for isothermal mould walls and coolant channels.
//   Void:    no material. Faces to Void (and to outside the grid) are adiabatic.
enum class Cell : std::uint8_t { Unknown = 0, Fixed = 1, Void = 2 };

template <class Real>
struct Grid {
    int nx = 0, ny = 1;       // number of cells
    Real dx = 1, dy = 1;      // cell size
    Real x0 = 0, y0 = 0;      // position of the south-west corner of cell (0,0)

    int n() const { return nx * ny; }
    int idx(int i, int j) const { return j * nx + i; }
    Real xc(int i) const { return x0 + (Real(i) + Real(0.5)) * dx; }  // cell-centre x
    Real yc(int j) const { return y0 + (Real(j) + Real(0.5)) * dy; }  // cell-centre y
    Real volume() const { return dx * dy; }                            // per unit depth
};

template <class Real>
struct Material {
    Real k;    // thermal conductivity, W/(m K)
    Real rho;  // density, kg/m^3
    Real cp;   // specific heat, J/(kg K)
    Real alpha() const { return k / (rho * cp); }  // thermal diffusivity, m^2/s
    Real rhocp() const { return rho * cp; }
};

// Representative property sets. Sources are listed in README.md (Materials).
template <class Real>
struct Materials {
    // ABS, amorphous. k 0.17-0.25, rho 1020-1060, cp 1300-1500 in the literature.
    static Material<Real> ABS() { return {Real(0.20), Real(1040), Real(1400)}; }
    // Polypropylene, semi-crystalline (latent heat ignored; see README Limitations).
    static Material<Real> PP() { return {Real(0.22), Real(905), Real(1900)}; }
    // P20 tool steel.
    static Material<Real> P20() { return {Real(29.0), Real(7850), Real(460)}; }
    // Fictitious unit material for verification problems: alpha = 1 exactly.
    static Material<Real> Unit() { return {Real(1), Real(1), Real(1)}; }
};

// Full description of a conduction problem on a uniform Cartesian grid.
template <class Real>
struct Problem {
    Grid<Real> grid;
    std::vector<Cell> mask;    // per cell
    std::vector<Real> k;       // per cell conductivity
    std::vector<Real> rhocp;   // per cell volumetric heat capacity
    std::vector<Real> Tfixed;  // per cell; meaningful for Fixed cells only
    std::vector<Real> T0;      // initial temperature (Unknown cells)

    explicit Problem(const Grid<Real>& g)
        : grid(g), mask(g.n(), Cell::Void), k(g.n(), Real(0)), rhocp(g.n(), Real(0)),
          Tfixed(g.n(), Real(0)), T0(g.n(), Real(0)) {}

    void set(int i, int j, Cell c, const Material<Real>& m, Real T) {
        const int p = grid.idx(i, j);
        mask[p] = c;
        k[p] = m.k;
        rhocp[p] = m.rhocp();
        Tfixed[p] = T;
        T0[p] = T;
    }

    int count(Cell c) const {
        int n = 0;
        for (auto m : mask) n += (m == c);
        return n;
    }
};

// ---------------------------------------------------------------------------------------------
// Geometry builders
// ---------------------------------------------------------------------------------------------

// 1D slab of thickness h between two isothermal walls at Tmould, initially at Tmelt.
// N polymer cells plus one Fixed cell on each side. The polymer occupies x in [0, h].
template <class Real>
Problem<Real> make_slab_1d(Real h, int N, const Material<Real>& mat, Real Tmelt, Real Tmould) {
    if (N < 2) throw std::invalid_argument("make_slab_1d: N must be >= 2");
    Grid<Real> g;
    g.nx = N + 2;
    g.ny = 1;
    g.dx = h / Real(N);
    g.dy = Real(1);       // unit depth in y; north/south faces are outside the grid -> adiabatic
    g.x0 = -g.dx;         // cell 0 is the west wall cell, so the slab starts at x = 0
    Problem<Real> p(g);
    for (int i = 0; i < g.nx; ++i) {
        const bool wall = (i == 0 || i == g.nx - 1);
        p.set(i, 0, wall ? Cell::Fixed : Cell::Unknown, mat, wall ? Tmould : Tmelt);
    }
    return p;
}

// 1D rod of length L with adiabatic ends (no Fixed cells at all), initial profile from f(x).
template <class Real, class F>
Problem<Real> make_rod_neumann_1d(Real L, int N, const Material<Real>& mat, F&& f) {
    Grid<Real> g;
    g.nx = N;
    g.ny = 1;
    g.dx = L / Real(N);
    g.dy = Real(1);
    Problem<Real> p(g);
    for (int i = 0; i < N; ++i) p.set(i, 0, Cell::Unknown, mat, f(g.xc(i)));
    return p;
}

// Unit square [0,1]^2 with homogeneous Dirichlet boundary (T = 0 on all four sides), used for
// the method of manufactured solutions. N x N unknown cells plus a ring of Fixed cells.
template <class Real, class F>
Problem<Real> make_unit_square_dirichlet(int N, const Material<Real>& mat, F&& T_init) {
    Grid<Real> g;
    g.nx = N + 2;
    g.ny = N + 2;
    g.dx = Real(1) / Real(N);
    g.dy = g.dx;
    g.x0 = -g.dx;
    g.y0 = -g.dy;
    Problem<Real> p(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const bool ring = (i == 0 || j == 0 || i == g.nx - 1 || j == g.ny - 1);
            p.set(i, j, ring ? Cell::Fixed : Cell::Unknown, mat, ring ? Real(0) : T_init(g.xc(i), g.yc(j)));
        }
    return p;
}

// Cross-section of a flat plate with one rib on top, inside an isothermal mould (every face of
// the polymer touches a wall at Tmould). Plate: width W, thickness t_plate. Rib: thickness t_rib,
// height h_rib, centred on the plate. Set h_rib = 0 for a plain plate. Cell size `dx` (square).
template <class Real>
Problem<Real> make_plate_rib_2d(Real W, Real t_plate, Real t_rib, Real h_rib, Real dx,
                                const Material<Real>& mat, Real Tmelt, Real Tmould) {
    const int nxp = std::max(2, int(std::lround(W / dx)));
    const int nyp = std::max(2, int(std::lround((t_plate + h_rib) / dx)));
    Grid<Real> g;
    g.nx = nxp + 2;
    g.ny = nyp + 2;
    g.dx = W / Real(nxp);
    g.dy = (t_plate + h_rib) / Real(nyp);
    g.x0 = -g.dx;
    g.y0 = -g.dy;
    Problem<Real> p(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const Real x = g.xc(i), y = g.yc(j);
            const bool in_plate = (x > 0 && x < W && y > 0 && y < t_plate);
            const bool in_rib = (h_rib > 0) && (std::abs(x - W / 2) < t_rib / 2) && (y >= t_plate && y < t_plate + h_rib);
            const bool polymer = in_plate || in_rib;
            p.set(i, j, polymer ? Cell::Unknown : Cell::Fixed, mat, polymer ? Tmelt : Tmould);
        }
    return p;
}

// Conjugate problem: a polymer plate cavity of thickness t_plate and width W inside a steel
// block of size Wb x Hb, with circular coolant channels of diameter d_ch held at Tcool, centred
// at (xc_i, yc_i). Steel and polymer are both Unknown; channels are Fixed; block edges adiabatic.
template <class Real>
Problem<Real> make_conjugate_2d(Real Wb, Real Hb, Real W, Real t_plate, Real dx,
                                const std::vector<std::pair<Real, Real>>& channels, Real d_ch,
                                const Material<Real>& polymer, const Material<Real>& steel,
                                Real Tmelt, Real Tsteel0, Real Tcool) {
    const int nx = std::max(2, int(std::lround(Wb / dx)));
    const int ny = std::max(2, int(std::lround(Hb / dx)));
    Grid<Real> g;
    g.nx = nx;
    g.ny = ny;
    g.dx = Wb / Real(nx);
    g.dy = Hb / Real(ny);
    Problem<Real> p(g);
    const Real cx = Wb / 2, cy = Hb / 2;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const Real x = g.xc(i), y = g.yc(j);
            const bool cavity = std::abs(x - cx) < W / 2 && std::abs(y - cy) < t_plate / 2;
            bool channel = false;
            for (const auto& c : channels)
                if (std::hypot(x - c.first, y - c.second) < d_ch / 2) channel = true;
            if (channel)
                p.set(i, j, Cell::Fixed, steel, Tcool);
            else if (cavity)
                p.set(i, j, Cell::Unknown, polymer, Tmelt);
            else
                p.set(i, j, Cell::Unknown, steel, Tsteel0);
        }
    return p;
}

}  // namespace moldcool
