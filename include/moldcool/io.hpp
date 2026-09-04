// Output: legacy VTK (readable by ParaView) and CSV.
#pragma once

#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "core.hpp"

namespace moldcool {

// Legacy ASCII VTK, STRUCTURED_POINTS, cell data. Open in ParaView, apply "Cell Data to Point
// Data" for smooth contours.
template <class Real>
void write_vtk(const std::string& path, const Grid<Real>& g, const std::vector<Cell>& mask,
               const std::vector<Real>& T, const std::vector<Real>* k = nullptr) {
    std::ofstream f(path);
    f << "# vtk DataFile Version 3.0\nmoldcool temperature field\nASCII\nDATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << g.nx + 1 << ' ' << g.ny + 1 << " 1\n";
    f << "ORIGIN " << g.x0 << ' ' << g.y0 << " 0\n";
    f << "SPACING " << g.dx << ' ' << g.dy << ' ' << std::min(g.dx, g.dy) << "\n";
    f << "CELL_DATA " << g.n() << "\n";
    f << "SCALARS T double 1\nLOOKUP_TABLE default\n";
    f << std::setprecision(10);
    for (int P = 0; P < g.n(); ++P) f << double(T[P]) << '\n';
    f << "SCALARS mask int 1\nLOOKUP_TABLE default\n";
    for (int P = 0; P < g.n(); ++P) f << int(mask[P]) << '\n';
    if (k) {
        f << "SCALARS k double 1\nLOOKUP_TABLE default\n";
        for (int P = 0; P < g.n(); ++P) f << double((*k)[P]) << '\n';
    }
}

// Field as CSV: x,y,mask,T (one row per cell), for the plotting scripts.
template <class Real>
void write_field_csv(const std::string& path, const Grid<Real>& g, const std::vector<Cell>& mask,
                     const std::vector<Real>& T) {
    std::ofstream f(path);
    f << std::setprecision(12) << "x,y,mask,T\n";
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const int P = g.idx(i, j);
            f << double(g.xc(i)) << ',' << double(g.yc(j)) << ',' << int(mask[P]) << ',' << double(T[P]) << '\n';
        }
}

}  // namespace moldcool
