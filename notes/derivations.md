# Derivations

Everything the code does, on paper. Notation: cell P, neighbours E W N S, cell size dx, dy,
depth 1, conductivity k, volumetric heat capacity c = rho cp, diffusivity alpha = k / c.

## 1. Finite-volume discretisation

Integrate the heat equation c dT/dt = div(k grad T) + f over cell P and apply the divergence
theorem:

    c_P V dT_P/dt = sum over faces of (k dT/dn) A  +  V f_P,     V = dx dy

Approximate the face flux with the two cell-centre values. For the east face:

    k_f (T_E - T_P) / dx * dy

where k_f is the **harmonic mean** 2 k_P k_E / (k_P + k_E). Why harmonic: for piecewise-constant
conductivity the flux is continuous across the face, so the two half-cells act as resistances in
series, (dx/2)/k_P + (dx/2)/k_E, which is dx / k_f. The arithmetic mean would let heat flow
through an insulator next to a conductor.

Define the face conductance g = k_f A / d. Then

    c_P V dT_P/dt = sum_nb g_nb (T_nb - T_P) + V f_P
                  = -( aP T_P - sum_nb g_nb T_nb ) + b_P + V f_P

with aP = sum of all g and b_P the contributions of prescribed-temperature faces. In matrix
form M dT/dt = -K T + b + V f, with M = diag(c_P V) and K the stiffness matrix.

**Dirichlet at a face.** A Fixed cell next to an Unknown one means "the face between them is at
T_fixed". The distance from the Unknown centre to the face is d/2, so g = k_P A / (d/2). The
boundary sits exactly on the face, which keeps the scheme second order; putting the boundary at
the Fixed cell's centre would shift every wall by half a cell (first-order geometric error).

**Adiabatic.** Zero flux: the face simply contributes nothing.

**Symmetry.** g is symmetric in P and nb, so K is symmetric. Its diagonal is the sum of the
off-diagonal magnitudes plus the Dirichlet conductances, so K is positive semi-definite, and
positive definite as soon as any Dirichlet face exists (Gershgorin plus irreducibility).

## 2. Truncation error

Taylor-expand T around the face centre. The two-point flux (T_E - T_P)/dx equals dT/dx at the
face plus (dx^2 / 24) d3T/dx3 + ..., and the cell average equals the centre value plus
(dx^2 / 24) d2T/dx2 + .... Both errors are O(dx^2) on a uniform grid, so the semi-discrete
operator is second order. Verified: slab_convergence and mms_2d both report order 2.00.

## 3. Theta method and its order

    (M/dt + theta K) T^{n+1} = (M/dt - (1 - theta) K) T^n + theta s^{n+1} + (1 - theta) s^n

Expand T(t^{n+1}) = T^n + dt T' + dt^2/2 T'' + ... and substitute T' = -M^{-1}K T + ...:
the leading local error is dt^2 (1/2 - theta) T''. It vanishes for theta = 1/2 (Crank-Nicolson,
order 2); theta = 0 and 1 are first order. Verified in temporal_order: 1.00, 1.05, 2.00.

## 4. Stability of forward Euler (theta = 0)

T^{n+1} = (I - dt M^{-1}K) T^n. For a uniform interior, M^{-1}K has eigenvalues
alpha * (4/dx^2) sin^2(theta_x/2) + alpha * (4/dy^2) sin^2(theta_y/2), theta in (0, pi]
(von Neumann: insert T_j = G^n exp(i theta j)). The largest is 4 alpha (1/dx^2 + 1/dy^2). The
amplification factor 1 - dt lambda must satisfy |.| <= 1, i.e.

    dt <= 2 / lambda_max  =  1 / ( 2 alpha (1/dx^2 + 1/dy^2) )      (r_x + r_y <= 1/2)

In 1D this is the familiar r = alpha dt / dx^2 <= 1/2. With Dirichlet faces the boundary rows
have a larger diagonal (3 g instead of 2 g in 1D). Gershgorin bounds the eigenvalues of K by the
row sums, which still gives 4 g for those rows, so lambda_max <= 4 alpha / dx^2 and r <= 1/2
remains the limit; the code's `explicit_dt_gershgorin` uses the simpler bound
dt <= min mass_P / aP_P (r <= 1/3 at the wall), sufficient but not necessary. The stability
test shows r = 0.49 non-increasing and r = 0.51 growing by 6e48 in 3000 steps.

**Cost argument.** In 2D, explicit cost per unit physical time scales as cells x steps
= (1/dx^2)(1/dt) with dt ~ dx^2, i.e. dx^-4. Halving dx costs 16x. Implicit methods remove the
dt constraint; GPUs remove the constant factor. Both are in this repo.

## 5. Crank-Nicolson on discontinuous data, and the Rannacher fix

CN amplification for an eigenmode with z = dt lambda is g(z) = (1 - z/2)/(1 + z/2). It is
bounded by 1 for all z > 0 (unconditionally stable), but g -> -1 as z -> infinity. The melt-
against-cold-wall initial condition has energy in the stiffest modes; with dt = dx those modes
have z ~ 4 dt / dx^2 = 4 / dx, so they flip sign every step without decaying. The measured
error stayed at 0.65 regardless of resolution (slab_convergence, column "CN plain").

Backward Euler has g = 1/(1 + z), which crushes stiff modes. Rannacher (1984): replace the
first two CN steps by four BE half-steps. The BE steps damp the non-smooth part, the damage to
the smooth part is O(dt^2) because only a fixed number of steps are affected, and second order
is recovered. Measured: 1.95, 2.04, 2.00, 2.00.

## 6. Energy identity

Sum the theta scheme over all Unknown cells. Internal fluxes cancel in pairs because g is
symmetric, so sum_P (K T)_P = sum_P gfix_P T_P (only wall faces survive). Hence

    H^{n+1} - H^n = dt [ theta Q^{n+1} + (1 - theta) Q^n ],
    H = sum_P c_P V T_P,   Q = sum_P (b_P - gfix_P T_P) + V f_P

exactly in real arithmetic. In double the explicit residual is ~1e-13 per step (round-off), the
CN residual ~5e-11 (CG tolerance 1e-12 on a system with condition number ~1.5), in float
~5e-5. Enthalpy is summed with Kahan compensation so that the residual measures the scheme,
not the summation.

## 7. Conjugate gradient

The implicit matrix A = M/dt + theta K is SPD, so CG minimises the A-norm of the error over
the Krylov space; with Jacobi preconditioning (divide by the diagonal) iteration counts are
~10 at r ~ 0.1 and ~120 at r ~ 80 (mms_2d, N = 160). The operator is never assembled; apply()
runs the stencil. The Eigen cross-check assembles A as a sparse matrix, solves with a direct
LDLT and Eigen's own CG, and agrees with the matrix-free solution to 1e-13.

## 8. The cooling-time formula

Slab thickness h, initial T_melt, walls at T_mould from t = 0. With theta = (T - T_mould) /
(T_melt - T_mould), separation of variables gives

    theta(x, t) = sum_{n odd} (4 / n pi) sin(n pi x / h) exp(-n^2 pi^2 alpha t / h^2)

At the mid-plane sin(n pi / 2) = +-1. Keep n = 1 and solve theta = (T_eject - T_mould) /
(T_melt - T_mould) for t:

    t_c = h^2 / (pi^2 alpha) * ln( (4/pi) (T_melt - T_mould) / (T_eject - T_mould) )

This is the cooling stage in every injection-moulding costing sheet. The n = 3 term is
(1/3) exp(-8 pi^2 Fo) relative, about 1e-6 at typical ejection (Fo ~ 0.18). For ABS at 2 mm,
230 -> 90 C in a 50 C mould, t_c = 5.15 s; the solver reproduces it to 4e-5.

The formula knows nothing about ribs, bosses or the mould. That is what the 2D cases quantify.
