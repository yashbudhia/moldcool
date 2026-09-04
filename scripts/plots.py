#!/usr/bin/env python3
"""Make the README figures from results/*.csv. Run from the repo root after ctest and
`moldcool all results`. Every figure is skipped if its CSV is missing."""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def load(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        print("skip", name)
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return rows


def col(rows, k, cast=float):
    return np.array([cast(r[k]) for r in rows])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(R, name))
    plt.close(fig)
    print("wrote", name)


def slab_convergence():
    rows = load("slab_convergence.csv")
    if not rows:
        return
    h = col(rows, "dx")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.loglog(h, col(rows, "err_ftcs"), "o-", label="FTCS, r = 0.25")
    ax.loglog(h, col(rows, "err_cn_plain"), "x--", label="Crank-Nicolson, dt = dx, plain")
    ax.loglog(h, col(rows, "err_cn_rannacher"), "s-", label="Crank-Nicolson, dt = dx, Rannacher start")
    ref = col(rows, "err_cn_rannacher")[0] * (h / h[0]) ** 2
    ax.loglog(h, ref, "k:", label="slope 2")
    ax.set_xlabel("dx")
    ax.set_ylabel("max |T - exact| at Fo = 0.1")
    ax.set_title("1D slab vs series solution")
    ax.legend(fontsize=7)
    save(fig, "slab_convergence.png")


def temporal_order():
    rows = load("temporal_order.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for name, mk in [("forward_euler", "o-"), ("backward_euler", "^-"), ("crank_nicolson", "s-")]:
        sub = [r for r in rows if r["scheme"] == name]
        if not sub:
            continue
        dt = col(sub, "dt")
        e = col(sub, "err")
        ax.loglog(dt, e, mk, label=name.replace("_", " "))
    dts = np.array([1e-5, 1e-1])
    ax.loglog(dts, 2e-1 * dts, "k:", label="slope 1")
    ax.loglog(dts, 3e-1 * dts**2, "k-.", label="slope 2")
    ax.set_xlabel("dt")
    ax.set_ylabel("max |T - semi-discrete exact|")
    ax.set_title("Temporal order (spatial error removed)")
    ax.legend(fontsize=7)
    save(fig, "temporal_order.png")


def mms():
    rows = load("mms2d_convergence.csv")
    if not rows:
        return
    h = col(rows, "dx")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.loglog(h, col(rows, "err_cn"), "s-", label="Crank-Nicolson vs exact")
    ax.loglog(h, col(rows, "err_be"), "^-", label="backward Euler vs exact")
    ax.loglog(h, col(rows, "be_minus_cn"), "x--", label="|BE - CN| (temporal part)")
    ax.loglog(h, col(rows, "err_cn")[0] * (h / h[0]) ** 2, "k:", label="slope 2")
    ax.loglog(h, col(rows, "be_minus_cn")[0] * (h / h[0]), "k-.", label="slope 1")
    ax.set_xlabel("dx (= dt)")
    ax.set_ylabel("max error at t = 0.5")
    ax.set_title("2D manufactured solution")
    ax.legend(fontsize=7)
    save(fig, "mms2d_convergence.png")


def stability():
    rows = load("stability.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for r_val, mk in [("0.49", "-"), ("0.51", "--")]:
        sub = [r for r in rows if r["r"] == r_val]
        if not sub:
            continue
        s = col(sub, "step")
        ratio = col(sub, "norm_ratio")
        ax.semilogy(s, ratio, mk, label=f"r = {r_val}")
    ax.set_xlabel("step")
    ax.set_ylabel("||T|| / ||T0||")
    ax.set_title("FTCS either side of r = 1/2")
    ax.legend()
    save(fig, "stability.png")


def slab_profiles():
    rows = load("slab_profiles.csv")
    if not rows:
        return
    ts = sorted(set(r["t"] for r in rows), key=float)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for t in ts:
        sub = [r for r in rows if r["t"] == t]
        x = col(sub, "x_mm")
        ax.plot(x, col(sub, "T_series"), "-", lw=1, color="0.3")
        ax.plot(x[::10], col(sub, "T_solver")[::10], "o", ms=3, label=f"t = {float(t):.2f} s")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("T [C]")
    ax.set_title("2 mm ABS plate: solver (dots) vs series (lines)")
    ax.legend(fontsize=7)
    save(fig, "slab_profiles.png")


def field(name, title, out, k_mask=None):
    rows = load(name)
    if not rows:
        return
    x = col(rows, "x") * 1e3
    y = col(rows, "y") * 1e3
    T = col(rows, "T")
    m = col(rows, "mask", int)
    xs, ys = np.unique(x), np.unique(y)
    Z = np.full((len(ys), len(xs)), np.nan)
    ix = np.searchsorted(xs, x)
    iy = np.searchsorted(ys, y)
    Z[iy, ix] = np.where(m == 0, T, np.nan)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    im = ax.pcolormesh(xs, ys, Z, shading="nearest", cmap="inferno")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title, fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="T [C]")
    save(fig, out)


def rib_sweep():
    rows = load("rib_sweep.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(col(rows, "rib_over_wall"), col(rows, "ratio_to_plate"), "o-")
    ax.axvspan(0.5, 0.6, color="green", alpha=0.15, label="handbook rib rule (0.5-0.6 of wall)")
    ax.set_xlabel("rib thickness / wall thickness")
    ax.set_ylabel("cooling time / plain plate")
    ax.set_title("Cost of a rib: cycle time vs rib thickness")
    ax.legend(fontsize=7)
    save(fig, "rib_sweep.png")


def histories():
    rows = load("rib_history.csv")
    conj = load("conjugate_history.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for case in ("plate", "rib"):
        sub = [r for r in rows if r["case"] == case]
        ax.plot(col(sub, "t"), col(sub, "Tmax"), label=f"{case}, isothermal mould")
    if conj:
        ax.plot(col(conj, "t"), col(conj, "Tmax_polymer"), "--", label="plate, steel block + channels")
    ax.axhline(90, color="k", ls=":", lw=1)
    ax.text(0.2, 92, "T_eject = 90 C", fontsize=7)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("hottest polymer cell [C]")
    ax.set_ylim(40, 240)
    ax.set_title("Time to ejection")
    ax.legend(fontsize=7)
    save(fig, "histories.png")


def energy():
    rows = load("energy_balance.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for case, mk in [("double_explicit", "-"), ("double_crank_nicolson", "--"), ("float_explicit", ":")]:
        sub = [r for r in rows if r["case"] == case]
        if sub:
            ax.semilogy(col(sub, "step"), col(sub, "step_rel_residual"), mk, label=case.replace("_", " "))
    ax.set_xlabel("step")
    ax.set_ylabel("|dH - dt Q| / |dt Q|")
    ax.set_title("Per-step energy balance residual")
    ax.legend(fontsize=7)
    save(fig, "energy_balance.png")


if __name__ == "__main__":
    slab_convergence()
    temporal_order()
    mms()
    stability()
    slab_profiles()
    field("rib_at_plate_tc.csv", "Plate + rib at the moment the plain plate would eject (5.15 s)", "rib_field.png")
    field("conjugate_at_eject.csv", "Steel block with coolant channels at polymer ejection", "conjugate_field.png")
    rib_sweep()
    histories()
    energy()
    sys.exit(0)
