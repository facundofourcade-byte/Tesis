"""f(q) = F[psi0 exp(i q.r)] con q = (qx, qy, 0), usando el mismo F que gl3d_k6_esc.cu.

psi0 se reconstruye de density.dat y phase.dat (salida del minimizador). Al multiplicar por
exp(i q.r) las condiciones de borde ganan el twist extra exp(i qx Lx) en x y exp(i qy Ly) en y.

Uso:  python twist_q.py Nx Ny Nz Nv dx By kd density.dat phase.dat [--sup 0] [--qmax Q] [--nq N]
Test: python twist_q.py --test
"""
import argparse
import cmath
import sys

import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (region 0: k < kd, region 1: k >= kd), mismos valores que gl3d_k6_esc.cu
COEF = dict(k1=(1.0, 0.0), k2=(24.8, 1.0), alfa=(0.4, 1.0),
            k3=(0.3, 0.0), k4=(0.1, 0.0), k6=(0.05, 0.0))


def vecinos(f, dx, Bz, By, tx, ty):
    """xp, xm, yp, ym como en Vecinos del .cu, con el twist extra exp(i tx), exp(i ty) en los bordes."""
    Nz, Ny, Nx = f.shape
    x, y, z = cp.arange(Nx) * dx, cp.arange(Ny) * dx, cp.arange(Nz)[:, None, None] * dx
    Ux = cp.exp(-1j * By * z * dx)
    Uy = cp.exp(-1j * Bz * x * dx)
    bx = cp.exp(1j * (Bz * Nx * dx * y + tx))       # borde x: magnetico + twist
    xp = cp.roll(f, -1, 2) * Ux
    xm = cp.roll(f, 1, 2) * Ux.conj()
    xp[..., -1] *= bx
    xm[..., 0] *= bx.conj()
    yp = cp.roll(f, -1, 1) * Uy
    ym = cp.roll(f, 1, 1) * Uy.conj()
    yp[:, -1] *= cmath.exp(1j * ty)
    ym[:, 0] *= cmath.exp(-1j * ty)
    return xp, xm, yp, ym


def energia(psi, dx, Bz, By, kd, coef=COEF, tx=0.0, ty=0.0):
    """F[psi] de compute_energy del .cu. psi con forma (Nz, Ny, Nx)."""
    abs2 = lambda a: a.real ** 2 + a.imag ** 2
    reg0 = cp.arange(psi.shape[0])[:, None, None] < kd
    c = {k: cp.where(reg0, *v) for k, v in coef.items()}

    n = vecinos(psi, dx, Bz, By, tx, ty)
    lz = cp.empty_like(psi)                           # d_z^2: centrada, descentrada en los bordes
    lz[1:-1] = psi[2:] - 2 * psi[1:-1] + psi[:-2]
    lz[0], lz[-1] = lz[1], lz[-2]
    phi = -(n[0] + n[1] + n[2] + n[3] - 4 * psi + lz) / dx ** 2
    m = vecinos(phi, dx, Bz, By, tx, ty)

    kin_psi = abs2(n[0] - psi) + abs2(n[2] - psi)
    kin_phi = abs2(m[0] - phi) + abs2(m[2] - phi)
    kin_psi[:-1] += abs2(psi[1:] - psi[:-1])
    kin_phi[:-1] += abs2(phi[1:] - phi[:-1])

    rho = abs2(psi)
    Dxpsi = -1j * (n[0] - n[1]) / (2 * dx)
    e = (c["k2"] * kin_psi / dx ** 2 + 0.5 * rho ** 2 - c["alfa"] * rho
         + c["k1"] * By * (psi.conj() * Dxpsi).real
         + c["k6"] * kin_phi / dx ** 2 - c["k4"] * abs2(phi)
         - c["k3"] * By * (Dxpsi.conj() * phi).real)
    return float(e.sum()) * dx ** 3


def f_q(psi0, qx, qy, dx, Bz, By, kd, coef=COEF):
    Nz, Ny, Nx = psi0.shape
    r_q = qx * cp.arange(Nx) * dx + qy * cp.arange(Ny)[:, None] * dx
    return energia(psi0 * cp.exp(1j * r_q), dx, Bz, By, kd, coef, qx * Nx * dx, qy * Ny * dx)


def selftest():
    # onda plana sin campo: f = dx^3 Nx Ny sum_k a^2 (k2 l - k4 l^2 + k6 l^3 + a^2/2 - alfa),
    # l = 2(1 - cos(q dx))/dx^2. Sin el twist de borde el enlace i = Nx-1 rompe la igualdad.
    Nx, Ny, Nz, kd, dx, a = 8, 6, 5, 2, 0.7, 0.9
    psi0 = cp.full((Nz, Ny, Nx), a, complex)
    for qx, qy in [(0.37, 0.0), (0.0, 0.41)]:
        l = 2 * (1 - np.cos((qx + qy) * dx)) / dx ** 2
        e = [a ** 2 * (COEF["k2"][r] * l - COEF["k4"][r] * l ** 2 + COEF["k6"][r] * l ** 3
                       + a ** 2 / 2 - COEF["alfa"][r]) for r in (0, 1)]
        esperado = dx ** 3 * Nx * Ny * (kd * e[0] + (Nz - kd) * e[1])
        got = f_q(psi0, qx, qy, dx, 0.0, 0.0, kd)
        assert abs(got - esperado) < 1e-10 * abs(esperado), (qx, qy, got, esperado)
    print("selftest ok")


def main():
    p = argparse.ArgumentParser()
    for name, t in [("Nx", int), ("Ny", int), ("Nz", int), ("Nv", int),
                    ("dx", float), ("By", float), ("kd", int), ("density", str), ("phase", str)]:
        p.add_argument(name, type=t)
    p.add_argument("--sup", type=int, default=1, help="0: sin terminos de orden superior (como sup_order)")
    p.add_argument("--qmax", type=float, default=0.5)
    p.add_argument("--nq", type=int, default=101)
    a = p.parse_args()

    coef = dict(COEF)
    if a.sup == 0:
        for k in ("k1", "k3", "k4", "k6"):
            coef[k] = (0.0, coef[k][1])

    Lx, Ly = a.Nx * a.dx, a.Ny * a.dx
    Bz = 2 * np.pi * a.Nv / (Lx * Ly)
    rho, theta = np.loadtxt(a.density, usecols=3), np.loadtxt(a.phase, usecols=3)
    N = a.Nx * a.Ny * a.Nz
    if rho.size != N or theta.size != N:
        sys.exit(f"Se esperaban {N} sitios, density tiene {rho.size} y phase {theta.size}")
    psi0 = cp.asarray(np.sqrt(rho) * np.exp(1j * theta)).reshape(a.Nz, a.Ny, a.Nx)

    q = np.linspace(-a.qmax, a.qmax, a.nq)
    fx = [f_q(psi0, qi, 0.0, a.dx, Bz, a.By, a.kd, coef) for qi in q]
    fy = [f_q(psi0, 0.0, qi, a.dx, Bz, a.By, a.kd, coef) for qi in q]
    print(f"f(0) = {f_q(psi0, 0.0, 0.0, a.dx, Bz, a.By, a.kd, coef):.14g}")
    np.savetxt("f_q.dat", np.column_stack([q, fx, fy]), header="q  f(q,0,0)  f(0,q,0)")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axi, fi, lab in [(ax[0], fx, "q_x"), (ax[1], fy, "q_y")]:
        axi.plot(q, fi, lw=2, color="#2a6fdb")
        axi.set_xlabel(f"${lab}$")
        axi.set_title(f"$f({lab})$, " + ("$q_y=0$" if lab == "q_x" else "$q_x=0$"))
        axi.grid(alpha=0.3)
    ax[0].set_ylabel(r"$f(q) = F[\psi_0 e^{i q\cdot r}]$")
    fig.tight_layout()
    fig.savefig("f_q.png", dpi=150)


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["--test"] else main()
