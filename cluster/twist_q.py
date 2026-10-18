"""f(q) = F[psi0 exp(i q.r)] con q = (qx, qy, 0), usando el mismo F que gl3d.cu.

psi0 se reconstruye de density.dat y phase.dat (salida del minimizador). Al multiplicar por
exp(i q.r) las condiciones de borde ganan el twist extra exp(i qx Lx) en x y exp(i qy Ly) en y.
Ademas calcula f(z, q): la misma suma restringida a la capa z (el enlace z (k, k+1) va a la capa k,
como en energy_density_functor), y chequea f(q) = sum_z f(z, q).

Uso:  python twist_q.py Nx Ny Nz Nv dx By kd density.dat phase.dat
                        [--k1 A B] [--k2 A B] [--alfa A B] [--k3 A B] [--k4 A B] [--k6 A B]
                        [--qmax Q] [--nq N]
      (A, B: valor en la region 0 y en la region 1; cluster.py los pasa todos, leidos del binario)
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

# (region 0: k < kd, region 1: k >= kd). Valores por defecto de main() de gl3d.cu, solo para
# correr este script a mano y para el selftest: cluster.py pasa los que uso el binario.
COEF = dict(k1=(0.0, 0.0), k2=(24.8, 1.0), alfa=(0.4, 1.0),
            k3=(0.0, 0.0), k4=(0.25, 0.0), k6=(0.001, 0.0))


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
    """F[psi] de compute_energy del .cu y su aporte por capa z. psi con forma (Nz, Ny, Nx).
    Devuelve (F total, array de Nz con F de cada capa)."""
    abs2 = lambda a: a.real ** 2 + a.imag ** 2
    reg0 = cp.arange(psi.shape[0])[:, None, None] < kd
    c = {k: cp.where(reg0, *v) for k, v in coef.items()}

    n = vecinos(psi, dx, Bz, By, tx, ty)
    lz = cp.empty_like(psi)                           # d_z^2 como D2 del .cu: centrada, descentrada en los bordes
    lz[1:-1] = psi[2:] - 2 * psi[1:-1] + psi[:-2]
    lz[0], lz[-1] = psi[0] - 2 * psi[1] + psi[2], psi[-1] - 2 * psi[-2] + psi[-3]
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
    return float(e.sum()) * dx ** 3, cp.asnumpy(e.sum(axis=(1, 2))) * dx ** 3


def f_q(psi0, qx, qy, dx, Bz, By, kd, coef=COEF):
    Nz, Ny, Nx = psi0.shape
    r_q = qx * cp.arange(Nx) * dx + qy * cp.arange(Ny)[:, None] * dx
    return energia(psi0 * cp.exp(1j * r_q), dx, Bz, By, kd, coef, qx * Nx * dx, qy * Ny * dx)


def stiffness(psi0, dx, Bz, By, kd, coef=COEF, qfit=0.02, nfit=9):
    """Dxx, Dyy, Dxy de f(q) ~ f(0) + (V/2) q.D.q cerca de q = 0: parabola en t in [-qfit, qfit] sobre
    f(t,0), f(0,t), f(t,t) = k+ t^2 y f(t,-t) = k- t^2; Dxx = 2/V a_x, Dyy = 2/V a_y, Dxy = (k+ - k-)/(2V)."""
    V = psi0.size * dx ** 3
    t = np.linspace(-qfit, qfit, nfit)
    a = [np.polyfit(t, [f_q(psi0, ux * ti, uy * ti, dx, Bz, By, kd, coef)[0] for ti in t], 2)[0]
         for ux, uy in [(1, 0), (0, 1), (1, 1), (1, -1)]]
    return 2 * a[0] / V, 2 * a[1] / V, (a[2] - a[3]) / (2 * V)


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
        got, capas = f_q(psi0, qx, qy, dx, 0.0, 0.0, kd)
        assert abs(got - esperado) < 1e-10 * abs(esperado), (qx, qy, got, esperado)
        assert np.allclose(capas, [dx ** 3 * Nx * Ny * e[k >= kd] for k in range(Nz)], rtol=1e-10)

    # psi = 1, una sola region (mismos coeficientes en las dos, sin salto en kd), densidad
    # |D psi|^2 - |psi|^2 + |psi|^4/2: f = V (q^2 - 1/2) -> Dxx = Dyy = 2, Dxy = 0
    std = dict(COEF, k1=(0.0, 0.0), k2=(1.0, 1.0), alfa=(1.0, 1.0), k3=(0.0, 0.0), k4=(0.0, 0.0), k6=(0.0, 0.0))
    D = stiffness(cp.ones((Nz, Ny, Nx), complex), dx, 0.0, 0.0, kd, std)
    assert np.allclose(D, (2, 2, 0), atol=1e-4), D
    print("selftest ok")


def main(argv=None):
    p = argparse.ArgumentParser()
    for name, t in [("Nx", int), ("Ny", int), ("Nz", int), ("Nv", int),
                    ("dx", float), ("By", float), ("kd", int), ("density", str), ("phase", str)]:
        p.add_argument(name, type=t)
    for k in COEF:
        p.add_argument(f"--{k}", type=float, nargs=2, default=COEF[k], help=f"{k} en region 0 y 1")
    p.add_argument("--qmax", type=float, default=0.5)
    p.add_argument("--nq", type=int, default=101)
    a = p.parse_args(argv)

    coef = {k: tuple(getattr(a, k)) for k in COEF}

    Lx, Ly = a.Nx * a.dx, a.Ny * a.dx
    Bz = 2 * np.pi * a.Nv / (Lx * Ly)
    rho, theta = np.loadtxt(a.density, usecols=3), np.loadtxt(a.phase, usecols=3)
    N = a.Nx * a.Ny * a.Nz
    if rho.size != N or theta.size != N:
        sys.exit(f"Se esperaban {N} sitios, density tiene {rho.size} y phase {theta.size}")
    psi0 = cp.asarray(np.sqrt(rho) * np.exp(1j * theta)).reshape(a.Nz, a.Ny, a.Nx)

    q = np.linspace(-a.qmax, a.qmax, a.nq)
    rx = [f_q(psi0, qi, 0.0, a.dx, Bz, a.By, a.kd, coef) for qi in q]
    ry = [f_q(psi0, 0.0, qi, a.dx, Bz, a.By, a.kd, coef) for qi in q]
    fx, fzx = np.array([t for t, _ in rx]), np.array([c for _, c in rx])   # fzx[iq, z]
    fy, fzy = np.array([t for t, _ in ry]), np.array([c for _, c in ry])
    for f, fz in [(fx, fzx), (fy, fzy)]:
        err = np.abs(f - fz.sum(axis=1)).max()
        assert err <= 1e-10 * np.abs(f).max(), f"f(q) != sum_z f(z,q), error {err}"
    print(f"f(0) = {f_q(psi0, 0.0, 0.0, a.dx, Bz, a.By, a.kd, coef)[0]:.14g}  (f(q) = sum_z f(z,q) ok)")
    print("Dxx = %.10g  Dyy = %.10g  Dxy = %.10g" % stiffness(psi0, a.dx, Bz, a.By, a.kd, coef))
    np.savetxt("f_q.dat", np.column_stack([q, fx, fy]), header="q  f(q,0,0)  f(0,q,0)")
    np.savetxt("f_z_qx.dat", np.column_stack([q, fzx]), header="qx  f(z=0,qx)  f(z=1,qx) ...  (qy=0)")
    np.savetxt("f_z_qy.dat", np.column_stack([q, fzy]), header="qy  f(z=0,qy)  f(z=1,qy) ...  (qx=0)")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axi, fi, lab in [(ax[0], fx, "q_x"), (ax[1], fy, "q_y")]:
        axi.plot(q, fi, lw=2, color="#2a6fdb")
        axi.set_xlabel(f"${lab}$")
        axi.set_title(f"$f({lab})$, " + ("$q_y=0$" if lab == "q_x" else "$q_x=0$"))
        axi.grid(alpha=0.3)
    ax[0].set_ylabel(r"$f(q) = F[\psi_0 e^{i q\cdot r}]$")
    fig.tight_layout()
    fig.savefig("f_q.png", dpi=150)

    # f(z,q): una fila por capa (cada una con su escala en y), columnas qx y qy
    fig, ax = plt.subplots(a.Nz, 2, figsize=(10, 2.2 * a.Nz), sharex=True, squeeze=False)
    for z in range(a.Nz):
        for col, fz, lab in [(0, fzx, "q_x"), (1, fzy, "q_y")]:
            ax[z, col].plot(q, fz[:, z], lw=2, color="#2a6fdb")
            ax[z, col].set_title(f"capa k={z}: $f(z, {lab})$", fontsize=9)
            ax[z, col].grid(alpha=0.3)
    ax[-1, 0].set_xlabel("$q_x$  ($q_y=0$)")
    ax[-1, 1].set_xlabel("$q_y$  ($q_x=0$)")
    fig.tight_layout()
    fig.savefig("f_z_q.png", dpi=120)


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["--test"] else main()
