"""f(q) = min F[psi] con el twist externo q = (qx, qy) en los bordes xy, corriendo minimizador.cu para cada q:
    psi(x+Lx, y) = psi(x, y) exp(i (Bz Lx y + qx Lx)),   psi(x, y+Ly) = psi(x, y) exp(i qy Ly).

Barre qx con qy = 0 y qy con qx = 0. Primero minimiza en q = 0 desde SEED; para q != 0 la semilla es
psi(q=0) exp(i q.r), que ya cumple el twist: todos los q arrancan de la misma configuracion de vortices (y del
mismo branch de f(q), que es periodica en qx con periodo 2 pi / Lx). Todo corre en el directorio actual:
density.dat, phase.dat y run_log.dat quedan los de la ultima corrida, seed_q.dat es la semilla de la ultima.
f(q) total es E_final del binario; f(z, q) por capa sale de energia() sobre la psi minimizada.

Salidas: f_q.dat (q, f(q,0), f(0,q)), f_z_qx.dat, f_z_qy.dat (q, f de cada capa), f_q.png, f_z_q.png.

corriente() no se usa en el barrido: queda para analizar una psi a mano (sum J dx^3 = df/dq en el minimo).

Los parametros se cambian aca (red, semilla, k1 k3 k4 k6 de la region 0, barrido en q) y en main() de
minimizador.cu (k2, alfa y la region 1). Los coeficientes para energia() y corriente() se leen de la linea
que imprime el binario, asi que siempre son los mismos que uso el minimizador.

Uso:  python barrido_q.py          (con el binario ./minimizador en el directorio actual)
Test: python barrido_q.py --test
"""
import re
import subprocess
import sys

import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINARIO = "./minimizador"

# Parametros del binario: minimizador Nx Ny Nz Nv dx qx qy By seed.dat kd [k1 k3 k4 k6]
Nx, Ny, Nz, Nv = 64, 64, 8, 4
dx, By = 0.5, 0.0
SEED = "seed.dat"           # Nx*Ny*Nz lineas 'Re Im'
kd = 4                      # 0 <= kd < Nz
K1, K3, K4, K6 = 0.0, 0.0, 0.25, 0.001   # region 0 (argv[11..14] del binario)

# Barrido: q = QMAX * (-NQ..NQ) / NQ en qx (qy = 0) y en qy (qx = 0)
QMAX, NQ = 0.05, 5

# (region 0: k < kd, region 1: k >= kd). Solo para el selftest: main() usa los que imprime el binario.
COEF = dict(k1=(0.0, 0.0), k2=(24.8, 1.0), alfa=(0.2, 1.0),
            k3=(0.0, 0.0), k4=(0.1, 0.0), k6=(0.001, 0.0))


def vecinos(f, dx, Bz, By, tx, ty):
    """xp, xm, yp, ym como en Vecinos del .cu, con tx = qx Lx y ty = qy Ly."""
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
    yp[:, -1] *= np.exp(1j * ty)
    ym[:, 0] *= np.exp(-1j * ty)
    return xp, xm, yp, ym


def H(psi, n, dx):
    """phi = H psi = (Dx^2 + Dy^2) psi = -lap_xy psi (n = vecinos(psi)), como H del .cu: solo en el plano."""
    return -(n[0] + n[1] + n[2] + n[3] - 4 * psi) / dx ** 2


def energia(psi, dx, Bz, By, kd, coef, tx=0.0, ty=0.0):
    """F[psi] de compute_energy del .cu y su aporte por capa z (Neumann en z: enlaces solo adentro). psi con forma (Nz, Ny, Nx).
    Devuelve (F total, array de Nz con F de cada capa)."""
    abs2 = lambda a: a.real ** 2 + a.imag ** 2
    reg0 = cp.arange(psi.shape[0])[:, None, None] < kd
    c = {k: cp.where(reg0, *v) for k, v in coef.items()}

    n = vecinos(psi, dx, Bz, By, tx, ty)
    phi = H(psi, n, dx)
    m = vecinos(phi, dx, Bz, By, tx, ty)

    kin_psi = abs2(n[0] - psi) + abs2(n[2] - psi)
    kin_phi = abs2(m[0] - phi) + abs2(m[2] - phi)
    kin_psi[:-1] += abs2(psi[1:] - psi[:-1])

    rho = abs2(psi)
    Dxpsi = -1j * (n[0] - n[1]) / (2 * dx)
    e = (c["k2"] * kin_psi / dx ** 2 + 0.5 * rho ** 2 - c["alfa"] * rho
         + c["k1"] * By * (psi.conj() * Dxpsi).real
         + c["k6"] * kin_phi / dx ** 2 - c["k4"] * abs2(phi)
         - c["k3"] * By * (Dxpsi.conj() * phi).real)
    return float(e.sum()) * dx ** 3, cp.asnumpy(e.sum(axis=(1, 2))) * dx ** 3


def corriente(psi, dx, Bz, By, kd, coef, tx=0.0, ty=0.0):
    """(jx, jy, jz) de J = -dF/dA (sin la de magnetizacion), discretizada como F en energia(), a = x, y:
    j_a = 2 k2 Re[psi* Da psi] + k1 By |psi|^2 d_ax - 2 k4 Re[(H psi)* Da psi + psi* Da H psi]
        + 2 k6 Re[(H psi)* Da H psi + (H^2 psi)* Da psi + psi* Da H^2 psi]
        - k3 By Re[psi* H psi d_ax + (Dx psi)* Da psi + psi* Da Dx psi]
    Los terminos k4 y k6 van juntos como 2 k6 Re(phi* Da phi) + 2 Re(Q* Da psi + psi* Da Q), phi = H psi,
    Q = dF/dphi* = k6 H phi - k4 phi. En los d_ax el psi va promediado en x, Mx psi = (psi(x+dx) + psi(x-dx))/2
    con los enlaces (-> psi si dx -> 0): es lo que deriva el Dx centrado, y asi sum_xy J dx^3 = dF/dq exacto.
    jz = 2 k2 Re[psi* Dz psi] con Dz adelantada en el enlace (k, k+1) de la capa k, como en energia();
    Neumann: k = Nz-1 no tiene enlace y jz = 0. tx, ty: el twist con el que se minimizo psi."""
    reg0 = cp.arange(psi.shape[0])[:, None, None] < kd
    k1, k2, k3, k4, k6 = (cp.where(reg0, *coef[k]) for k in ("k1", "k2", "k3", "k4", "k6"))
    V = lambda f: vecinos(f, dx, Bz, By, tx, ty)
    D = lambda v: (-1j * (v[0] - v[1]) / (2 * dx), -1j * (v[2] - v[3]) / (2 * dx))
    n = V(psi)
    phi = H(psi, n, dx)
    m = V(phi)
    Q = k6 * H(phi, m, dx) - k4 * phi
    Dpsi = D(n)
    j = [2 * (k2 * psi.conj() * a + k6 * phi.conj() * b + Q.conj() * a + psi.conj() * c).real
         - k3 * By * (Dpsi[0].conj() * a + psi.conj() * d).real
         for a, b, c, d in zip(Dpsi, D(m), D(V(Q)), D(V(Dpsi[0])))]
    Mx = (n[0] + n[1]) / 2
    j[0] += By * (k1 * psi.conj() * Mx - k3 * Mx.conj() * phi).real
    jz = cp.zeros_like(j[0])
    jz[:-1] = 2 * k2[:-1] * (-1j * psi[:-1].conj() * (psi[1:] - psi[:-1]) / dx).real
    return j[0], j[1], jz


def minimizar(qx, qy, psi_seed):
    """Corre el binario con twist (qx, qy) desde psi_seed (Nz, Ny, Nx), en el directorio actual.
    Devuelve (E_final, coeficientes {nombre: (region 0, region 1)}, psi minimizada)."""
    s = psi_seed.ravel()
    np.savetxt("seed_q.dat", np.column_stack([s.real, s.imag]), fmt="%.16e")
    args = [BINARIO, *map(str, (Nx, Ny, Nz, Nv, dx, qx, qy, By, "seed_q.dat", kd, K1, K3, K4, K6))]
    print(f"q = ({qx:+.6f}, {qy:+.6f}) ...", flush=True)
    out = subprocess.run(args, stdout=subprocess.PIPE, text=True, check=True).stdout
    coef = {k: (float(a), float(b)) for k, a, b in re.findall(r"\b(k1|k2|alfa|k3|k4|k6)=\(([^,]+),([^)]+)\)", out)}
    if set(coef) != set(COEF):
        raise RuntimeError(f"No se pudieron leer los coeficientes del binario (se leyo {coef})")
    rho, theta = np.loadtxt("density.dat", usecols=3), np.loadtxt("phase.dat", usecols=3)
    return (float(re.search(r"E_final= (\S+)", out).group(1)), coef,
            (np.sqrt(rho) * np.exp(1j * theta)).reshape(Nz, Ny, Nx))


def main():
    Lx, Ly = Nx * dx, Ny * dx
    Bz = 2 * np.pi * Nv / (Lx * Ly)
    x, y = np.arange(Nx) * dx, np.arange(Ny)[:, None] * dx
    s = np.loadtxt(SEED)
    cero = minimizar(0.0, 0.0, (s[:, 0] + 1j * s[:, 1]).reshape(Nz, Ny, Nx))
    coef, psi0 = cero[1], cero[2]

    def f(qx, qy):
        """(f total del binario, f por capa de energia()) con psi minimizada en (qx, qy)."""
        E, _, psi = cero if qx == qy == 0 else minimizar(qx, qy, psi0 * np.exp(1j * (qx * x + qy * y)))
        Ep, fz = energia(cp.asarray(psi), dx, Bz, By, kd, coef, qx * Lx, qy * Ly)
        print(f"  f = {E:.14g}  (sum_z f(z) de python: {Ep:.14g})")
        return E, fz

    q = QMAX * np.arange(-NQ, NQ + 1) / NQ
    rx, ry = [f(qi, 0.0) for qi in q], [f(0.0, qi) for qi in q]
    fx, fzx = np.array([t for t, _ in rx]), np.array([c for _, c in rx])   # fzx[iq, z]
    fy, fzy = np.array([t for t, _ in ry]), np.array([c for _, c in ry])
    np.savetxt("f_q.dat", np.column_stack([q, fx, fy]), header="q  f(q,0)  f(0,q)")
    np.savetxt("f_z_qx.dat", np.column_stack([q, fzx]), header="qx  f(z=0,qx)  f(z=1,qx) ...  (qy=0)")
    np.savetxt("f_z_qy.dat", np.column_stack([q, fzy]), header="qy  f(z=0,qy)  f(z=1,qy) ...  (qx=0)")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axi, fi, lab in [(ax[0], fx, "q_x"), (ax[1], fy, "q_y")]:
        axi.plot(q, fi, "o-", lw=2, color="#2a6fdb")
        axi.set_xlabel(f"${lab}$")
        axi.set_title(f"$f({lab})$, " + ("$q_y=0$" if lab == "q_x" else "$q_x=0$"))
        axi.grid(alpha=0.3)
    ax[0].set_ylabel(r"$f(q) = \min F[\psi]$")
    fig.tight_layout()
    fig.savefig("f_q.png", dpi=150)

    # f(z,q): una fila por capa (cada una con su escala en y), columnas qx y qy
    fig, ax = plt.subplots(Nz, 2, figsize=(10, 2.2 * Nz), sharex=True, squeeze=False)
    for z in range(Nz):
        for col, fz, lab in [(0, fzx, "q_x"), (1, fzy, "q_y")]:
            ax[z, col].plot(q, fz[:, z], "o-", lw=2, color="#2a6fdb")
            ax[z, col].set_title(f"capa k={z}: $f(z, {lab})$", fontsize=9)
            ax[z, col].grid(alpha=0.3)
    ax[-1, 0].set_xlabel("$q_x$  ($q_y=0$)")
    ax[-1, 1].set_xlabel("$q_y$  ($q_x=0$)")
    fig.tight_layout()
    fig.savefig("f_z_q.png", dpi=120)


def selftest():
    # onda plana sin campo con el twist q L: f = dx^3 Nx Ny sum_k a^2 (k2 l - k4 l^2 + k6 l^3 + a^2/2 - alfa),
    # l = 2(1 - cos(q dx))/dx^2. Sin el twist de borde el enlace i = Nx-1 rompe la igualdad.
    Nx, Ny, Nz, kd, dx, a = 8, 6, 5, 2, 0.7, 0.9
    x, y = cp.arange(Nx) * dx, cp.arange(Ny)[:, None] * dx
    for qx, qy in [(0.37, 0.0), (0.0, 0.41)]:
        l = 2 * (1 - np.cos((qx + qy) * dx)) / dx ** 2
        e = [a ** 2 * (COEF["k2"][r] * l - COEF["k4"][r] * l ** 2 + COEF["k6"][r] * l ** 3
                       + a ** 2 / 2 - COEF["alfa"][r]) for r in (0, 1)]
        esperado = dx ** 3 * Nx * Ny * (kd * e[0] + (Nz - kd) * e[1])
        psi = cp.full((Nz, Ny, Nx), a, complex) * cp.exp(1j * (qx * x + qy * y))
        got, capas = energia(psi, dx, 0.0, 0.0, kd, COEF, qx * Nx * dx, qy * Ny * dx)
        assert abs(got - esperado) < 1e-10 * abs(esperado), (qx, qy, got, esperado)
        assert np.allclose(capas, [dx ** 3 * Nx * Ny * e[k >= kd] for k in range(Nz)], rtol=1e-10)

    # psi aleatoria con campo, k1 k3 k4 k6 != 0, salto en kd y twist (tx, ty) != 0:
    # sum J dx^3 = d/dh F_{t + h L}[psi exp(i h r)] en h = 0 (= df/dq si psi minimiza F con twist t).
    # En z no hay twist (Neumann): f(hz) = F_t[psi exp(i hz z)]
    rng = np.random.default_rng(0)
    psi = cp.asarray(rng.normal(size=(Nz, Ny, Nx)) + 1j * rng.normal(size=(Nz, Ny, Nx)))
    Bz, By, h, tx, ty = 2 * np.pi / (Nx * Ny * dx ** 2), 0.3, 1e-5, 0.8, -1.3
    c = dict(COEF, k1=(0.3, -0.2), k3=(0.5, 0.1))
    z = cp.arange(Nz)[:, None, None] * dx
    F = lambda p, tx, ty: energia(p, dx, Bz, By, kd, c, tx, ty)[0]
    fs = [lambda q: F(psi * cp.exp(1j * q * x), tx + q * Nx * dx, ty),
          lambda q: F(psi * cp.exp(1j * q * y), tx, ty + q * Ny * dx),
          lambda q: F(psi * cp.exp(1j * q * z), tx, ty)]
    for j, f in zip(corriente(psi, dx, Bz, By, kd, c, tx, ty), fs):
        dfdq = (f(h) - f(-h)) / (2 * h)
        assert abs(float(j.sum()) * dx ** 3 - dfdq) < 1e-6 * abs(dfdq), (float(j.sum()) * dx ** 3, dfdq)
    print("selftest ok")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["--test"] else main()
