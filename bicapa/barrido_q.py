"""G(q) = min_psi F_q[psi] de la bicapa con el twist externo q = (qx, qy) en los bordes, corriendo minimizador.cu
para cada q:
    psi_k(x+Lx, y) = psi_k(x, y) exp(i (Bz Lx y + qx Lx)),   psi_k(x, y+Ly) = psi_k(x, y) exp(i qy Ly),   k = 1, 2.

Barre qx con qy = 0 y qy con qx = 0. Primero minimiza en q = 0 desde SEED; para q != 0 la semilla es
psi(q=0) exp(i q.r) en las dos capas, que ya cumple el twist: todos los q arrancan de la misma configuracion de
vortices (y del mismo branch de G(q), que es periodica en qx con periodo 2 pi / Lx). Todo corre en el directorio
actual: density.dat, phase.dat y run_log.dat quedan los de la ultima corrida, seed_q.dat es la semilla de la ultima.
G(q) total es E_final del binario; G_k(q) de cada capa sale de energia() sobre la psi minimizada (el termino de
Josephson -J (psi_1^* psi_2 + c.c.) va mitad a cada capa, asi G = G_1 + G_2).

Stiffness: D_ii = (1/V) d^2 G / dq_i^2, V = Nx Ny dx^2, con la derivada segunda de la parabola a q^2 + b q + c
ajustada a G(qx, 0) y a G(0, qy) alrededor de su minimo (NFIT puntos a cada lado): D_ii = 2 a / V. Con By d != 0
la capa 2 prefiere qx = -By d, asi que el minimo de G puede no estar en q = 0: se imprime el q del minimo, -b / (2a).

Salidas: G_q.dat (q, G(q,0), G(0,q), G_1(q,0), G_2(q,0), G_1(0,q), G_2(0,q)), G_q.png (con las parabolas),
G_capas_q.png (G_1 y G_2, una fila por capa con su propia escala, columnas qx y qy).

Los parametros se cambian aca (red, semilla, K1 K2 d J By, barrido en q) y en main() de minimizador.cu (alfa,
beta, Vp, w). Los alfa, beta (y K) para energia() se leen de la linea que imprime el binario y el pinning del Al,
Vp sum_i exp(-|r-R_i|^2/2w^2) en la red triangular de Nv sitios, de pinning.dat que escribe el binario, asi que
siempre son los mismos que uso el minimizador.

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

# Parametros del binario: minimizador Nx Ny dx Nv By K1 K2 d J seed.dat qx qy
Nx, Ny, Nv = 64, 64, 4
dx, By = 0.5, 0.0
K1, K2 = 1.0, 1.0
d, J = 1.0, 0.1
SEED = "seed.dat"           # 2*Nx*Ny lineas 'Re Im' (primero psi_1, despues psi_2)

# Barrido: q = QMAX * (-NQ..NQ) / NQ en qx (qy = 0) y en qy (qx = 0)
QMAX, NQ = 0.05, 5
NFIT = 3                    # stiffness: parabola con NFIT puntos a cada lado del minimo de G(q)

COEF = ("K", "alfa", "beta")   # pares (capa 1, capa 2) que imprime el binario


def vecinos(f, dx, Bz, By, d, tx, ty):
    """xp, yp (enlaces adelantados) como en Vecinos del .cu, f con forma (2, Ny, Nx), tx = qx Lx, ty = qy Ly."""
    _, Ny, Nx = f.shape
    x, y, k = cp.arange(Nx) * dx, cp.arange(Ny) * dx, cp.arange(2)[:, None, None]
    Ux = cp.exp(1j * k * By * d * dx)               # exp(-i A_x dx), A_x = -By d solo en la capa 2
    Uy = cp.exp(-1j * Bz * x * dx)
    xp = cp.roll(f, -1, 2) * Ux
    xp[..., -1] *= cp.exp(1j * (Bz * Nx * dx * y + tx))   # borde x: magnetico + twist
    yp = cp.roll(f, -1, 1) * Uy
    yp[:, -1] *= np.exp(1j * ty)
    return xp, yp


def energia(psi, dx, Bz, By, d, J, coef, tx=0.0, ty=0.0, pin=0.0):
    """F[psi] de compute_energy del .cu y su aporte por capa. psi con forma (2, Ny, Nx), coef = {K, alfa, beta},
    pin: Vp sum_i exp(-|r-R_i|^2/2w^2) con forma (Ny, Nx), suma al coeficiente de |psi_1|^2.
    Devuelve (F total, [F capa 1, F capa 2])."""
    abs2 = lambda a: a.real ** 2 + a.imag ** 2
    K, a2, a4 = (cp.asarray(v, float)[:, None, None]
                 for v in (coef["K"], (-coef["alfa"][0], coef["alfa"][1]), coef["beta"]))
    xp, yp = vecinos(psi, dx, Bz, By, d, tx, ty)
    rho = abs2(psi)
    e = (K * (abs2(xp - psi) + abs2(yp - psi)) / dx ** 2 + a2 * rho + 0.5 * a4 * rho ** 2
         - J * (psi.conj() * psi[::-1]).real)
    e[0] += pin * rho[0]
    capas = cp.asnumpy(e.sum(axis=(1, 2))) * dx ** 2
    return float(capas.sum()), capas


def stiffness(q, G, V, nfit):
    """D = (1/V) d^2 G/dq^2 de la parabola a q^2 + b q + c ajustada con nfit puntos a cada lado del minimo de G.
    Devuelve (2 a / V, (a, b, c))."""
    i0 = int(np.argmin(G))
    if i0 in (0, len(G) - 1):
        print(f"AVISO: el minimo de G esta en el borde del barrido (q = {q[i0]:.6g}): agrandar QMAX")
    m = slice(max(i0 - nfit, 0), i0 + nfit + 1)
    p = np.polyfit(q[m], G[m], 2)
    return 2 * p[0] / V, p


def minimizar(qx, qy, psi_seed):
    """Corre el binario con twist (qx, qy) desde psi_seed (2, Ny, Nx), en el directorio actual.
    Devuelve (E_final, coeficientes {K, alfa, beta: (capa 1, capa 2)}, psi minimizada)."""
    s = psi_seed.ravel()
    np.savetxt("seed_q.dat", np.column_stack([s.real, s.imag]), fmt="%.16e")
    args = [BINARIO, *map(str, (Nx, Ny, dx, Nv, By, K1, K2, d, J, "seed_q.dat", qx, qy))]
    print(f"q = ({qx:+.6f}, {qy:+.6f}) ...", flush=True)
    out = subprocess.run(args, stdout=subprocess.PIPE, text=True, check=True).stdout
    print("  " + "".join(re.findall(r"Chequeo gradiente.*", out)))
    coef = {k: (float(a), float(b)) for k, a, b in re.findall(r"\b(K|alfa|beta)=\(([^,]+),([^)]+)\)", out)}
    if set(coef) != set(COEF):
        raise RuntimeError(f"No se pudieron leer los coeficientes del binario (se leyo {coef})")
    rho, theta = np.loadtxt("density.dat", usecols=3), np.loadtxt("phase.dat", usecols=3)
    return (float(re.search(r"E_final= (\S+)", out).group(1)), coef,
            (np.sqrt(rho) * np.exp(1j * theta)).reshape(2, Ny, Nx))


def main():
    Lx, Ly = Nx * dx, Ny * dx
    Bz = 2 * np.pi * Nv / (Lx * Ly)
    x, y = np.arange(Nx) * dx, np.arange(Ny)[:, None] * dx
    s = np.loadtxt(SEED)
    cero = minimizar(0.0, 0.0, (s[:, 0] + 1j * s[:, 1]).reshape(2, Ny, Nx))
    coef, psi0 = cero[1], cero[2]
    pin = cp.asarray(np.loadtxt("pinning.dat", usecols=2).reshape(Ny, Nx))

    def G(qx, qy):
        """(G total del binario, [G_1, G_2] de energia()) con psi minimizada en (qx, qy)."""
        E, _, psi = cero if qx == qy == 0 else minimizar(qx, qy, psi0 * np.exp(1j * (qx * x + qy * y)))
        Ep, capas = energia(cp.asarray(psi), dx, Bz, By, d, J, coef, qx * Lx, qy * Ly, pin)
        print(f"  G ={E:.14g}  (G_1 + G_2 de python: {Ep:.14g})")
        if abs(Ep - E) > 1e-8 * abs(E):
            print("AVISO: energia() no coincide con el binario")
        return E, capas

    q = QMAX * np.arange(-NQ, NQ + 1) / NQ
    rx, ry = [G(qi, 0.0) for qi in q], [G(0.0, qi) for qi in q]
    Gx, Gkx = np.array([t for t, _ in rx]), np.array([c for _, c in rx])   # Gkx[iq, capa]
    Gy, Gky = np.array([t for t, _ in ry]), np.array([c for _, c in ry])
    np.savetxt("G_q.dat", np.column_stack([q, Gx, Gy, Gkx, Gky]),
               header="q  G(q,0)  G(0,q)  G_1(q,0)  G_2(q,0)  G_1(0,q)  G_2(0,q)")

    V = Nx * Ny * dx ** 2
    (Dxx, px), (Dyy, py) = stiffness(q, Gx, V, NFIT), stiffness(q, Gy, V, NFIT)
    print(f"Dxx = {Dxx:.10g}  (minimo en qx = {-px[1] / (2 * px[0]):.6g})")
    print(f"Dyy = {Dyy:.10g}  (minimo en qy = {-py[1] / (2 * py[0]):.6g})")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axi, Gi, p, D, lab in [(ax[0], Gx, px, Dxx, "q_x"), (ax[1], Gy, py, Dyy, "q_y")]:
        axi.plot(q, Gi, "o", color="#2a6fdb")
        qq = np.linspace(q[0], q[-1], 200)
        axi.plot(qq, np.polyval(p, qq), lw=1.5, color="#e0542e", label=f"ajuste: $D$ = {D:.6g}")
        axi.legend(fontsize=8)
        axi.set_xlabel(f"${lab}$")
        axi.set_title(f"$G({lab})$, " + ("$q_y=0$" if lab == "q_x" else "$q_x=0$"))
        axi.grid(alpha=0.3)
    ax[0].set_ylabel(r"$G(q) = \min_\psi F_q[\psi]$")
    fig.tight_layout()
    fig.savefig("G_q.png", dpi=150)

    # G_k(q): una fila por capa (cada una con su escala en y), columnas qx y qy
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for k, color in [(0, "#2a6fdb"), (1, "#e0542e")]:
        for col, Gk, lab in [(0, Gkx, "q_x"), (1, Gky, "q_y")]:
            ax[k, col].plot(q, Gk[:, k], "o-", lw=2, color=color)
            ax[k, col].set_title(f"capa {k + 1}: $G_{k + 1}({lab})$", fontsize=9)
            ax[k, col].grid(alpha=0.3)
    ax[-1, 0].set_xlabel("$q_x$  ($q_y=0$)")
    ax[-1, 1].set_xlabel("$q_y$  ($q_x=0$)")
    fig.tight_layout()
    fig.savefig("G_capas_q.png", dpi=150)


def selftest():
    # onda plana sin Bz, psi_k = a_k exp(i q.r) con el twist q L, By d != 0. Por sitio, en la capa k:
    # K_k a_k^2 l_k + a2_k a_k^2 + a4_k a_k^4 / 2 - J a_1 a_2,  a2 = (-alfa_1, alfa_2), a4 = beta,
    # l_k = 2 [2 - cos((qx + k By d) dx) - cos(qy dx)] / dx^2  (la capa 2 ve qx corrido por -A_d,x = By d).
    # Sin el twist de borde el enlace i = Nx-1 rompe la igualdad. El pinning (arbitrario) solo suma
    # dx^2 a_1^2 sum_r pin(r) a la capa 1.
    Nx, Ny, dx, By, d, J = 8, 6, 0.7, 0.3, 1.1, 0.25
    c = dict(K=(1.3, 0.6), alfa=(1.0, 0.4), beta=(1.0, 2.0))
    amp = np.array([0.9, 0.4])
    pin = np.random.default_rng(0).random((Ny, Nx))
    x, y = cp.arange(Nx) * dx, cp.arange(Ny)[:, None] * dx
    for qx, qy in [(0.37, 0.0), (0.0, 0.41), (0.2, -0.3)]:
        psi = cp.asarray(amp)[:, None, None] * cp.exp(1j * (qx * x + qy * y))
        got, capas = energia(psi, dx, 0.0, By, d, J, c, qx * Nx * dx, qy * Ny * dx, cp.asarray(pin))
        esperado = [Nx * Ny * dx ** 2 * (
            c["K"][k] * amp[k] ** 2 * 2 * (2 - np.cos((qx + k * By * d) * dx) - np.cos(qy * dx)) / dx ** 2
            + (-1) ** (k + 1) * c["alfa"][k] * amp[k] ** 2 + 0.5 * c["beta"][k] * amp[k] ** 4
            - J * amp[0] * amp[1]) + (k == 0) * dx ** 2 * amp[0] ** 2 * pin.sum() for k in (0, 1)]
        assert np.allclose(capas, esperado, rtol=1e-10, atol=0), (qx, qy, capas, esperado)
        assert abs(got - sum(esperado)) < 1e-10 * abs(sum(esperado)), (got, esperado)

    # stiffness(): G = 3 + (V/2) D (q - q0)^2 con el minimo corrido de q = 0, y basura (+100) fuera de
    # los 3 puntos a cada lado del minimo (paso 0.025): si la ventana no se usara, el ajuste daria otra cosa
    q, V = np.linspace(-0.5, 0.5, 41), 7.0
    D, p = stiffness(q, 3 + V / 2 * 1.7 * (q - 0.2) ** 2 + 100 * (abs(q - 0.2) > 0.08), V, 3)
    assert abs(D - 1.7) < 1e-3 and abs(-p[1] / (2 * p[0]) - 0.2) < 1e-6, (D, p)
    print("selftest ok")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["--test"] else main()
