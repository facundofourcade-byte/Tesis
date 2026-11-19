"""Version Feyrer de bicapa/barrido_q.py: G(q) = min_psi F_q[psi] de la bicapa con un borde abierto, corriendo
minimizador_feyrer.cu, que en cada llamada hace dos minimizaciones:
    x abierto (sufijo _abx): Dx psi = 0 en x = 0, Lx;   psi_k(x, y+Ly) = psi_k(x, y) exp(i qy Ly)
    y abierto (sufijo _aby): Dy psi = 0 en y = 0, Ly;   psi_k(x+Lx, y) = psi_k(x, y) exp(i qx Lx)
Bz = 0 siempre: no hay vortices ni pinning, y el unico potencial vector es A_d = (-By d, 0) en la capa 2.
En la direccion abierta el twist no tiene efecto, asi que con x abierto solo se barre qy (G(0, qy), Dyy) y con y
abierto solo qx (G(qx, 0), Dxx). Cada llamada al binario usa qx = qy = q y da un punto de cada curva.

G(0) es la minimizacion en q = 0 desde SEED (la misma semilla para los dos bordes).
Para q != 0 la semilla es psi(q=0) exp(i q y) con x abierto y psi(q=0) exp(i q x) con y abierto, de cada borde.
G total es E_final_ab{x,y} del binario; G_1, G_2 de cada capa salen de energia(), con la misma definicion que en
barrido_q.py. Stiffness: D_ii = 2 a / V de la parabola ajustada alrededor del minimo (NFIT puntos a cada lado).

Salidas: density_q0_ab{x,y}.dat, phase_q0_ab{x,y}.dat (psi en q = 0),
G_q.dat (q, G(q,0) [y abierto], G(0,q) [x abierto], G_1(q,0), G_2(q,0), G_1(0,q), G_2(0,q)), G_q.png, G_capas_q.png.
main() devuelve (Dxx, Dyy, qx del minimo, qy del minimo).

Los parametros se cambian aca y en main() de minimizador_feyrer.cu (alfa_1, beta), como en barrido_q.py.

Uso:  python barrido_q_feyrer.py          (con el binario ./minimizador_feyrer en el directorio actual)
Test: python barrido_q_feyrer.py --test
"""
import re
import shutil
import subprocess
import sys

import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINARIO = "./minimizador_feyrer"
AB = ("x", "y")             # borde abierto de cada minimizacion: ab = 0 -> x, ab = 1 -> y

# Parametros del binario: minimizador_feyrer Nx Ny dx By K1 K2 d J g seed_abx seed_aby qx qy
Nx, Ny = 256, 256
dx, By = 0.5, 0.0
K1, K2 = 1.0, 400.0
d, J = 0.032, 0.01112
g = 2.0                     # alfa_2 = 1 + 3.8e-7 K2 g^2 By^2 (lo calcula el binario)
SEED = "seed.dat"           # 2*Nx*Ny lineas 'Re Im' (primero psi_1, despues psi_2), para los dos bordes

# Barrido: q = QMAX * (-NQ..NQ) / NQ, qx con y abierto y qy con x abierto
QMAX, NQ = 0.05, 25
NFIT = 4                    # stiffness: parabola con NFIT puntos a cada lado del minimo de G(q)

ESPESOR = (5 / 70) ** 2 / 12     # espesor finito del Al: suma ESPESOR By^2 a -alfa_1 (como en el .cu)
COEF = ("K", "alfa", "beta")   # pares (capa 1, capa 2) que imprime el binario


def vecinos(f, dx, By, d, tx, ty, ab):
    """xp, yp (enlaces adelantados) como en Vecinos del .cu, f con forma (2, Ny, Nx), tx = qx Lx, ty = qy Ly,
    ab = borde abierto (0 = x, 1 = y): ahi el vecino a traves del borde es el mismo sitio (enlace con D psi = 0)."""
    _, Ny, Nx = f.shape
    k = cp.arange(2)[:, None, None]
    Ux = cp.exp(1j * k * By * d * dx)               # exp(-i A_x dx), A_x = -By d solo en la capa 2 (A_y = 0)
    xp = cp.roll(f, -1, 2) * Ux
    if ab == 0:
        xp[..., -1] = f[..., -1]
    else:
        xp[..., -1] *= np.exp(1j * tx)
    yp = cp.roll(f, -1, 1)
    if ab == 1:
        yp[:, -1] = f[:, -1]
    else:
        yp[:, -1] *= np.exp(1j * ty)
    return xp, yp


def energia(psi, dx, By, d, J, coef, ab, tx=0.0, ty=0.0):
    """F[psi] de compute_energy del .cu con el borde ab abierto, y su aporte por capa (ver barrido_q.py)."""
    abs2 = lambda a: a.real ** 2 + a.imag ** 2
    K, a2, a4 = (cp.asarray(v, float)[:, None, None]
                 for v in (coef["K"], (-coef["alfa"][0] + ESPESOR * By ** 2, coef["alfa"][1]), coef["beta"]))
    xp, yp = vecinos(psi, dx, By, d, tx, ty, ab)
    rho = abs2(psi)
    e = K * (abs2(xp - psi) + abs2(yp - psi)) / dx ** 2 + a2 * rho + 0.5 * a4 * rho ** 2
    e[1] -= 2 * J * (psi[0].conj() * psi[1]).real   # -J (psi_1^* psi_2 + c.c.) completo en la capa 2
    capas = cp.asnumpy(e.sum(axis=(1, 2))) * dx ** 2
    total = float(capas.sum())
    capas[1] /= dx ** 2 * float(rho[0].sum())          # V <|psi_1|^2> = dx^2 sum |psi_1|^2
    return total, capas


def stiffness(q, G, V, nfit):
    """D = (1/V) d^2 G/dq^2 de la parabola a q^2 + b q + c ajustada con nfit puntos a cada lado del minimo de G.
    Devuelve (2 a / V, (a, b, c))."""
    i0 = int(np.argmin(G))
    if i0 in (0, len(G) - 1):
        print(f"AVISO: el minimo de G esta en el borde del barrido (q = {q[i0]:.6g}): agrandar QMAX")
    m = slice(max(i0 - nfit, 0), i0 + nfit + 1)
    p = np.polyfit(q[m], G[m], 2)
    return 2 * p[0] / V, p


def minimizar(q, seeds):
    """Corre el binario con qx = qy = q desde seeds = (psi x abierto, psi y abierto), cada una (2, Ny, Nx), en el
    directorio actual. Devuelve ([(E_final, psi) por borde], coeficientes)."""
    for t, s in zip(AB, seeds):
        s = s.ravel()
        np.savetxt(f"seed_q_ab{t}.dat", np.column_stack([s.real, s.imag]), fmt="%.16e")
    args = [BINARIO, *map(str, (Nx, Ny, dx, By, K1, K2, d, J, g, *[f"seed_q_ab{t}.dat" for t in AB], q, q))]
    print(f"q = {q:+.6f} ...", flush=True)
    out = subprocess.run(args, stdout=subprocess.PIPE, text=True, check=True).stdout
    print("  " + "\n  ".join(re.findall(r"Chequeo gradiente.*", out)))
    coef = {k: (float(a), float(b)) for k, a, b in re.findall(r"\b(K|alfa|beta)=\(([^,]+),([^)]+)\)", out)}
    if set(coef) != set(COEF):
        raise RuntimeError(f"No se pudieron leer los coeficientes del binario (se leyo {coef})")
    res = []
    for t in AB:
        rho, theta = np.loadtxt(f"density_ab{t}.dat", usecols=3), np.loadtxt(f"phase_ab{t}.dat", usecols=3)
        res.append((float(re.search(rf"E_final_ab{t}= (\S+)", out).group(1)),
                    (np.sqrt(rho) * np.exp(1j * theta)).reshape(2, Ny, Nx)))
    return res, coef


def main():
    Lx, Ly = Nx * dx, Ny * dx
    x, y = np.arange(Nx) * dx, np.arange(Ny)[:, None] * dx
    s = np.loadtxt(SEED)
    s = (s[:, 0] + 1j * s[:, 1]).reshape(2, Ny, Nx)
    cero, coef = minimizar(0.0, (s, s))
    for t in AB:
        shutil.copy(f"density_ab{t}.dat", f"density_q0_ab{t}.dat")   # los .dat se pisan con cada q
        shutil.copy(f"phase_ab{t}.dat", f"phase_q0_ab{t}.dat")
    psi0 = [c[1] for c in cero]

    def G(q):
        """[(G, [G_1, G_2]) con x abierto en (0, q), idem con y abierto en (q, 0)]."""
        res = cero if q == 0 else minimizar(q, (psi0[0] * np.exp(1j * q * y), psi0[1] * np.exp(1j * q * x)))[0]
        out = []
        for ab, (E, psi) in enumerate(res):
            Ep, capas = energia(cp.asarray(psi), dx, By, d, J, coef, ab, q * Lx, q * Ly)
            print(f"  G_ab{AB[ab]} = {E:.14g}  (de python: {Ep:.14g})")
            if abs(Ep - E) > 1e-8 * abs(E):
                print("AVISO: energia() no coincide con el binario")
            out.append((E, capas))
        return out

    q = QMAX * np.arange(-NQ, NQ + 1) / NQ
    r = [G(qi) for qi in q]
    Gy, Gky = np.array([t[0][0] for t in r]), np.array([t[0][1] for t in r])   # x abierto: G(0, qy)
    Gx, Gkx = np.array([t[1][0] for t in r]), np.array([t[1][1] for t in r])   # y abierto: G(qx, 0)
    np.savetxt("G_q.dat", np.column_stack([q, Gx, Gy, Gkx, Gky]),
               header="q  G(q,0)[y abierto]  G(0,q)[x abierto]  G_1(q,0)  G_2(q,0)  G_1(0,q)  G_2(0,q)")

    V = Nx * Ny * dx ** 2
    (Dxx, px), (Dyy, py) = stiffness(q, Gx, V, NFIT), stiffness(q, Gy, V, NFIT)
    print(f"Dxx (y abierto) = {Dxx:.10g}  (minimo en qx = {-px[1] / (2 * px[0]):.6g})")
    print(f"Dyy (x abierto) = {Dyy:.10g}  (minimo en qy = {-py[1] / (2 * py[0]):.6g})")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for axi, Gi, p, D, lab, t in [(ax[0], Gx, px, Dxx, "q_x", "y"), (ax[1], Gy, py, Dyy, "q_y", "x")]:
        axi.plot(q, Gi, "o", color="#2a6fdb")
        qq = np.linspace(q[0], q[-1], 200)
        axi.plot(qq, np.polyval(p, qq), lw=1.5, color="#e0542e", label=f"ajuste: $D$ = {D:.6g}")
        axi.legend(fontsize=8)
        axi.set_xlabel(f"${lab}$")
        axi.set_title(f"$G({lab})$, {t} abierto")
        axi.grid(alpha=0.3)
    ax[0].set_ylabel(r"$G(q) = \min_\psi F_q[\psi]$")
    fig.tight_layout()
    fig.savefig("G_q.png", dpi=150)

    fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for k, color in [(0, "#2a6fdb"), (1, "#e0542e")]:
        for col, Gk, lab in [(0, Gkx, "q_x"), (1, Gky, "q_y")]:
            ax[k, col].plot(q, Gk[:, k], "o-", lw=2, color=color)
            ax[k, col].set_title(f"capa {k + 1}: $G_{k + 1}({lab})$", fontsize=9)
            ax[k, col].grid(alpha=0.3)
    ax[-1, 0].set_xlabel("$q_x$  (y abierto)")
    ax[-1, 1].set_xlabel("$q_y$  (x abierto)")
    fig.tight_layout()
    fig.savefig("G_capas_q.png", dpi=150)
    plt.close("all")
    return Dxx, Dyy, -px[1] / (2 * px[0]), -py[1] / (2 * py[0])


def selftest():
    # onda plana, psi_k = a_k exp(i q.r), By d != 0, borde ab abierto: igual que en barrido_q.py, pero en la
    # direccion abierta hay un enlace menos por fila (N - 1 en vez de N) y el twist de ese borde no entra.
    Nx, Ny, dx, By, d, J = 8, 6, 0.7, 0.3, 1.1, 0.25
    c = dict(K=(1.3, 0.6), alfa=(1.0, 0.4), beta=(1.0, 2.0))
    amp = np.array([0.9, 0.4])
    x, y = cp.arange(Nx) * dx, cp.arange(Ny)[:, None] * dx
    for ab in (0, 1):
        for qx, qy in [(0.37, 0.0), (0.0, 0.41), (0.2, -0.3)]:
            psi = cp.asarray(amp)[:, None, None] * cp.exp(1j * (qx * x + qy * y))
            got, capas = energia(psi, dx, By, d, J, c, ab, qx * Nx * dx, qy * Ny * dx)
            V = Nx * Ny * dx ** 2
            lx, ly = Ny * (Nx - (ab == 0)) * dx ** 2, Nx * (Ny - (ab == 1)) * dx ** 2   # area de enlaces x, y
            esperado = [
                c["K"][k] * amp[k] ** 2 * 2 * (lx * (1 - np.cos((qx + k * By * d) * dx)) + ly * (1 - np.cos(qy * dx))) / dx ** 2
                + V * (((-1) ** (k + 1) * c["alfa"][k] + (k == 0) * ESPESOR * By ** 2) * amp[k] ** 2
                       + 0.5 * c["beta"][k] * amp[k] ** 4 - 2 * k * J * amp[0] * amp[1]) for k in (0, 1)]
            assert abs(got - sum(esperado)) < 1e-10 * abs(sum(esperado)), (ab, got, esperado)
            esperado[1] /= V * amp[0] ** 2       # G_2 normalizada por V <|psi_1|^2> = V a_1^2
            assert np.allclose(capas, esperado, rtol=1e-10, atol=0), (ab, qx, qy, capas, esperado)

    q, V = np.linspace(-0.5, 0.5, 41), 7.0
    D, p = stiffness(q, 3 + V / 2 * 1.7 * (q - 0.2) ** 2 + 100 * (abs(q - 0.2) > 0.08), V, 3)
    assert abs(D - 1.7) < 1e-3 and abs(-p[1] / (2 * p[0]) - 0.2) < 1e-6, (D, p)
    print("selftest ok")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["--test"] else main()
