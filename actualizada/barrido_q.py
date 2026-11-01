"""f(q) = min F[psi] con el twist externo q = (qx, qy) en los bordes xy, corriendo minimizador.cu para cada q:
    psi(x+Lx, y) = psi(x, y) exp(i (Bz Lx y + qx Lx)),   psi(x, y+Ly) = psi(x, y) exp(i qy Ly).

Primero minimiza en q = 0 desde SEED. Para q != 0 la semilla es psi(q=0) exp(i q.r), que ya cumple el twist:
todos los q arrancan de la misma configuracion de vortices (y del mismo branch de f(q), que es periodica en
qx con periodo 2 pi / Lx). Cada corrida queda en q_<qx>_<qy>/ (density.dat, phase.dat, run_log.dat, salida.txt
y capas.dat: k, f(z), sum jx, sum jy, sum jz de esa capa).

Stiffness: f(q) ~ f(0) + (V/2) q.D.q. Se minimiza sobre las rectas q = t u, u = (1,0), (0,1), (1,1), (1,-1),
y se ajustan parabolas en |t| <= QFIT. Como psi minimiza F, df/dq = sum J dx^3 (J = -dF/dA, corriente()),
asi que D sale tambien de la pendiente de J: D_ab = (1/V) d(sum J_a dx^3)/dq_b.

Los parametros se cambian aca (red, semilla, k1 k3 k4 k6 de la region 0, barrido en q) y en main() de
minimizador.cu (k2, alfa y la region 1). Los coeficientes para energia() y corriente() se leen de la linea
que imprime el binario, asi que siempre son los mismos que uso el minimizador.

Uso:  python barrido_q.py          (con el binario ./minimizador en el directorio actual)
Test: python barrido_q.py --test
"""
import os
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

# Barrido: t = QMAX * (-NQ..NQ) / NQ sobre cada recta q = t u; parabolas con |t| <= QFIT
QMAX, NQ, QFIT = 0.05, 5, 0.05
DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]

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


def stiffness(t, R, V, qfit):
    """R[u]: filas (f, sum Jx dx^3, sum Jy dx^3) sobre q = t u. Devuelve ((Dxx, Dyy, Dxy) de las parabolas de f,
    (Dxx, Dyy, Dxy) de las pendientes de J). f(t,t) = k+ t^2, f(t,-t) = k- t^2 -> Dxy = (k+ - k-)/(2V)."""
    m = np.abs(t) <= qfit
    a = [np.polyfit(t[m], R[u][m, 0], 2)[0] for u in DIRS]
    s = lambda u, col: np.polyfit(t[m], R[u][m, col], 1)[0] / V
    return (2 * a[0] / V, 2 * a[1] / V, (a[2] - a[3]) / (2 * V)), (s((1, 0), 1), s((0, 1), 2), s((0, 1), 1))


def leer_psi(d):
    rho, theta = np.loadtxt(f"{d}/density.dat", usecols=3), np.loadtxt(f"{d}/phase.dat", usecols=3)
    return (np.sqrt(rho) * np.exp(1j * theta)).reshape(Nz, Ny, Nx)


def minimizar(qx, qy, psi_seed):
    """Corre el binario con twist (qx, qy) desde psi_seed (Nz, Ny, Nx) en q_<qx>_<qy>/.
    Devuelve (E_final, coeficientes {nombre: (region 0, region 1)}, psi minimizada, carpeta)."""
    d = f"q_{qx:+.6f}_{qy:+.6f}"
    os.makedirs(d, exist_ok=True)
    s = psi_seed.ravel()
    np.savetxt(f"{d}/seed.dat", np.column_stack([s.real, s.imag]), fmt="%.16e")
    args = [os.path.abspath(BINARIO), *map(str, (Nx, Ny, Nz, Nv, dx, qx, qy, By, "seed.dat", kd, K1, K3, K4, K6))]
    print(f"q = ({qx:+.6f}, {qy:+.6f}) ...", flush=True)
    out = subprocess.run(args, cwd=d, stdout=subprocess.PIPE, text=True, check=True).stdout
    with open(f"{d}/salida.txt", "w") as f:
        f.write(out)
    coef = {k: (float(a), float(b)) for k, a, b in re.findall(r"\b(k1|k2|alfa|k3|k4|k6)=\(([^,]+),([^)]+)\)", out)}
    if set(coef) != set(COEF):
        raise RuntimeError(f"No se pudieron leer los coeficientes del binario (se leyo {coef})")
    return float(re.search(r"E_final= (\S+)", out).group(1)), coef, leer_psi(d), d


def main():
    Lx, Ly = Nx * dx, Ny * dx
    Bz = 2 * np.pi * Nv / (Lx * Ly)
    V = Nx * Ny * Nz * dx ** 3
    r = (np.arange(Nx) * dx, np.arange(Ny)[:, None] * dx)
    s = np.loadtxt(SEED)
    cero = minimizar(0.0, 0.0, (s[:, 0] + 1j * s[:, 1]).reshape(Nz, Ny, Nx))
    coef, psi0 = cero[1], cero[2]

    t = QMAX * np.arange(-NQ, NQ + 1) / NQ
    res = {}    # (qx, qy) -> (f, sum Jx dx^3, sum Jy dx^3)
    for u in DIRS:
        for q in [(u[0] * ti + 0.0, u[1] * ti + 0.0) for ti in t]:     # + 0.0: -0.0 -> 0.0 en el nombre de carpeta
            if q in res:
                continue
            E, _, psi, d = cero if q == (0, 0) else minimizar(*q, psi0 * np.exp(1j * (q[0] * r[0] + q[1] * r[1])))
            tw = (q[0] * Lx, q[1] * Ly)
            Ep, fz = energia(cp.asarray(psi), dx, Bz, By, kd, coef, *tw)
            J = [cp.asnumpy(j.sum(axis=(1, 2))) * dx ** 3 for j in corriente(cp.asarray(psi), dx, Bz, By, kd, coef, *tw)]
            np.savetxt(f"{d}/capas.dat", np.column_stack([np.arange(Nz), fz, *J]),
                       header="k  f(z)  sum jx dx^3  sum jy dx^3  sum jz dx^3")
            print(f"  f = {E:.14g}  (python: {Ep:.14g})  sum J dx^3 = ({J[0].sum():.10g}, {J[1].sum():.10g})")
            res[q] = (E, J[0].sum(), J[1].sum())

    np.savetxt("f_q.dat", [(*q, *v) for q, v in res.items()], header="qx  qy  f(q)  sum Jx dx^3  sum Jy dx^3")
    R = {u: np.array([res[u[0] * ti, u[1] * ti] for ti in t]) for u in DIRS}
    Df, DJ = stiffness(t, R, V, QFIT)
    print("de f(q): Dxx = %.10g  Dyy = %.10g  Dxy = %.10g" % Df)
    print("de J(q): Dxx = %.10g  Dyy = %.10g  Dxy = %.10g" % DJ)

    # izq: f(t u) - f(0) con su parabola; der: df/dt (diferencias) vs u.J (debe coincidir en el minimo)
    m = np.abs(t) <= QFIT
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for u, c in zip(DIRS, ["#2a6fdb", "#e0542e", "#2a9d5c", "#8e44ad"]):
        f = R[u][:, 0] - res[0, 0][0]
        ax[0].plot(t, f, "o", color=c, label=f"u = {u}")
        ax[0].plot(t, np.polyval(np.polyfit(t[m], f[m], 2), t), color=c, lw=1)
        ax[1].plot(t, np.gradient(R[u][:, 0], t), color=c, lw=1, label=f"df/dt, u = {u}")
        ax[1].plot(t, u[0] * R[u][:, 1] + u[1] * R[u][:, 2], "o", color=c, ms=4)
    ax[0].set_ylabel(r"$f(t\,u) - f(0)$")
    ax[1].set_ylabel(r"$df/dt$ (linea),  $u \cdot \sum J\,dx^3$ (puntos)")
    for a in ax:
        a.set_xlabel("$t$  ($q = t\\,u$)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("f_q.png", dpi=150)


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

    # stiffness() con f = f0 + (V/2) q.D.q y J = V D q exactos
    Dv, V, t = np.array([[2.0, 0.3], [0.3, 1.5]]), 7.0, np.linspace(-0.1, 0.1, 9)
    R = {u: np.array([(1 + V / 2 * q @ Dv @ q, *(V * Dv @ q)) for q in (np.outer(t, u))]) for u in DIRS}
    for D in stiffness(t, R, V, 0.1):
        assert np.allclose(D, (2.0, 1.5, 0.3)), D
    print("selftest ok")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["--test"] else main()
