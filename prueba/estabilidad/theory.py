"""Estabilidad lineal de la region 0 de prueba.cu en la red (Nv = 0, Bz = 0).

Para psi ~ exp(i(q x + qy y)) en una capa z, con p = q - By z (el enlace Ux corre el momento),
los operadores del .cu son diagonales:
  H  -> h = (4/dx^2)(sin^2(p dx/2) + sin^2(qy dx/2))   en [0, 8/dx^2]
  Dx -> s = sin(p dx)/dx
y la parte cuadratica de la densidad de energia (sin el -alfa) es
  g(p, qy) = k2 h - k4 h^2 + k6 h^3 + By s (k1 - k3 h).
Si min g < 0 el estado uniforme es inestable; condensando en ese modo |psi|^2 ~ alfa - g_min y la
energia de la region 0 ~ -(alfa - g_min)^2/2 * V0: "explota" cuando |g_min| >> alfa.
"""
import numpy as np

K2, ALFA = 24.8, 0.2


def g_min(dx, By=0.0, k1=0.0, k3=0.0, k4=0.1, k6=0.001, n=801):
    t = np.linspace(-np.pi, np.pi, n)                 # p dx (el corrimiento By z solo recorre esto)
    u = np.linspace(0, np.pi, n // 2 + 1)[:, None]    # qy dx
    h = 4 / dx**2 * (np.sin(t / 2)**2 + np.sin(u / 2)**2)
    s = np.sin(t) / dx
    g = K2 * h - k4 * h**2 + k6 * h**3 + By * s * (k1 - k3 * h)
    i = np.unravel_index(g.argmin(), g.shape)
    return float(g[i]), float(t[i[1]]), float(u[i[0], 0])


def k4_critico(dx, k6=0.001):
    """k4 minimo para que exista h <= 8/dx^2 con k2 - k4 h + k6 h^2 < 0 (By = 0, k6 > 0)."""
    h = min(8 / dx**2, np.sqrt(K2 / k6))
    return K2 / h + k6 * h


if __name__ == "__main__":
    # self-check: umbral de dx = 0.5 visto en el barrido (k4 = 0.80 estable, 0.85 inestable)
    assert g_min(0.5, k4=0.80)[0] >= 0 > g_min(0.5, k4=0.85)[0]
    assert abs(k4_critico(0.5) - (K2 / 32 + 0.032)) < 1e-12
    for dx in (0.5, 0.4, 0.3, 0.25, 0.2, 0.1):
        print(f"dx={dx}: k4c={k4_critico(dx):.4f}  g_min(k6=-1e-3)={g_min(dx, k6=-1e-3)[0]:.4g}")
