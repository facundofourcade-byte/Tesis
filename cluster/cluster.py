"""Corre gl3d (binario de gl3d.cu) y despues calcula f(q) con twist_q.py.

El binario tiene que estar en el directorio desde donde se corre este script. Deja ahi
density.dat, phase.dat y run_log.dat, y twist_q.py usa density.dat y phase.dat para f(q).

Los parametros se cambian en dos lugares:
  - aca: la red, la semilla y k1 k3 k4 k6 de la region 0 (se le pasan al binario);
  - en main() de gl3d.cu: k2, alfa y todos los de la region 1.
twist_q.py recibe los coeficientes de la linea que imprime el binario al arrancar, asi que
siempre usa exactamente los mismos que el minimizador.

Uso:  python cluster.py   (los parametros se editan abajo)
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import twist_q

BINARIO = "./gl3d"

# Parametros del binario: gl3d Nx Ny Nz Nv dx By seed.dat kd [k1 k3 k4 k6]
Nx, Ny, Nz, Nv = 64, 64, 8, 4
dx, By = 0.5, 0.0
SEED = "seed.dat"           # Nx*Ny*Nz lineas 'Re Im'
kd = 4                      # 0 <= kd < Nz, Nz >= 4

# Coeficientes de la region 0 (argv[9..12] del binario)
K1 = 0.0
K3 = 0.0
K4 = 0.25
K6 = 0.001

# Barrido en q de twist_q.py
QMAX, NQ = 0.5, 101


def correr_binario():
    """Corre el binario mostrando su salida y devuelve los coeficientes que imprime al arrancar,
    como {nombre: (region 0, region 1)}."""
    if not os.path.isfile(BINARIO):
        raise FileNotFoundError(f"No se encontro el binario {BINARIO} en {os.getcwd()}")
    args = [BINARIO, *map(str, (Nx, Ny, Nz, Nv, dx, By, SEED, kd, K1, K3, K4, K6))]
    coef = {}
    with subprocess.Popen(args, stdout=subprocess.PIPE, text=True) as proc:
        for linea in proc.stdout:
            print(linea, end="", flush=True)
            if not coef and linea.startswith("Nx="):
                coef = {k: (float(a), float(b)) for k, a, b in
                        re.findall(r"\b(k1|k2|alfa|k3|k4|k6)=\(([^,]+),([^)]+)\)", linea)}
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args)
    if set(coef) != set(twist_q.COEF):
        raise RuntimeError(f"No se pudieron leer los coeficientes del binario (se leyo {coef})")
    return coef


if __name__ == "__main__":
    coef = correr_binario()
    flags = [x for k, (a, b) in coef.items() for x in (f"--{k}", a, b)]
    twist_q.main(list(map(str, (Nx, Ny, Nz, Nv, dx, By, kd, "density.dat", "phase.dat",
                                *flags, "--qmax", QMAX, "--nq", NQ))))
