"""Corre gl3d (binario de gl3d.cu) y despues calcula f(q) con twist_q.py.

El binario tiene que estar en el directorio desde donde se corre este script. Deja ahi
density.dat, phase.dat y run_log.dat, y twist_q.py usa density.dat y phase.dat para f(q).

Uso:  python cluster.py   (los parametros se editan abajo)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import twist_q

BINARIO = "./gl3d"

# Parametros del binario: gl3d Nx Ny Nz Nv dx By seed.dat kd [k4 k6]
Nx, Ny, Nz, Nv = 64, 64, 8, 4
dx, By = 0.5, 0.0
SEED = "seed.dat"           # Nx*Ny*Nz lineas 'Re Im'
kd = 4                      # 0 <= kd < Nz, Nz >= 4

# k4 y k6 de la region 0 (argv[9], argv[10] del binario). En la region 1 son 0, como en main() del .cu
K4 = 0.25
K6 = 0.001

# Barrido en q de twist_q.py
QMAX, NQ = 0.5, 101


def correr_binario():
    if not os.path.isfile(BINARIO):
        raise FileNotFoundError(f"No se encontro el binario {BINARIO} en {os.getcwd()}")
    subprocess.run([BINARIO, *map(str, (Nx, Ny, Nz, Nv, dx, By, SEED, kd, K4, K6))], check=True)


if __name__ == "__main__":
    correr_binario()
    twist_q.main(list(map(str, (Nx, Ny, Nz, Nv, dx, By, kd, "density.dat", "phase.dat",
                                "--k4", K4, 0.0, "--k6", K6, 0.0,
                                "--qmax", QMAX, "--nq", NQ))))
