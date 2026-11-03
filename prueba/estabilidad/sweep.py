"""Barrido de estabilidad de prueba.cu (compilado para CPU como prueba_cpu). Nv = 0 siempre.

Uso: python sweep.py <nombre> ; los casos de cada barrido estan en CASOS. Escribe <nombre>.json.
"""
import json, os, re, subprocess, sys, tempfile
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "prueba_cpu")
ALFA, MAXIT = (0.2, 1.0), 80000


def correr(p):
    """p: dict con Nx Ny Nz dx By kd k1 k3 k4 k6 (+ timeout). Devuelve p + resultados."""
    Nx, Ny, Nz, kd = p["Nx"], p["Ny"], p["Nz"], p["kd"]
    N = Nx * Ny * Nz
    rng = np.random.default_rng(1)
    z = 0.5 + 0.05 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    with tempfile.TemporaryDirectory(dir=HERE) as d:
        np.savetxt(os.path.join(d, "seed.dat"), np.c_[z.real, z.imag])
        args = [BIN, *map(str, (Nx, Ny, Nz, 0, p["dx"], p["By"], "seed.dat", kd,
                                p["k1"], p["k3"], p["k4"], p["k6"]))]
        try:
            r = subprocess.run(args, cwd=d, capture_output=True, text=True, timeout=p.get("timeout", 900))
            out, err, estado = r.stdout, r.stderr.strip(), None
        except subprocess.TimeoutExpired as e:
            out, err, estado = (e.stdout or b"").decode(), "", "timeout"
        its = [(int(a), float(b), float(c)) for a, b, c in
               re.findall(r"Iter (\d+) \|grad\|\^2 = (\S+) Energy: (\S+)", out)]
        dens = None
        if os.path.exists(os.path.join(d, "density.dat")):
            rho = np.loadtxt(os.path.join(d, "density.dat"))[:, 3].reshape(Nz, Ny, Nx)
            dens = [float(rho[:kd].max()) if kd else None, float(rho[kd:].max()), float(rho.min())]
    if estado is None:
        if "no finito" in err: estado = "NaN/inf"
        elif "sin paso" in err: estado = "estancado"
        elif its and its[-1][0] >= MAXIT - 10: estado = "max_iter"
        else: estado = "convergio"
    dx = p["dx"]
    Eref = -sum(a * a / 2 for a in ALFA[:1]) * kd * Nx * Ny * dx**3 - ALFA[1]**2 / 2 * (Nz - kd) * Nx * Ny * dx**3
    E = its[-1][2] if its else float("nan")
    return {**p, "estado": estado, "iter": its[-1][0] if its else 0, "grad2": its[-1][1] if its else None,
            "E": E, "Eref": Eref, "E/|Eref|": E / abs(Eref), "rho_max_r0,r1,min": dens, "stderr": err[:120]}


BASE = dict(Nx=16, Ny=16, Nz=8, kd=4, dx=0.5, By=0.0, k1=0.0, k3=0.0, k4=0.1, k6=0.001)
def var(**kw): return [{**BASE, **{k: v[i] if isinstance(v, list) else v for k, v in kw.items()}}
                       for i in range(max(len(v) for v in kw.values() if isinstance(v, list)))]

CASOS = {
    "k4": var(k4=[-1.0, 0.0, 0.1, 0.25, 0.3, 0.5, 0.7, 0.8, 0.85, 1.0, 1.5, 3.0]),
    "k6": var(k6=[0.01, 0.001, 1e-4, 0.0, -1e-4, -1e-3, -1e-2, -0.1]),
    "k4k6": [{**BASE, "k4": k4, "k6": k6} for k6 in (0.0, 0.01, 0.1) for k4 in (0.3, 0.8, 1.0, 2.0, 4.0, 8.0)],
    "dx": [{**BASE, "dx": dx, "Nx": n, "Ny": n, "Nz": nz, "kd": nz // 2, "k4": k4, "timeout": 1800}
           for k4 in (0.1, 0.25, 0.4)
           for dx, n, nz in ((0.5, 16, 8), (0.4, 20, 10), (0.3, 16, 14), (0.25, 16, 16), (0.2, 16, 20))],
    "By": var(By=[0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 4 * np.pi, 8 * np.pi]),
    "k1": var(By=1.0, k1=[0.0, 1.0, 5.0, 20.0, 50.0, 200.0, -50.0]),
    "k3": var(By=1.0, k3=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, -10.0]),
    "k3_k6_0": var(By=1.0, k6=0.0, k4=0.0, k3=[0.1, 0.5, 1.0, 2.0, 5.0]),
    "kd": var(kd=[0, 1, 4, 7]),
    "k3b": var(By=1.0, k3=[11.0, 12.0, 15.0, 20.0]) + var(By=[2.0, 4.0], k3=6.0),
    "k6neg_dx": [{**BASE, "dx": dx, "Nz": nz, "kd": nz // 2, "k6": k6, "timeout": 1800}
                 for k6 in (-1e-4, -1e-3) for dx, nz in ((0.5, 8), (0.3, 14), (0.25, 16), (0.2, 20))],
}

if __name__ == "__main__":
    nombre = sys.argv[1]
    with Pool(4) as pool:
        res = pool.map(correr, CASOS[nombre], chunksize=1)
    json.dump(res, open(os.path.join(HERE, "resultados", nombre + ".json"), "w"), indent=1)
    for r in res:
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k not in ("Nx", "Ny")})
