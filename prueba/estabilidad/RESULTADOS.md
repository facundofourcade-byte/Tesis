# Análisis de estabilidad de `prueba.cu` (Nv = 0)

## Cómo se corrió

No hay GPU en el contenedor, así que `prueba.cu` se compiló **sin modificar** para CPU: Thrust con backend
CPP (headers de `nvidia-cuda-cccl-cu12`) y `shim/cuda_runtime.h`, que define lo mínimo de CUDA que usa el
archivo. El único cambio mecánico es la línea del lanzamiento `kernel_grad<<<...>>>`, que el `sed` pasa a
la macro `LAUNCH` del shim:

```bash
pip download nvidia-cuda-cccl-cu12 --no-deps -d cccl && python3 -m zipfile -e cccl/*.whl cccl
sed 's/kernel_grad<<<Blocks,Threads>>>(\(.*\));/LAUNCH(kernel_grad, Blocks, Threads, \1);/' ../prueba.cu > prueba_cpu.cpp
g++ -O3 -std=c++17 -I shim -I cccl/nvidia/cuda_cccl/include -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP prueba_cpu.cpp -o prueba_cpu
python3 sweep.py k4      # barridos: k4 k6 k4k6 dx By k1 k3 k3b k3_k6_0 kd k6neg_dx
python3 show.py k4
```

- Semilla: `psi = 0.5 + 0.05 (ruido gaussiano complejo)`, fija (rng seed 1), para que estén poblados todos los modos.
- Base: `16x16x8, dx = 0.5, kd = 4, By = 0, k1 = k3 = 0, k4 = 0.1, k6 = 0.001` (región 1 y k2, alfa: los de `main()`).
- `Eref = -sum_r alfa_r^2/2 V_r` es la energía de ambas regiones uniformes sin acoplar (-66.56 en la base).
  Sin términos de Lifshitz/orden superior la energía no puede bajar de ahí; `E/|Eref| << -1` es explosión.
- Estado: `convergio` (|grad|^2 < 1e-10 Ntot), `max_iter` (80000 iteraciones sin converger).
  No apareció ningún caso de NaN ni de line search estancado.

## Predicción analítica (`theory.py`)

Con Nv = 0 (Bz = 0) los operadores del .cu son diagonales en ondas planas de la capa z, con p = q - By z:
`H -> h = (4/dx^2)(sin^2(p dx/2) + sin^2(qy dx/2))` en `[0, 8/dx^2]` y `Dx -> s = sin(p dx)/dx`.
La parte cuadrática de la región 0 es

    g(p, qy) = k2 h - k4 h^2 + k6 h^3 + By s (k1 - k3 h)

Si `g_min < 0` el estado uniforme es inestable y el sistema condensa en ese modo con `|psi|^2 ≈ alfa - g_min`,
energía `≈ -(alfa - g_min)^2/2 V0`. Como en la red `h <= 8/dx^2`, el cuártico siempre acota la energía (no
hay -infinito en una red finita), pero `g_min` escala como potencias de `1/dx`: la "explosión" diverge al
refinar la red. **Con By = 0 la densidad final medida coincide con `alfa - g_min` a 4 cifras** cuando
`|g_min| ≳ 100` (cerca del umbral la interfaz con la región 1 la baja algo: 42.4 contra 44.3), así que la inestabilidad es exactamente la del funcional discretizado, no un defecto del minimizador.

Umbrales (By = 0, k6 > 0): existe `h <= 8/dx^2` con `k2 - k4 h + k6 h^2 < 0` sii
`k4 > k4c(dx) = k2/hmax + k6 hmax`, `hmax = min(8/dx^2, sqrt(k2/k6))`. En el continuo `k4c = 2 sqrt(k2 k6)`.

| dx | k4c (k6=1e-3) | k6c (k4=0.1, k6 < k6c inestable) | k3c (By=1) |
|----|---------------|-----------------------------------|------------|
| 0.5  | 0.807 | -2.11e-2 | 11.5 |
| 0.4  | 0.546 |          |      |
| 0.3  | 0.368 | -2.01e-3 | 6.69 |
| 0.25 | 0.322 | -7.32e-4 | 5.58 |
| 0.2  | 0.315 | -1.20e-4 | 4.46 |
| ->0  | 0.315 | 0^-      | finito |

## Resultados numéricos

### k4 (dx = 0.5, k6 = 1e-3)

| k4 | estado | iter | E | E/\|Eref\| | rho_max reg. 0 | alfa - g_min |
|----|--------|------|---|-----------|----------------|--------------|
| -1, 0, 0.1, 0.25, 0.3, 0.5, 0.7, 0.8 | convergio | 140–330 | -52.17 | -0.78 | 0.51 | estable |
| 0.85 | convergio | 900 | -9.15e4 | -1.4e3 | 42.4 | 44.3 |
| 1.0  | **max_iter** | 80000 | -2.28e6 | -3.4e4 | 197.6 | 197.8 |
| 1.5  | **max_iter** | 80000 | -3.12e7 | -4.7e5 | 709.8 | 709.8 |
| 3.0  | **max_iter** | 80000 | -3.19e8 | -4.8e6 | 2245.8 | 2245.8 |

Umbral medido entre 0.80 y 0.85; predicho 0.807. El modo inestable es el de escala de red (p dx = qy dx = π, damero).

### k6 (dx = 0.5, k4 = 0.1)

| k6 | estado | E | rho_max |
|----|--------|---|---------|
| 1e-2, 1e-3, 1e-4, 0, -1e-4, -1e-3, -1e-2 | convergio | -52.17 | 0.51 |
| -0.1 | **max_iter** | -4.23e8 | 2585.8 (pred. 2585.8) |

Con dx = 0.5 incluso k6 < 0 es estable hasta k6 ≈ -0.021, porque la red corta en `h = 32` y `k2 h` domina.
En el continuo cualquier k6 < 0 es inestable (ver barrido k6 < 0 vs dx abajo).

### k4 y k6 juntos (dx = 0.5)

| k6 \ k4 | 0.3 | 0.8 | 1.0 | 2.0 | 4.0 | 8.0 |
|---------|-----|-----|-----|-----|-----|-----|
| 0    | estable | ρ=23.0 (conv.) | ρ=230 max_iter | ρ=1255 max_iter | ρ=3303 max_iter | ρ=7399 max_iter |
| 0.01 | estable | estable | estable | ρ=927 max_iter | ρ=2975 max_iter | ρ=7071 max_iter |
| 0.1  | estable | estable | estable | estable | ρ=329 max_iter | ρ=4122 max_iter |

Todas las densidades coinciden con `alfa - g_min` (k6 = 0, k4 = 0.8 da 23.0 contra 25.8 predicho por la interfaz).

### Red: dx (L = Nx dx ≈ 8, Lz = Nz dx ≈ 4, kd = Nz/2, k6 = 1e-3)

| dx | red | k4 = 0.1 | k4 = 0.25 | k4 = 0.4 |
|----|-----|----------|-----------|----------|
| 0.5  | 16x16x8  | conv. (220 it) | conv. (190 it) | conv. (190 it) |
| 0.4  | 20x20x10 | conv. (300 it) | conv. (250 it) | conv. (200 it) |
| 0.3  | 16x16x14 | conv. (400 it) | conv. (300 it) | **max_iter, ρ=253.9, E=-1.42e6** (pred. 253.9) |
| 0.25 | 16x16x16 | conv. (590 it) | conv. (390 it) | **max_iter, ρ=1282, E=-2.55e7** (pred. 1282) |
| 0.2  | 16x16x20 | conv. (900 it) | conv. (570 it) | **max_iter, ρ=3040, E=-9.29e7** (pred. 3040) |

- k4 = 0.4 es estable en la red gruesa y explota al refinar: el umbral `k4c(dx)` baja a 0.315.
  k4 = 0.25 (el de `cluster_prueba.py`) queda por debajo del umbral del continuo: estable para todo dx.
- En el lado estable las iteraciones crecen al achicar dx (220 → 900) por el número de condición
  (`~ k6 512/dx^6 + 8 k2/dx^2`), pero siempre converge.

### k6 < 0 vs dx (k4 = 0.1)

| dx | k6 = -1e-4 | k6 = -1e-3 |
|----|------------|------------|
| 0.5  | conv. (210 it) | conv. (190 it) |
| 0.3  | conv. (360 it) | conv. (300 it) |
| 0.25 | conv. (400 it) | **max_iter, ρ=561.3, E=-4.75e6** (pred. 561.4) |
| 0.2  | conv. (450 it) | **max_iter, ρ=7040, E=-5.03e8** (pred. 7040) |

k6 = -1e-3 es estable hasta dx = 0.3 y explota en dx ≤ 0.25; k6 = -1e-4 explotaría recién en dx ≲ 0.19
(umbral `k6c ≈ -k2 dx^4/64`). La energía del modo de red crece como `(k6 512/dx^6)^2`: diverge en el continuo.

### By (k1 = k3 = 0)

| By | estado | E | rho_max reg. 0 |
|----|--------|---|----------------|
| 0     | conv. | -52.17 | 0.51 |
| 0.1   | conv. | -32.80 | 0.37 |
| 0.5   | conv. | -1.90  | 2.7e-4 |
| 1     | conv. | -0.51  | 2.7e-5 |
| 2, 4, 8, 4π | conv. | ~1e-9 | ~1e-14 |
| 8π    | conv. | -52.17 | 0.51 |

By solo **no desestabiliza**: es campo orbital y destruye el orden (ψ → 0 para By ≳ 2), con convergencia normal.
By = 8π (By dx² = 2π) reproduce exactamente By = 0: el enlace `exp(-i By z dx)` es periódico en `By dx^2`
con período 2π, aliasing de red a tener en cuenta (By dx² = π, By = 4π, es el campo máximo representable).

### k1 (By = 1)

| k1 | estado | E | rho_max | g_min |
|----|--------|---|---------|-------|
| 0, 1, 5, 20 | conv. | -0.51 … -0.49 | ~2e-5 | -0.01 … -4.0 |
| 50  | conv. | -1.05e4 | 13.4 | -23.9 |
| -50 | conv. | -1.34e4 | 15.6 | -23.9 |
| 200 | **max_iter** | -3.28e6 | 243.7 | -250 |

El término de Lifshitz lineal da `g_min ≈ -(k1 By)^2/(4 k2)` (helicoidal). Es acotado en el continuo pero
cuadrático en `k1 By`; con By = 1 la supresión orbital lo tapa hasta k1 ~ 20 y explota entre 20 y 50.
La asimetría k1 = ±50 viene de que el gauge `A_x = By z` no es simétrico en z (origen en la capa 0).

### k3 (By = 1, k4 = 0.1, k6 = 1e-3)

| k3 | estado | E | rho_max | g_min |
|----|--------|---|---------|-------|
| -10, 0, 0.5, 1, 2, 5, 10, 11 | conv. | -0.48 … -0.67 | ≤ 1.7e-4 | ≥ 0 |
| 12 | conv. | -34.9 | 0.91 | -24.9 |
| 15 | **max_iter** | -1.13e6 | 154.3 | -171.5 |
| 20 | **max_iter** | -8.13e6 | 405.3 | -419.2 |
| 6 (By = 2) | conv. | ~0 | ~0 | -24.9 |
| 6 (By = 4) | **max_iter** | -1.33e7 | 569.2 | -618.6 |

Umbral medido entre 11 y 12, predicho 11.5. Lo que importa es el producto `k3 By` (umbral ≈ 11.5 a dx = 0.5,
∝ dx al refinar). Con k4 = k6 = 0 y k3 ≤ 5 (By = 1) también es estable: a dx = 0.5 el umbral es 12.4.
El modo inestable no es el damero: `p dx ≈ 1.71, qy dx = π`.

### kd (base)

kd = 0, 1, 4, 7: todos convergen (E = -128, -105.7, -52.2, -11.7). Sin región 0 (kd = 0) da E = Eref exacto.

## Conclusiones

1. **Inestabilidades encontradas** (energía → valores muy negativos, ρ ≫ alfa, y en general sin convergencia):
   - `k4` grande respecto de `sqrt(k2 k6)`: umbral `k4c(dx)`, 0.807 a dx = 0.5 → 0.315 en el continuo.
   - `k6 < 0`: estable en red gruesa por el corte de la red, pero el umbral → 0⁻ como dx⁴; inestable en el continuo.
   - `k3 By` grande (≳ 11.5 a dx = 0.5, ∝ dx): modo de Lifshitz cúbico de escala de red.
   - `k1 By` grande (≳ 20–50 a By = 1): fase helicoidal con `|psi|^2 ~ (k1 By)^2/(4 k2)`.
   - Refinar la red (dx ↓) vuelve inestables puntos que eran estables (k4 = 0.4: estable en dx ≥ 0.4, explota en dx ≤ 0.3).
2. Todas las corridas `max_iter` están del lado inestable: el minimizador no falla, persigue un mínimo cuya
   escala (`|grad|^2` ∝ ρ³) hace inalcanzable la tolerancia absoluta `1e-10 Ntot`. Del lado estable siempre convergió.
3. La densidad final coincide con `alfa - g_min` de la dispersión de la red, así que basta evaluar
   `theory.g_min(dx, By, k1, k3, k4, k6) >= 0` antes de correr para saber si un punto es seguro.
4. By solo no desestabiliza (supresión orbital); ojo con `By dx^2` cerca de π por aliasing.
