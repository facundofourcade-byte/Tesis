
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>

using cd = thrust::complex<double>;
#define HD __host__ __device__

// ------------------------------------------------------------------
// Geometria y campos:
//   x,y : periodicas, con twist magnetico (torus + traslacion magnetica)
//   z   : abierta (caras libres en z=0 y z=(Nz-1)*dx), condiciones de
//         borde NATURALES del funcional (ver mas abajo)
//
// Potencial vector:  A = (B_y * z, B_z * x, 0)
//   B = rot(A) = (0, B_y, B_z)
//
// Convencion de operadores (como en el enunciado):
//   D = -i grad - A      ->  D_x = -i d/dx - B_y z,  D_y = -i d/dy - B_z x,
//                            D_z = -i d/dz            (A_z = 0)
//   D^2 = D_x D_x + D_y D_y + D_z D_z = -(grad - iA)^2   (operador positivo)
//   D^4 = D^2 D^2,  D^6 = D^2 D^2 D^2
//
// Densidad de energia:
//   f = k6 |D (D^2 psi)|^2 - k4 |D^2 psi|^2 - k3 B_y Re[(D^2 psi)* D_x psi]
//       + |D psi|^2 + k1 B_y Re[psi* D_x psi] + 1/2 |psi|^4 - |psi|^2
//   donde |D phi|^2 = sum_j |D_j phi|^2.
//
// Derivada funcional:
//   dF/dpsi* = k6 D^6 psi - k4 D^4 psi - (k3 B_y/2) {D^2, D_x} psi
//              + D^2 psi - psi + |psi|^2 psi + k1 B_y D_x psi
//
// Condiciones de borde:
//   x, y : periodicas con traslacion magnetica (identicas al codigo previo)
//   z    : en cada cara, simultaneamente
//     a) k6 d_z D^2 psi = 0
//     b) -k4 D^2 psi - (k3 B_y/2) D_x psi + k6 D^4 psi = 0
//     c) d_z psi - k4 d_z D^2 psi - (k3 B_y/2) d_z D_x psi + k6 d_z D^4 psi = 0
//
// ------------------------------------------------------------------
// DISCRETIZACION (variacionalmente consistente)
//
// Las tres condiciones en z son las condiciones de borde NATURALES del
// funcional (son los coeficientes de d_z^2 dpsi*, d_z dpsi* y dpsi* en el
// termino de borde de la variacion, con psi libre en las caras). Por eso NO
// se imponen con puntos fantasma: se discretiza la ENERGIA sin imponer
// ninguna condicion en z, y el gradiente que se usa en el gradiente
// conjugado es la derivada EXACTA de esa energia discreta. En el minimo el
// gradiente discreto se anula en todos los nodos, incluidos los de borde, y
// las filas de borde de esas ecuaciones son la version discreta de a), b) y
// c) (exactamente como el codigo anterior obtenia Neumann d_z psi = 0 sin
// punto fantasma). Al final de la minimizacion se miden explicitamente los
// residuos de a), b), c) en las dos caras (check_boundary_conditions).
//
// Operadores discretos (N = nodos, w_k = pesos de trapecio en z: 1/2 en
// k=0 y k=Nz-1, 1 en el resto):
//   * Enlaces covariantes en x,y (links magneticos + twist de borde, igual
//     que antes). Diferencias hacia adelante G_x, G_y sobre enlaces.
//   * G_z: diferencia hacia adelante sobre los Nz-1 enlaces en z.
//   * L_xy: laplaciano covariante compacto en x,y ( = -G_x^+ G_x - G_y^+ G_y ).
//   * L_z : d^2/dz^2 en TODOS los nodos SIN imponer condicion de borde:
//           centrada en el interior; en las caras, centrada usando un valor
//           fantasma obtenido por EXTRAPOLACION polinomial de grado LZ_Q=7
//           desde el interior (no es una condicion de borde, solo reconstruye
//           la funcion). Con grado 3 seria la clasica (2,-5,4,-1), pero ese
//           error O(dx^2) localizado en la cara, amplificado por los
//           operadores anidados D^4 y d_z D^4, impide que b) y c) converjan
//           en el borde; con grado 7 si convergen (verificado contra una
//           solucion espectral del continuo).
//   * P = -(L_xy + L_z)  ->  phi = P psi  es  D^2 psi  en todos los nodos.
//   * Dx^c = -i (psi_{i+1} U - psi_{i-1} U^*)/(2dx): D_x centrada covariante
//           (hermitica), igual que el termino de Lifshitz anterior.
//
// Energia discreta (E = dx^3 * suma):
//   sum_nodos w_k [ |G_x psi|^2 + |G_y psi|^2 + k6 (|G_x phi|^2 + |G_y phi|^2)
//                   - k4 |phi|^2 - k3 B_y Re(phi* Dx psi)
//                   + k1 B_y Re(psi* Dx psi) + 1/2|psi|^4 - |psi|^2 ]
//   + sum_enlaces_z [ |G_z psi|^2 + k6 |G_z phi|^2 ]
//
// Con K = G^+ W G (= -w L_xy - L_z^Neumann) el gradiente exacto es
//   g = dE/dpsi* / dx^3
//     = K psi + k6 P^+ K P psi - k4 P^+ W P psi
//       - (k3 B_y/2) (P^+ W Dx psi + Dx W P psi)
//       + k1 B_y W Dx psi + W (|psi|^2 - 1) psi
// Lejos de las caras (a mas de LZ_Q+1 planos) W = 1 y P^+ = P = K = D^2, por lo que
//   g = k6 D^6 - k4 D^4 - (k3 B_y/2){D^2, D_x} + D^2 - 1 + |psi|^2 + k1 B_y D_x
// con el orden de los operadores respetado (D^2 y D_x no conmutan: el
// anticonmutador aparece solo, como P^+ W Dx + Dx W P). Se calcula en tres
// pasos: phi = P psi ; chi = k6 K phi - k4 W phi - (k3 B_y/2) W Dx psi ;
//        g = P^+ chi + K psi - (k3 B_y/2) W Dx phi + k1 B_y W Dx psi + W(|psi|^2-1)psi
//
// En x,y el simbolo de P, K es s = (4/dx^2) sin^2(k dx/2) >= 0 y el de la
// parte cuadratica es k6 s^3 - k4 s^2 + s + ..., igual que en el continuo:
// la energia discreta esta acotada inferiormente si k6 > 0.
// ------------------------------------------------------------------

struct Params {
    int Nx, Ny, Nz;
    double dx, Lx, Ly, By, Bz;
    double k1, k3, k4, k6;
};

// ===================== BEGIN DISCRETE OPERATORS =====================

HD inline int lin(const Params& p, int i, int j, int k) {
    return i + p.Nx * (j + p.Ny * k);
}

// Peso de trapecio en z
HD inline double zw(const Params& p, int k) {
    return (k == 0 || k == p.Nz - 1) ? 0.5 : 1.0;
}

// Link en x desde (i,j,k) a (i+1,j,k):  exp(-i B_y z dx), mas el twist de
// traslacion magnetica exp(i (B_z Lx y + qx Lx)) en el enlace que cruza el borde.
HD inline cd link_x(const Params& p, int i, int j, int k, double qx) {
    double ph = -p.By * (k * p.dx) * p.dx;
    if (i == p.Nx - 1) ph += p.Bz * p.Lx * (j * p.dx) + qx * p.Lx;
    return cd(cos(ph), sin(ph));
}

// Link en y desde (i,j,k) a (i,j+1,k):  exp(-i B_z x dx), mas twist qy Ly en el borde.
HD inline cd link_y(const Params& p, int i, int j, int k, double qy) {
    double ph = -p.Bz * (i * p.dx) * p.dx;
    if (j == p.Ny - 1) ph += qy * p.Ly;
    return cd(cos(ph), sin(ph));
}

// Vecinos transportados paralelamente al nodo (i,j,k)
HD inline cd nb_xp(const Params& p, const cd* f, int i, int j, int k, double qx) {
    int ip = (i + 1) % p.Nx;
    return link_x(p, i, j, k, qx) * f[lin(p, ip, j, k)];
}
HD inline cd nb_xm(const Params& p, const cd* f, int i, int j, int k, double qx) {
    int im = (i - 1 + p.Nx) % p.Nx;
    return conj(link_x(p, im, j, k, qx)) * f[lin(p, im, j, k)];
}
HD inline cd nb_yp(const Params& p, const cd* f, int i, int j, int k, double qy) {
    int jp = (j + 1) % p.Ny;
    return link_y(p, i, j, k, qy) * f[lin(p, i, jp, k)];
}
HD inline cd nb_ym(const Params& p, const cd* f, int i, int j, int k, double qy) {
    int jm = (j - 1 + p.Ny) % p.Ny;
    return conj(link_y(p, i, jm, k, qy)) * f[lin(p, i, jm, k)];
}

// Laplaciano covariante compacto en el plano x-y: (grad - iA)^2 restringido a x,y
HD inline cd lap_xy(const Params& p, const cd* f, int i, int j, int k, double qx, double qy) {
    cd f0 = f[lin(p, i, j, k)];
    return (nb_xp(p, f, i, j, k, qx) + nb_xm(p, f, i, j, k, qx)
          + nb_yp(p, f, i, j, k, qy) + nb_ym(p, f, i, j, k, qy) - 4.0 * f0) / (p.dx * p.dx);
}

// D_x = -i d/dx - B_y z, diferencia centrada covariante (hermitica)
HD inline cd cov_Dx(const Params& p, const cd* f, int i, int j, int k, double qx) {
    return cd(0.0, -1.0) * (nb_xp(p, f, i, j, k, qx) - nb_xm(p, f, i, j, k, qx)) / (2.0 * p.dx);
}

// Fila de borde de d^2/dz^2: es la diferencia centrada en la cara con el
// valor fantasma psi_{-1} obtenido por extrapolacion polinomial de grado
// LZ_Q desde los nodos 0..LZ_Q (NO es una condicion de borde: solo
// reconstruye la funcion). Con LZ_Q = 3 se obtiene la clasica (2,-5,4,-1).
// Se usa LZ_Q alto para que el error de truncamiento en la cara sea
// O(dx^(LZ_Q-1)) y las derivadas altas (D^4 psi, d_z D^4 psi) que entran en
// las condiciones b) y c) converjan tambien en el borde.
#ifndef LZ_Q
#define LZ_Q 7
#endif
HD inline double binom(int n, int r) {
    double b = 1.0;
    for (int t = 1; t <= r; t++) b = b * (n - r + t) / t;
    return b;
}
HD inline double lz_bnd(int o) {
    if (o < 0 || o > LZ_Q) return 0.0;
    // ghost: psi_{-1} = sum_m (-1)^m C(Q+1, m+1) psi_m
    double e = ((o % 2) ? -1.0 : 1.0) * binom(LZ_Q + 1, o + 1);
    double c = e;
    if (o == 0) c -= 2.0;
    if (o == 1) c += 1.0;
    return c;
}

// Elemento (fila m, columna c) de la matriz L_z * dx^2
HD inline double lz_coef(const Params& p, int m, int c) {
    if (m == 0)        return lz_bnd(c);
    if (m == p.Nz - 1) return lz_bnd(p.Nz - 1 - c);
    if (c == m)        return -2.0;
    if (c == m - 1 || c == m + 1) return 1.0;
    return 0.0;
}

// L_z f : d^2 f/dz^2 en todos los nodos, sin condicion de borde impuesta
HD inline cd lz_apply(const Params& p, const cd* f, int i, int j, int k) {
    cd s(0.0, 0.0);
    if (k == 0) {
        for (int o = 0; o <= LZ_Q; o++) s += lz_bnd(o) * f[lin(p, i, j, o)];
    } else if (k == p.Nz - 1) {
        for (int o = 0; o <= LZ_Q; o++) s += lz_bnd(o) * f[lin(p, i, j, p.Nz - 1 - o)];
    } else {
        s = f[lin(p, i, j, k - 1)] - 2.0 * f[lin(p, i, j, k)] + f[lin(p, i, j, k + 1)];
    }
    return s / (p.dx * p.dx);
}

// L_z^T f : transpuesta (adjunta) de L_z, necesaria para el gradiente exacto
HD inline cd lzT_apply(const Params& p, const cd* f, int i, int j, int k) {
    cd s(0.0, 0.0);
    for (int m = k - 1; m <= k + 1; m++)
        if (m >= 1 && m <= p.Nz - 2)
            s += lz_coef(p, m, k) * f[lin(p, i, j, m)];
    s += lz_coef(p, 0, k) * f[lin(p, i, j, 0)];
    s += lz_coef(p, p.Nz - 1, k) * f[lin(p, i, j, p.Nz - 1)];
    return s / (p.dx * p.dx);
}

// G_z^T G_z con signo: laplaciano en z truncado (solo los enlaces que existen)
HD inline cd lzN_apply(const Params& p, const cd* f, int i, int j, int k) {
    cd f0 = f[lin(p, i, j, k)];
    cd s(0.0, 0.0);
    if (k < p.Nz - 1) s += f[lin(p, i, j, k + 1)] - f0;
    if (k > 0)        s += f[lin(p, i, j, k - 1)] - f0;
    return s / (p.dx * p.dx);
}

// P f = D^2 f  (con L_z descentrada en las caras)
HD inline cd apply_P(const Params& p, const cd* f, int i, int j, int k, double qx, double qy) {
    return -lap_xy(p, f, i, j, k, qx, qy) - lz_apply(p, f, i, j, k);
}
// P^+ f
HD inline cd apply_PT(const Params& p, const cd* f, int i, int j, int k) {
    return -lap_xy(p, f, i, j, k, 0.0, 0.0) - lzT_apply(p, f, i, j, k);
}
// K f = G^+ W G f  (gradiente de sum w |G f|^2)
HD inline cd apply_K(const Params& p, const cd* f, int i, int j, int k) {
    return -zw(p, k) * lap_xy(p, f, i, j, k, 0.0, 0.0) - lzN_apply(p, f, i, j, k);
}

// Densidad de energia discreta en el nodo (i,j,k) (sin el factor dx^3).
// phi debe ser P psi calculado con los mismos (qx,qy).
HD inline double energy_density(const Params& p, const cd* psi, const cd* phi,
                                int i, int j, int k, double qx, double qy) {
    int id = lin(p, i, j, k);
    cd psi0 = psi[id];
    cd phi0 = phi[id];
    double w = zw(p, k);

    cd Gx_psi = (nb_xp(p, psi, i, j, k, qx) - psi0) / p.dx;
    cd Gy_psi = (nb_yp(p, psi, i, j, k, qy) - psi0) / p.dx;
    cd Gx_phi = (nb_xp(p, phi, i, j, k, qx) - phi0) / p.dx;
    cd Gy_phi = (nb_yp(p, phi, i, j, k, qy) - phi0) / p.dx;
    cd Dx_psi = cov_Dx(p, psi, i, j, k, qx);

    double a2 = norm(psi0);

    double e = norm(Gx_psi) + norm(Gy_psi)
             + p.k6 * (norm(Gx_phi) + norm(Gy_phi))
             - p.k4 * norm(phi0)
             - p.k3 * p.By * (conj(phi0) * Dx_psi).real()
             + p.k1 * p.By * (conj(psi0) * Dx_psi).real()
             + 0.5 * a2 * a2 - a2;
    e *= w;

    // Enlaces en z (A_z = 0, sin link): solo los Nz-1 que existen
    if (k < p.Nz - 1) {
        int idz = lin(p, i, j, k + 1);
        cd Gz_psi = (psi[idz] - psi0) / p.dx;
        cd Gz_phi = (phi[idz] - phi0) / p.dx;
        e += norm(Gz_psi) + p.k6 * norm(Gz_phi);
    }
    return e;
}

// Paso 2 del gradiente: chi = k6 K phi - k4 W phi - (k3 B_y/2) W Dx psi
HD inline cd grad_chi(const Params& p, const cd* psi, const cd* phi, int i, int j, int k) {
    double w = zw(p, k);
    return p.k6 * apply_K(p, phi, i, j, k)
         - p.k4 * w * phi[lin(p, i, j, k)]
         - 0.5 * p.k3 * p.By * w * cov_Dx(p, psi, i, j, k, 0.0);
}

// Paso 3: g = P^+ chi + K psi - (k3 B_y/2) W Dx phi + k1 B_y W Dx psi + W(|psi|^2-1)psi
HD inline cd grad_final(const Params& p, const cd* psi, const cd* phi, const cd* chi,
                        int i, int j, int k) {
    double w = zw(p, k);
    cd psi0 = psi[lin(p, i, j, k)];
    return apply_PT(p, chi, i, j, k)
         + apply_K(p, psi, i, j, k)
         - 0.5 * p.k3 * p.By * w * cov_Dx(p, phi, i, j, k, 0.0)
         + p.k1 * p.By * w * cov_Dx(p, psi, i, j, k, 0.0)
         + w * (norm(psi0) - 1.0) * psi0;
}

// ====================== END DISCRETE OPERATORS ======================


int Nx, Ny, Nz, Nv;
double dx, Lx, Ly, By, Bz;
double k1, k3, k4, k6;
Params prm;

// Buffers de trabajo para phi = D^2 psi y chi (se usan en energia y gradiente)
cd* g_phi_buf = nullptr;
cd* g_chi_buf = nullptr;
dim3 g_Blocks, g_Threads;

inline int hidx(int i,int j,int k){ return i + Nx*(j + Ny*k); }


__global__ void kernel_apply_P(Params p, const cd* f, cd* out, double qx, double qy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;
    out[lin(p, i, j, k)] = apply_P(p, f, i, j, k, qx, qy);
}

__global__ void kernel_grad_chi(Params p, const cd* psi, const cd* phi, cd* chi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;
    chi[lin(p, i, j, k)] = grad_chi(p, psi, phi, i, j, k);
}

__global__ void kernel_grad_final(Params p, const cd* psi, const cd* phi, const cd* chi, cd* grad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;
    grad[lin(p, i, j, k)] = grad_final(p, psi, phi, chi, i, j, k);
}


struct energy_density_functor {
    Params p;
    const cd* psi;
    const cd* phi;
    double qx, qy;

    energy_density_functor(Params _p, const cd* _psi, const cd* _phi, double _qx, double _qy)
        : p(_p), psi(_psi), phi(_phi), qx(_qx), qy(_qy) {}

    __device__ double operator()(int id) const {
        int i = id % p.Nx;
        int t = id / p.Nx;
        int j = t % p.Ny;
        int k = t / p.Ny;
        return energy_density(p, psi, phi, i, j, k, qx, qy);
    }
};


// Energia total. (qx,qy) != 0 solo para el calculo de stiffness (twist extra
// en los enlaces de borde); la minimizacion usa qx = qy = 0.
double compute_energy(const cd* d_psi, double qx = 0.0, double qy = 0.0) {
    kernel_apply_P<<<g_Blocks, g_Threads>>>(prm, d_psi, g_phi_buf, qx, qy);
    cudaDeviceSynchronize();

    energy_density_functor f(prm, d_psi, g_phi_buf, qx, qy);
    double total_energy = thrust::transform_reduce(
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(Nx * Ny * Nz),
        f,
        double(0.0),
        thrust::plus<double>()
    );
    return total_energy * dx * dx * dx;   // elemento de volumen 3D
}

double compute_energy_twisted(const cd* d_psi, double qx, double qy) {
    return compute_energy(d_psi, qx, qy);
}


// Gradiente exacto de la energia discreta: grad = dE/dpsi* / dx^3
void compute_gradient(const cd* psi, cd* grad) {
    kernel_apply_P<<<g_Blocks, g_Threads>>>(prm, psi, g_phi_buf, 0.0, 0.0);
    kernel_grad_chi<<<g_Blocks, g_Threads>>>(prm, psi, g_phi_buf, g_chi_buf);
    kernel_grad_final<<<g_Blocks, g_Threads>>>(prm, psi, g_phi_buf, g_chi_buf, grad);
    cudaDeviceSynchronize();
}


cd dot(const cd* d_a, const cd* d_b)
{
    auto pa = thrust::device_pointer_cast(d_a);
    auto pb = thrust::device_pointer_cast(d_b);
    int NTOT = Nx * Ny * Nz;

    return thrust::inner_product(
        pa, pa + NTOT,
        pb,
        cd(0.0, 0.0),
        thrust::plus<cd>(),
        [] __device__ (cd a, cd b) {return thrust::conj(a) * b;} );
}


double wrap_angle(double dtheta) {
    double r = std::remainder(dtheta, 2.0 * M_PI);
    if (r <= -M_PI)
        r += 2.0 * M_PI;
    return r;
}


// Vorticidad por plaqueta en el plano x-y, repetida para cada plano z.
// El campo en el plano (B_z) es el unico que atraviesa la plaqueta x-y,
// por lo que la formula es identica a la version 2D en cada capa z
// (el link U_x, al depender solo de z, aporta fases iguales y opuestas
// en los lados superior/inferior de la plaqueta y se cancela exactamente,
// como corresponde a que B_y no tiene flujo a traves del plano x-y).
int count_vortices(const std::vector<cd>& psi) {
    int total = 0;

    for(int k = 0; k < Nz; k++) {
        for(int j = 0; j < Ny; j++) {
            for(int i = 0; i < Nx; i++) {
                int ip = (i + 1) % Nx;
                int jp = (j + 1) % Ny;

                double t00 = thrust::arg(psi[hidx(i,  j,  k)]);
                double t10 = thrust::arg(psi[hidx(ip, j,  k)]);
                double t11 = thrust::arg(psi[hidx(ip, jp, k)]);
                double t01 = thrust::arg(psi[hidx(i,  jp, k)]);

                if (i == Nx - 1) {
                    double twist_j  = Bz * Lx * (j * dx);
                    double twist_jp = Bz * Lx * (jp * dx);

                    t10 = thrust::arg(psi[hidx(ip, j,  k)] * thrust::exp(cd(0, twist_j)));
                    t11 = thrust::arg(psi[hidx(ip, jp, k)] * thrust::exp(cd(0, twist_jp)));
                }

                double sum = 0.0;
                sum += wrap_angle(t10 - t00);
                sum += wrap_angle(t11 - t10);
                sum += wrap_angle(t01 - t11);
                sum += wrap_angle(t00 - t01);
                total += (int)std::round(sum / (2.0 * M_PI));
            }
        }
    }
    return total;
}


double compute_betaA(const std::vector<cd>& psi)
{
    double s2=0.0,s4=0.0;
    for(auto& p:psi){
        double a2=thrust::norm(p);
        s2+=a2;
        s4+=a2*a2;
    }
    s2/=psi.size();
    s4/=psi.size();
    return s4/(s2*s2);
}


void write_field(const std::vector<cd>& psi,
                 const std::string& fname,
                 bool density)
{
    std::ofstream file(fname);
    file<<std::setprecision(14);

    for(int k=0;k<Nz;k++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                int id=hidx(i,j,k);
                double x=i*dx;
                double y=j*dx;
                double z=k*dx;
                double val = density ?
                             thrust::norm(psi[id]) :
                             thrust::arg(psi[id]);
                file<<x<<" "<<y<<" "<<z<<" "<<val<<"\n";
            }
            file<<"\n";
        }
        file<<"\n";
    }
}


// Busqueda lineal con condiciones de Wolfe fuertes (biseccion). Arranca en
// alpha_init (el paso aceptado en la iteracion anterior) y, mientras no haya
// cota superior, expande duplicando alpha. Los terminos k6 D^6 hacen el
// problema mucho mas rigido (autovalores ~ k6/dx^6), por lo que el paso
// optimo puede ser muy chico: arrancar siempre en alpha=1 desperdiciaria
// muchas evaluaciones de energia.
double line_search_wolfe(
    const cd* psi,
    const cd* dir,
    const cd* grad,
    cd* trial,
    cd* g_trial,
    double alpha_init)
{
    const double c1 = 1e-4;
    const double c2 = 0.1;
    const double alpha_max = 10.0;
    double alpha = std::min(alpha_init, alpha_max);
    double alpha_lo = 0.0, alpha_hi = -1.0;   // alpha_hi < 0: sin cota superior aun
    int NTOT = Nx * Ny * Nz;

    auto ppsi = thrust::device_pointer_cast(psi);
    auto ptrial = thrust::device_pointer_cast(trial);
    auto pdir = thrust::device_pointer_cast(dir);

    double E0 = compute_energy(psi);
    double slope0 = dot(grad, dir).real();

    for(int iter = 0; iter < 60; iter++) {

        thrust::transform(ppsi, ppsi + NTOT,
                  pdir, ptrial,
                  [alpha] __device__ (cd p, cd d){ return p + alpha * d; });

        double E = compute_energy(trial);

        if(!(E <= E0 + c1 * alpha * slope0)) {   // tambien atrapa NaN
            alpha_hi = alpha;
            alpha = 0.5 * (alpha_lo + alpha_hi);
            continue;
        }

        compute_gradient(trial, g_trial);
        double slope = dot(g_trial, dir).real();

        if(std::abs(slope) <= c2 * std::abs(slope0))
            return alpha;

        if(slope >= 0)
            alpha_hi = alpha;
        else
            alpha_lo = alpha;

        if (alpha_hi > 0.0)
            alpha = 0.5 * (alpha_lo + alpha_hi);
        else if (alpha >= alpha_max)
            return alpha;                         // Armijo ok en el paso maximo
        else
            alpha = std::min(2.0 * alpha, alpha_max);
    }
    // Sin Wolfe estricto: devolver el mayor paso que cumplio Armijo
    return alpha_lo > 0.0 ? alpha_lo : alpha;
}


struct polak_ribiere_op {
    __host__ __device__
    cd operator()(const cd& g, const cd& g_old) const {
        return thrust::conj(g) * (g - g_old);
    }
};


// Verificacion del gradiente: compara la derivada direccional de la energia
// (diferencias centradas) con 2 dx^3 Re<g, d>. Debe dar un error relativo
// pequeno (~1e-6 o menor) si el gradiente es la derivada exacta de la energia.
void check_gradient(const cd* psi, cd* grad, cd* trial) {
    int NTOT = Nx * Ny * Nz;
    compute_gradient(psi, grad);

    // Direccion de prueba: d = (1 + i/2) * exp(i 0.3 id) (acotada, no trivial)
    thrust::device_vector<cd> d_dir(NTOT);
    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(NTOT),
                      d_dir.begin(),
                      [] __device__ (int id) {
                          double a = 0.3 * id + 0.7 * sin(0.11 * id);
                          return cd(1.0, 0.5) * cd(cos(a), sin(a));
                      });
    const cd* dir = thrust::raw_pointer_cast(d_dir.data());

    double eps = 1e-5;
    auto ppsi = thrust::device_pointer_cast(psi);
    auto ptrial = thrust::device_pointer_cast(trial);
    thrust::transform(ppsi, ppsi + NTOT, d_dir.begin(), ptrial,
                      [eps] __device__ (cd p, cd d){ return p + eps * d; });
    double Ep = compute_energy(trial);
    thrust::transform(ppsi, ppsi + NTOT, d_dir.begin(), ptrial,
                      [eps] __device__ (cd p, cd d){ return p - eps * d; });
    double Em = compute_energy(trial);

    double fd = (Ep - Em) / (2.0 * eps);
    double an = 2.0 * dx * dx * dx * dot(grad, dir).real();
    std::cout << "[check_gradient] dE/de (dif. finitas) = " << fd
              << "   2 dx^3 Re<g,d> = " << an
              << "   error relativo = " << std::abs(fd - an) / std::max(std::abs(an), 1e-300)
              << "\n";
}


// Mide los residuos de las condiciones de borde a), b), c) en las caras z=0
// y z=(Nz-1)dx, con D^2 psi = P psi, D^4 psi = P P psi (los mismos operadores
// de la energia) y d/dz descentrada de 8 puntos (orden 7). Se reporta
// el valor RMS del residuo y, como escala, la suma de los RMS de cada
// termino por separado (residuo relativo = RMS(residuo)/escala).
void check_boundary_conditions(const thrust::device_vector<cd>& d_psi) {
    int NTOT = Nx * Ny * Nz;
    const cd* psi_ptr = thrust::raw_pointer_cast(d_psi.data());
    thrust::device_vector<cd> d_phi(NTOT), d_d4(NTOT);
    cd* phi_ptr = thrust::raw_pointer_cast(d_phi.data());
    cd* d4_ptr  = thrust::raw_pointer_cast(d_d4.data());

    kernel_apply_P<<<g_Blocks, g_Threads>>>(prm, psi_ptr, phi_ptr, 0.0, 0.0);   // D^2 psi
    kernel_apply_P<<<g_Blocks, g_Threads>>>(prm, phi_ptr, d4_ptr, 0.0, 0.0);    // D^4 psi
    cudaDeviceSynchronize();

    std::vector<cd> psi(NTOT), phi(NTOT), d4(NTOT), dxpsi(NTOT);
    thrust::copy(d_psi.begin(), d_psi.end(), psi.begin());
    thrust::copy(d_phi.begin(), d_phi.end(), phi.begin());
    thrust::copy(d_d4.begin(),  d_d4.end(),  d4.begin());
    for (int k = 0; k < Nz; k++)
        for (int j = 0; j < Ny; j++)
            for (int i = 0; i < Nx; i++)
                dxpsi[hidx(i,j,k)] = cov_Dx(prm, psi.data(), i, j, k, 0.0);

    double h = 0.5 * k3 * By;
    for (int face = 0; face < 2; face++) {
        int k0 = (face == 0) ? 0 : Nz - 1;
        int s  = (face == 0) ? 1 : -1;       // direccion hacia el interior
        // d/dz descentrada de 2do orden en la cara
        // d/dz descentrada de 8 puntos: sum_{n=1}^{7} (-1)^{n+1} Delta^n f / n
        const double w8[8] = {-363.0/140.0, 7.0, -21.0/2.0, 35.0/3.0,
                              -35.0/4.0, 21.0/5.0, -7.0/6.0, 1.0/7.0};
        auto dz = [&](const std::vector<cd>& f, int i, int j) {
            cd v(0.0, 0.0);
            for (int m = 0; m < 8; m++) v += w8[m] * f[hidx(i,j,k0+m*s)];
            return v / (dx * s);
        };
        double ra = 0, rb = 0, rc = 0;
        double sb[3] = {0,0,0}, sc[4] = {0,0,0,0}, sa = 0;
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                int id = hidx(i,j,k0);
                cd a_t = k6 * dz(phi, i, j);

                cd b1 = -k4 * phi[id], b2 = -h * dxpsi[id], b3 = k6 * d4[id];

                cd c1 = dz(psi, i, j), c2 = -k4 * dz(phi, i, j);
                cd c3 = -h * dz(dxpsi, i, j), c4 = k6 * dz(d4, i, j);

                ra += thrust::norm(a_t);           sa += thrust::norm(dz(phi, i, j));
                rb += thrust::norm(b1 + b2 + b3);
                sb[0] += thrust::norm(b1); sb[1] += thrust::norm(b2); sb[2] += thrust::norm(b3);
                rc += thrust::norm(c1 + c2 + c3 + c4);
                sc[0] += thrust::norm(c1); sc[1] += thrust::norm(c2);
                sc[2] += thrust::norm(c3); sc[3] += thrust::norm(c4);
            }
        }
        double n = Nx * Ny;
        auto rms = [n](double v){ return std::sqrt(v / n); };
        double scale_b = rms(sb[0]) + rms(sb[1]) + rms(sb[2]);
        double scale_c = rms(sc[0]) + rms(sc[1]) + rms(sc[2]) + rms(sc[3]);
        std::cout << "[BC z=" << (face == 0 ? "0" : "Lz") << "] "
                  << "a) RMS(k6 dz D^2psi) = " << rms(ra)
                  << " (RMS dz D^2psi = " << rms(sa) << ")\n"
                  << "           b) RMS = " << rms(rb) << "  escala = " << scale_b
                  << "  relativo = " << (scale_b > 0 ? rms(rb) / scale_b : 0.0) << "\n"
                  << "           c) RMS = " << rms(rc) << "  escala = " << scale_c
                  << "  relativo = " << (scale_c > 0 ? rms(rc) / scale_c : 0.0) << "\n";
    }
    std::cout << "  (Las condiciones son naturales: se cumplen en el minimo discreto "
                 "con error de discretizacion que tiende a cero al refinar dx.)\n";
}


int main(int argc, char* argv[]) {

    if (argc < 8) {
        std::cerr << "Uso: " << argv[0]
                  << " Nx Ny Nz Nv dx B_y seed.dat [k1=1] [k3=0] [k4=0] [k6=0]   (Nz >= "
                  << 2 * (LZ_Q + 1) << ")" << std::endl;
        std::cerr << "  f = k6|D(D^2 psi)|^2 - k4|D^2 psi|^2 - k3 By Re[(D^2 psi)* D_x psi]\n"
                     "      + |D psi|^2 + k1 By Re[psi* D_x psi] + 1/2|psi|^4 - |psi|^2\n"
                     "  (k1=0, k3=k4=k6=0 -> GL estandar; k1=1 reproduce el Lifshitz anterior)"
                  << std::endl;
        return 1;
    }

    Nx = std::stoi(argv[1]);
    Ny = std::stoi(argv[2]);
    Nz = std::stoi(argv[3]);
    Nv = std::stoi(argv[4]);
    dx = std::stod(argv[5]);
    By = std::stod(argv[6]);
    std::ifstream seed_file(argv[7]);

    // Constantes del funcional (posiciones 8..11). k1 ocupa el lugar del
    // antiguo flag lifshitz_on: k1 = 1 / 0 equivale a Lifshitz activado / apagado.
    k1 = (argc >= 9)  ? std::stod(argv[8])  : 1.0;
    k3 = (argc >= 10) ? std::stod(argv[9])  : 0.0;
    k4 = (argc >= 11) ? std::stod(argv[10]) : 0.0;
    k6 = (argc >= 12) ? std::stod(argv[11]) : 0.0;

    if (k1 < 0 || k3 < 0 || k4 < 0 || k6 < 0) {
        std::cerr << "Error: todas las constantes k1, k3, k4, k6 deben ser >= 0." << std::endl;
        return 1;
    }
    if (k6 == 0.0 && (k3 != 0.0 || k4 != 0.0)) {
        std::cerr << "Error: con k3 o k4 > 0 hace falta k6 > 0 (si no la energia no esta "
                     "acotada inferiormente a escala de la grilla)." << std::endl;
        return 1;
    }
    if (Nz < 2 * (LZ_Q + 1)) {
        std::cerr << "Error: hace falta Nz >= " << 2 * (LZ_Q + 1)
                  << " (la fila de borde de d^2/dz^2 usa " << LZ_Q + 1 << " planos por cara)." << std::endl;
        return 1;
    }

    Lx = Nx * dx;
    Ly = Ny * dx;
    Bz = 2.0 * M_PI * Nv / (Lx * Ly);   // flujo cuantizado a traves del plano x-y

    prm.Nx = Nx; prm.Ny = Ny; prm.Nz = Nz;
    prm.dx = dx; prm.Lx = Lx; prm.Ly = Ly; prm.By = By; prm.Bz = Bz;
    prm.k1 = k1; prm.k3 = k3; prm.k4 = k4; prm.k6 = k6;

    int NTOT = Nx * Ny * Nz;

    std::cout << "Nx=" << Nx << " Ny=" << Ny << " Nz=" << Nz << " Nv=" << Nv
              << " dx=" << dx << " B_z(fuera de plano)=" << Bz
              << " B_y(en el plano)=" << By
              << " -- z abierta (condiciones naturales), x,y periodicas\n"
              << "k1=" << k1 << " k3=" << k3 << " k4=" << k4 << " k6=" << k6 << "\n";

    std::vector<cd> psi (NTOT);
    thrust::device_vector<cd> d_psi(NTOT);
    thrust::device_vector<cd> d_grad(NTOT);
    thrust::device_vector<cd> d_grad_old(NTOT);
    thrust::device_vector<cd> d_dir(NTOT);
    thrust::device_vector<cd> trial(NTOT);
    thrust::device_vector<cd> g_trial(NTOT);
    thrust::device_vector<cd> d_phi_buf(NTOT);
    thrust::device_vector<cd> d_chi_buf(NTOT);

    cd* psi_ptr  = thrust::raw_pointer_cast(d_psi.data());
    cd* grad_ptr = thrust::raw_pointer_cast(d_grad.data());
    cd* dir_ptr = thrust::raw_pointer_cast(d_dir.data());
    cd* grad_old_ptr = thrust::raw_pointer_cast(d_grad_old.data());
    cd* trial_ptr = thrust::raw_pointer_cast(trial.data());
    cd* g_trial_ptr = thrust::raw_pointer_cast(g_trial.data());
    g_phi_buf = thrust::raw_pointer_cast(d_phi_buf.data());
    g_chi_buf = thrust::raw_pointer_cast(d_chi_buf.data());

    g_Threads = dim3(8, 8, 8);
    g_Blocks  = dim3((Nx + 7) / 8, (Ny + 7) / 8, (Nz + 7) / 8);

    int max_iter = 80000;
    double tol = 1e-10 * NTOT;
    int restart_period = 100;

    for(auto& p : psi){
      double r, im;
      if (!(seed_file >> r >> im)) break;
      p = cd(r, im);
    }
    d_psi = psi;

    check_gradient(psi_ptr, grad_ptr, trial_ptr);

    std::ofstream file("grad3_gpu.dat");
    file<<std::setprecision(14);


    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);
    cudaEventRecord(t_start);

    double alpha_prev = 1.0;

    for(int k = 0; k < max_iter; k++){
        compute_gradient(psi_ptr, grad_ptr);

        double norm2 = dot(grad_ptr, grad_ptr).real();

        if(k % 10 == 0){
            double E = compute_energy(psi_ptr);
            std::cout << "Iter " << k << " |grad|^2 = " << norm2 << " Energy: " << E << "\n";
            file << k << " " << norm2 << " " << E;
        }

        if(norm2 < tol) break;

        double beta = 0.0;

        if(k == 0 || (k % restart_period == 0))
            thrust::transform(d_grad.begin(), d_grad.end(), d_dir.begin(), thrust::negate<cd>());
        else {
            double num = (thrust::inner_product(
            d_grad.begin(), d_grad.end(),
            d_grad_old.begin(),
            cd(0.0, 0.0),
            thrust::plus<cd>(),
            polak_ribiere_op() )).real();

            double den = dot(grad_old_ptr, grad_old_ptr).real();

            beta = std::max(0.0, num / den);

            thrust::transform(d_grad.begin(), d_grad.end(),
                  d_dir.begin(), d_dir.begin(),
                  [beta] __device__ (cd g, cd d){ return -g + beta * d; });

            // Si la direccion no es de descenso, reiniciar con -grad
            if (dot(grad_ptr, dir_ptr).real() >= 0.0) {
                beta = 0.0;
                thrust::transform(d_grad.begin(), d_grad.end(), d_dir.begin(), thrust::negate<cd>());
            }
        }

        double alpha = line_search_wolfe(psi_ptr, dir_ptr, grad_ptr, trial_ptr, g_trial_ptr,
                                         2.0 * alpha_prev);
        alpha_prev = alpha;

        if(k % 10 == 0)
            file << " " << alpha << " " << beta << "\n";


        thrust::transform(d_psi.begin(), d_psi.end(),
                  d_dir.begin(), d_psi.begin(),
                  [alpha] __device__ (cd p, cd d){ return p + alpha * d; });

        std::swap(d_grad, d_grad_old);
        grad_ptr     = thrust::raw_pointer_cast(d_grad.data());
        grad_old_ptr = thrust::raw_pointer_cast(d_grad_old.data());
    }

    cudaEventRecord(t_stop);
    cudaEventSynchronize(t_stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, t_start, t_stop);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_stop);
    std::cout << "Main loop elapsed time: " << elapsed_ms / 1000.0f << " s ("
              << elapsed_ms << " ms)\n";

    thrust::copy(d_psi.begin(), d_psi.end(), psi.begin());

    check_boundary_conditions(d_psi);

    int vort = count_vortices(psi);
    std::cout << "Final measured vortices (suma sobre todos los planos z): " << vort
              << "  (promedio por plano: " << (double)vort / Nz << ")\n";
    std::cout << "Final betaA: " << compute_betaA(psi) << "\n";
    write_field(psi, "density_gpu_3d.dat", true);
    write_field(psi, "phase_gpu_3d.dat", false);


    //Calculo de stiffness

    const int NQ = 13;
    double qy[NQ] = {-0.06,-0.05,-0.04,-0.03,-0.02,-0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06};
    double qx[NQ] = {0.0};
    double F[NQ] = {0.0};
    Params p = prm;
    for(int i = 0; i < NQ; i+=1){
      double qxi = qx[i], qyi = qy[i];
      thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(NTOT), d_psi.begin(), d_psi.begin(),
      [=] __device__ (int id, cd v){ return v * thrust::exp(cd(0.0, qxi*p.dx*(id % p.Nx) + qyi*p.dx*((id / p.Nx) % p.Ny))); });

      F[i] = compute_energy_twisted(psi_ptr, qxi, qyi);

      thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(NTOT), d_psi.begin(), d_psi.begin(),
      [=] __device__ (int id, cd v){ return v * thrust::exp(-cd(0.0, qxi*p.dx*(id % p.Nx) + qyi*p.dx*((id / p.Nx) % p.Ny)) ); });

      std::cout << "F = " << F[i] << " q= " << qy[i] <<std::endl;
    }

    return 0;
}
