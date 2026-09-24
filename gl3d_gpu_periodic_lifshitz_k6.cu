#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cassert>

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

// ------------------------------------------------------------------
// Geometria:
//   x,y : periodicas, con twist magnetico (torus + traslacion magnetica)
//   z   : abierta (caras libres en z=0 y z=(Nz-1)*dx)
//
// Potencial vector:  A = (B_y * z, B_z * x, 0)   ->  B = (0, B_y, B_z)
// Operador covariante: D = -i grad - A,  D_x = -i d/dx - A_x.
//
// Densidad funcional:
//   f = k6 |D(D^2 psi)|^2 - k4 |D^2 psi|^2 - k3 By Re[(D^2 psi)* D_x psi]
//       + |D psi|^2 + k1 By Re[psi* D_x psi] + 1/2 |psi|^4 - |psi|^2
//
// Derivada funcional:
//   dF/dpsi* = k6 D^6 psi - k4 D^4 psi - k3 By/2 {D^2, D_x} psi + D^2 psi
//              + k1 By D_x psi + (|psi|^2 - 1) psi
//
// Discretizacion (en la red, todo con los links magneticos):
//   * D_j psi  -> diferencia adelantada covariante sobre cada enlace
//                 (U_x = exp(-i B_y z dx), U_y = exp(-i B_z x dx), U_z = 1).
//   * D^2      -> K = sum_j D_j^+ D_j = -(laplaciano covariante): hermitico,
//                 asi D^4 = K^2, D^6 = K^3 y el orden de los D_j queda fijado.
//   * D_x      -> diferencia centrada covariante -i(U psi_{x+} - U^+ psi_{x-})/2dx
//                 (hermitica, como en el codigo anterior del termino k1).
//
// Caras en z: la energia discreta es la cuadratura del funcional con caras
// LIBRES, sin imponer ninguna condicion de borde:
//   * nivel 1 (|D psi|^2): enlaces en z solo entre planos existentes (Nz-1).
//   * nivel 2 (phi = D^2 psi, terminos k4 y k3): solo en los planos interiores
//     k = 1..Nz-2, donde la derivada segunda en z es centrada y NO necesita
//     punto fantasma. En los planos de borde phi no se define.
//   * nivel 3 (|D phi|^2, termino k6): enlaces en z solo entre planos interiores.
// El minimo de esta energia satisface, en el limite dx -> 0, las condiciones
// de borde naturales del funcional continuo en cada cara z:
//   a) k6 d_z D^2 psi = 0
//   b) -k4 D^2 psi - k3 By/2 D_x psi + k6 D^4 psi = 0
//   c) d_z psi - k3 By/2 d_z D_x psi + k6 d_z D^4 psi = 0
// (convergencia O(dx)). OJO: usar el laplaciano de grafo tambien en los planos
// de borde impondria d_z psi = 0 (Neumann) y NO estas condiciones.
//
// El gradiente es la derivada EXACTA de la energia discreta (asi el CG + Wolfe
// son consistentes). Con P = restriccion a planos interiores:
//   phi = P K psi,   chi = P [ k6 M phi - k4 phi - k3 By/2 D_x psi ]
//   grad = K (psi + chi) + D_x (k1 By psi - k3 By/2 phi) + (|psi|^2-1) psi
// con M = K restringido a los planos interiores (enlaces z solo entre ellos).
// En el bulk esto es exactamente la derivada funcional de arriba.
// ------------------------------------------------------------------

int Nx, Ny, Nz, Nv;
double dx, Lx, Ly, By, Bz, k1, k3, k4, k6;
cd *d_phi_buf, *d_chi_buf;   // buffers auxiliares (phi = D^2 psi, chi)
dim3 Blocks, Threads;

__constant__ int d_Nx, d_Ny, d_Nz;
__constant__ double d_dx, d_Lx, d_By, d_Bz, d_k1, d_k3, d_k4, d_k6;

inline int hidx(int i,int j,int k){ return i + Nx*(j + Ny*k); }


// Vecinos en x,y de f en (i,j,k), ya multiplicados por sus links y por los
// twists de las costuras periodicas (x: exp(i B_z Lx y), por A_y = B_z x).
// qx,qy: twist extra en las costuras, solo para el calculo de stiffness
// (0 en la minimizacion).
struct XY { cd xp, xm, yp, ym; };

__device__ XY nb_xy(const cd* f, int i, int j, int k, double qx, double qy) {
    int ip = (i + 1) % d_Nx, im = (i - 1 + d_Nx) % d_Nx;
    int jp = (j + 1) % d_Ny, jm = (j - 1 + d_Ny) % d_Ny;
    double y = j * d_dx, z = k * d_dx;

    cd Ux = thrust::exp(cd(0, -d_By * z * d_dx));
    cd Uy = thrust::exp(cd(0, -d_Bz * (i * d_dx) * d_dx));

    XY n;
    n.xp = Ux * f[ip + d_Nx*(j + d_Ny*k)];
    n.xm = thrust::conj(Ux) * f[im + d_Nx*(j + d_Ny*k)];
    if (i == d_Nx - 1) n.xp *= thrust::exp(cd(0,  d_Bz * d_Lx * y + qx * d_Lx));
    if (i == 0)        n.xm *= thrust::exp(cd(0, -d_Bz * d_Lx * y - qx * d_Lx));

    n.yp = Uy * f[i + d_Nx*(jp + d_Ny*k)];
    n.ym = thrust::conj(Uy) * f[i + d_Nx*(jm + d_Ny*k)];
    if (j == d_Ny - 1) n.yp *= thrust::exp(cd(0,  qy * d_Ny * d_dx));
    if (j == 0)        n.ym *= thrust::exp(cd(0, -qy * d_Ny * d_dx));
    return n;
}

// K f = D^2 f = -(laplaciano covariante), con enlaces en z solo entre
// planos klo..khi (klo=0,khi=Nz-1: red completa; 1,Nz-2: planos interiores).
__device__ cd Kop(const cd* f, int i, int j, int k, int klo, int khi, double qx, double qy) {
    int id = i + d_Nx*(j + d_Ny*k);
    cd f0 = f[id];
    XY n = nb_xy(f, i, j, k, qx, qy);
    cd s = n.xp + n.xm + n.yp + n.ym - 4.0 * f0;
    if (k < khi) s += f[id + d_Nx*d_Ny] - f0;
    if (k > klo) s += f[id - d_Nx*d_Ny] - f0;
    return -s / (d_dx * d_dx);
}

// D_x f centrada y covariante: -i (f_x+ - f_x-) / (2 dx)
__device__ cd Dxop(const XY& n) {
    return cd(0.0, -1.0) * (n.xp - n.xm) / (2.0 * d_dx);
}

__device__ bool interior(int k) { return k > 0 && k < d_Nz - 1; }

#define KERNEL_INDEX                                         \
    int i = blockIdx.x * blockDim.x + threadIdx.x;           \
    int j = blockIdx.y * blockDim.y + threadIdx.y;           \
    int k = blockIdx.z * blockDim.z + threadIdx.z;           \
    if (i >= d_Nx || j >= d_Ny || k >= d_Nz) return;         \
    int id = i + d_Nx*(j + d_Ny*k);

// phi = P K psi  (cero en los planos de borde)
__global__ void compute_phi(const cd* psi, cd* phi, double qx, double qy) {
    KERNEL_INDEX
    phi[id] = interior(k) ? Kop(psi, i, j, k, 0, d_Nz - 1, qx, qy) : cd(0.0, 0.0);
}

// chi = P [ k6 M phi - k4 phi - k3 By/2 D_x psi ]
__global__ void compute_chi(const cd* psi, const cd* phi, cd* chi) {
    KERNEL_INDEX
    if (!interior(k)) { chi[id] = cd(0.0, 0.0); return; }
    cd Dxpsi = Dxop(nb_xy(psi, i, j, k, 0.0, 0.0));
    chi[id] = d_k6 * Kop(phi, i, j, k, 1, d_Nz - 2, 0.0, 0.0)
            - d_k4 * phi[id] - 0.5 * d_k3 * d_By * Dxpsi;
}

// grad = K (psi + chi) + D_x (k1 By psi - k3 By/2 phi) + (|psi|^2 - 1) psi
__global__ void compute_grad(const cd* psi, const cd* phi, const cd* chi, cd* grad) {
    KERNEL_INDEX
    cd psi0 = psi[id];
    cd Dxpsi = Dxop(nb_xy(psi, i, j, k, 0.0, 0.0));
    cd Dxphi = Dxop(nb_xy(phi, i, j, k, 0.0, 0.0));
    grad[id] = Kop(psi, i, j, k, 0, d_Nz - 1, 0.0, 0.0)
             + Kop(chi, i, j, k, 0, d_Nz - 1, 0.0, 0.0)
             + d_k1 * d_By * Dxpsi - 0.5 * d_k3 * d_By * Dxphi
             + (thrust::norm(psi0) - 1.0) * psi0;
}

void compute_gradient(const cd* psi, cd* grad) {
    compute_phi<<<Blocks,Threads>>>(psi, d_phi_buf, 0.0, 0.0);
    compute_chi<<<Blocks,Threads>>>(psi, d_phi_buf, d_chi_buf);
    compute_grad<<<Blocks,Threads>>>(psi, d_phi_buf, d_chi_buf, grad);
    cudaDeviceSynchronize();
}


struct energy_density_functor {
    const cd* psi;
    const cd* phi;
    double qx, qy;

    energy_density_functor(const cd* _psi, const cd* _phi, double _qx, double _qy)
        : psi(_psi), phi(_phi), qx(_qx), qy(_qy) {}

    __device__ double operator()(int id) const {
        int i = id % d_Nx;
        int t = id / d_Nx;
        int j = t % d_Ny;
        int k = t / d_Ny;
        int NxNy = d_Nx * d_Ny;
        double h2 = d_dx * d_dx;

        cd psi0 = psi[id];
        XY n = nb_xy(psi, i, j, k, qx, qy);

        // |D psi|^2: enlaces adelantados; en z solo si existe el plano k+1
        double e = (thrust::norm(n.xp - psi0) + thrust::norm(n.yp - psi0)) / h2;
        if (k < d_Nz - 1)
            e += thrust::norm(psi[id + NxNy] - psi0) / h2;

        // 1/2 |psi|^4 - |psi|^2
        double a2 = thrust::norm(psi0);
        e += 0.5 * a2 * a2 - a2;

        // k1 By Re[psi* D_x psi]
        cd Dxpsi = Dxop(n);
        e += d_k1 * d_By * (thrust::conj(psi0) * Dxpsi).real();

        // Terminos con phi = D^2 psi: solo en planos interiores
        if (k > 0 && k < d_Nz - 1) {
            cd phi0 = phi[id];
            XY m = nb_xy(phi, i, j, k, qx, qy);
            e += -d_k4 * thrust::norm(phi0)
                 - d_k3 * d_By * (thrust::conj(phi0) * Dxpsi).real();
            double g3 = thrust::norm(m.xp - phi0) + thrust::norm(m.yp - phi0);
            if (k < d_Nz - 2)
                g3 += thrust::norm(phi[id + NxNy] - phi0);
            e += d_k6 * g3 / h2;
        }
        return e;
    }
};


// qx,qy != 0 solo para el stiffness (twist extra en las costuras).
double compute_energy(const cd* d_psi, double qx = 0.0, double qy = 0.0) {
    compute_phi<<<Blocks,Threads>>>(d_psi, d_phi_buf, qx, qy);
    cudaDeviceSynchronize();
    energy_density_functor f(d_psi, d_phi_buf, qx, qy);
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


// Chequeo: el gradiente debe ser la derivada exacta de la energia discreta.
// dE/deps en psi + eps*d  =  2 dx^3 Re<grad, d>. Se compara con la diferencia
// centrada de la energia en una direccion pseudo-aleatoria d.
void check_gradient(const cd* psi, cd* grad, cd* d, cd* trial) {
    int NTOT = Nx * Ny * Nz;
    auto pd = thrust::device_pointer_cast(d);
    auto ppsi = thrust::device_pointer_cast(psi);
    auto ptrial = thrust::device_pointer_cast(trial);
    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(NTOT), pd,
        [] __device__ (int id){ return cd(sin(0.37 * id + 1.0), cos(1.71 * id)); });

    compute_gradient(psi, grad);
    double exact = 2.0 * dx * dx * dx * dot(grad, d).real();

    double eps = 1e-5;
    thrust::transform(ppsi, ppsi + NTOT, pd, ptrial, [eps] __device__ (cd p, cd q){ return p + eps * q; });
    double Ep = compute_energy(trial);
    thrust::transform(ppsi, ppsi + NTOT, pd, ptrial, [eps] __device__ (cd p, cd q){ return p - eps * q; });
    double Em = compute_energy(trial);
    double fd = (Ep - Em) / (2.0 * eps);

    std::cout << "Chequeo de gradiente: analitico = " << exact << "  diferencias = " << fd
              << "  error relativo = " << std::abs(exact - fd) / std::abs(fd) << "\n";
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


double line_search_wolfe(
    const cd* psi,
    const cd* dir,
    const cd* grad,
    cd* trial,
    cd* g_trial)
{
    double c1 = 1e-4;
    double c2 = 0.1;
    double alpha = 1.0;
    double alpha_lo = 0.0, alpha_hi = 10.0;
    int NTOT = Nx * Ny * Nz;

    auto ppsi = thrust::device_pointer_cast(psi);
    auto ptrial = thrust::device_pointer_cast(trial);
    auto pdir = thrust::device_pointer_cast(dir);

    double E0 = compute_energy(psi);
    double slope0 = dot(grad, dir).real();

    for(int iter = 0; iter < 30; iter++) {

        thrust::transform(ppsi, ppsi + NTOT,
                  pdir, ptrial,
                  [alpha] __device__ (cd p, cd d){ return p + alpha * d; });

        double E = compute_energy(trial);

        if(E > E0 + c1 * alpha * slope0) {
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

        alpha = 0.5 * (alpha_lo + alpha_hi);
    }
    return alpha;
}


struct polak_ribiere_op {
    __host__ __device__
    cd operator()(const cd& g, const cd& g_old) const {
        return thrust::conj(g) * (g - g_old);
    }
};


int main(int argc, char* argv[]) {

    if (argc < 12) {
        std::cerr << "Uso: " << argv[0]
                  << " Nx Ny Nz Nv dx B_y seed.dat k1 k3 k4 k6" << std::endl;
        return 1;
    }

    Nx = std::stoi(argv[1]);
    Ny = std::stoi(argv[2]);
    Nz = std::stoi(argv[3]);
    Nv = std::stoi(argv[4]);
    dx = std::stod(argv[5]);
    By = std::stod(argv[6]);
    std::ifstream seed_file(argv[7]);
    k1 = std::stod(argv[8]);
    k3 = std::stod(argv[9]);
    k4 = std::stod(argv[10]);
    k6 = std::stod(argv[11]);

    if (Nz < 4) {
        std::cerr << "Nz debe ser >= 4 (hacen falta planos interiores para D^2 y D^3)" << std::endl;
        return 1;
    }
    if (k1 < 0 || k3 < 0 || k4 < 0 || k6 < 0) {
        std::cerr << "Las constantes k1, k3, k4, k6 deben ser >= 0" << std::endl;
        return 1;
    }
    if (!seed_file) {
        std::cerr << "No se pudo abrir " << argv[7] << std::endl;
        return 1;
    }

    Lx = Nx * dx;
    Ly = Ny * dx;
    Bz = 2.0 * M_PI * Nv / (Lx * Ly);   // flujo cuantizado a traves del plano x-y

    int NTOT = Nx * Ny * Nz;

    std::cout << "Nx=" << Nx << " Ny=" << Ny << " Nz=" << Nz << " Nv=" << Nv
              << " dx=" << dx << " B_z(fuera de plano)=" << Bz
              << " B_y(en el plano)=" << By
              << " k1=" << k1 << " k3=" << k3 << " k4=" << k4 << " k6=" << k6
              << " -- z con caras libres, x,y periodicas\n";

    cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(int));
    cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(int));
    cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(int));
    cudaMemcpyToSymbol(d_dx, &dx, sizeof(double));
    cudaMemcpyToSymbol(d_Lx, &Lx, sizeof(double));
    cudaMemcpyToSymbol(d_By, &By, sizeof(double));
    cudaMemcpyToSymbol(d_Bz, &Bz, sizeof(double));
    cudaMemcpyToSymbol(d_k1, &k1, sizeof(double));
    cudaMemcpyToSymbol(d_k3, &k3, sizeof(double));
    cudaMemcpyToSymbol(d_k4, &k4, sizeof(double));
    cudaMemcpyToSymbol(d_k6, &k6, sizeof(double));

    std::vector<cd> psi (NTOT);
    thrust::device_vector<cd> d_psi(NTOT);
    thrust::device_vector<cd> d_grad(NTOT);
    thrust::device_vector<cd> d_grad_old(NTOT);
    thrust::device_vector<cd> d_dir(NTOT);
    thrust::device_vector<cd> trial(NTOT);
    thrust::device_vector<cd> g_trial(NTOT);
    thrust::device_vector<cd> d_phi(NTOT);
    thrust::device_vector<cd> d_chi(NTOT);

    cd* psi_ptr  = thrust::raw_pointer_cast(d_psi.data());
    cd* grad_ptr = thrust::raw_pointer_cast(d_grad.data());
    cd* dir_ptr = thrust::raw_pointer_cast(d_dir.data());
    cd* grad_old_ptr = thrust::raw_pointer_cast(d_grad_old.data());
    cd* trial_ptr = thrust::raw_pointer_cast(trial.data());
    cd* g_trial_ptr = thrust::raw_pointer_cast(g_trial.data());
    d_phi_buf = thrust::raw_pointer_cast(d_phi.data());
    d_chi_buf = thrust::raw_pointer_cast(d_chi.data());

    Threads = dim3(8, 8, 8);
    Blocks = dim3((Nx + 7) / 8, (Ny + 7) / 8, (Nz + 7) / 8);

    int max_iter = 80000;
    double tol = 1e-10 * NTOT;
    int restart_period = 100;

    for(auto& p : psi){
      double r, im;
      if (!(seed_file >> r >> im)) break;
      p = cd(r, im);
    }
    d_psi = psi;

    check_gradient(psi_ptr, grad_ptr, dir_ptr, trial_ptr);

    std::ofstream file("grad3_gpu.dat");
    file<<std::setprecision(14);


    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);
    cudaEventRecord(t_start);

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
        }

        double alpha = line_search_wolfe(psi_ptr, dir_ptr, grad_ptr, trial_ptr, g_trial_ptr);

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

    int vort = count_vortices(psi);
    std::cout << "Final measured vortices (suma sobre todos los planos z): " << vort
              << "  (promedio por plano: " << (double)vort / Nz << ")\n";
    std::cout << "Final betaA: " << compute_betaA(psi) << "\n";
    write_field(psi, "density_gpu_3d.dat", true);
    write_field(psi, "phase_gpu_3d.dat", false);


    // Calculo de stiffness: F(q) con psi -> psi exp(i q.r) y el twist extra
    // correspondiente en las costuras periodicas.
    const int NQ = 13;
    double qy[NQ] = {-0.06,-0.05,-0.04,-0.03,-0.02,-0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06};
    double qx[NQ] = {0.0};
    double F[NQ] = {0.0};
    for(int n = 0; n < NQ; n++){
      double qxn = qx[n], qyn = qy[n];
      thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(NTOT), d_psi.begin(), d_psi.begin(),
      [=] __device__ (int id, cd p){ return p * thrust::exp(cd(0.0, qxn*d_dx*(id % d_Nx) + qyn*d_dx*((id / d_Nx) % d_Ny))); });

      F[n] = compute_energy(psi_ptr, qxn, qyn);

      thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(NTOT), d_psi.begin(), d_psi.begin(),
      [=] __device__ (int id, cd p){ return p * thrust::exp(-cd(0.0, qxn*d_dx*(id % d_Nx) + qyn*d_dx*((id / d_Nx) % d_Ny))); });

      std::cout << "F = " << F[n] << " q= " << qyn << std::endl;
    }

    return 0;
}
