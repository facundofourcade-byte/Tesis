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
#include <thrust/transform.h>

using cd = thrust::complex<double>;

int Nx, Ny, Nz, Nv;
double dx, Lx, Ly, Bz, By;
double k1[2], k2[2], k3[2], k4[2], k6[2], alfa[2];
int kd;
cd *d_phi, *d_chi;   // Auxiliares del gradiente: phi = D^2 psi, chi = dE/dphi*

__constant__ int d_Nx, d_Ny, d_Nz;
__constant__ double d_dx, d_Lx, d_Bz, d_By;
__constant__ double d_k1[2], d_k2[2], d_k3[2], d_k4[2], d_k6[2], d_alfa[2];
__constant__ int d_kd;

inline int hidx3(int i,int j,int k){ return i + Nx*(j + Ny*k); }


// Convencion: D = -i grad - A (hermitico) => D^2 = -(laplaciano covariante), Dx = -i (d/dx - i Ax).
// Gauge A = (By z, Bz x, 0), periodico en xy con twist magnetico en x. Bordes LIBRES en z.
__device__ inline int didx(int i,int j,int k){ return i + d_Nx*(j + d_Ny*k); }
__device__ inline int layer(int k){ return (k < d_kd) ? 0 : 1; }
__device__ inline bool interior(int k){ return k > 0 && k < d_Nz - 1; }   // Donde D^2 tiene stencil completo

// Link covariante hacia el vecino (i+s, j, k), s = +-1, con twist de borde en x
__device__ inline cd link_x(int i,int j,int k,int s){
    double ph = -s * d_By * (k * d_dx) * d_dx;
    if (s ==  1 && i == d_Nx - 1) ph += d_Bz * d_Lx * (j * d_dx);
    if (s == -1 && i == 0)        ph -= d_Bz * d_Lx * (j * d_dx);
    return thrust::exp(cd(0.0, ph));
}
__device__ inline cd link_y(int i,int s){ return thrust::exp(cd(0.0, -s * d_Bz * (i * d_dx) * d_dx)); }

__device__ inline cd fx(const cd* f,int i,int j,int k,int s){ return f[didx((i + s + d_Nx) % d_Nx, j, k)] * link_x(i,j,k,s); }
__device__ inline cd fy(const cd* f,int i,int j,int k,int s){ return f[didx(i, (j + s + d_Ny) % d_Ny, k)] * link_y(i,s); }

// Dx f, centrada
__device__ cd Dx(const cd* f,int i,int j,int k){
    return cd(0.0, -1.0) * (fx(f,i,j,k,1) - fx(f,i,j,k,-1)) / (2.0 * d_dx);
}

// D^2 f, los vecinos en z fuera del dominio no aportan (D^2 es hermitica con esta definicion)
__device__ cd D2(const cd* f,int i,int j,int k){
    cd s = fx(f,i,j,k,1) + fx(f,i,j,k,-1) + fy(f,i,j,k,1) + fy(f,i,j,k,-1) - 6.0 * f[didx(i,j,k)];
    if (k > 0)        s += f[didx(i,j,k-1)];
    if (k < d_Nz - 1) s += f[didx(i,j,k+1)];
    return -s / (d_dx * d_dx);
}

// div(w grad f), con w escalon: el enlace (k,k+1) usa w de la capa de k. Solo enlaces dentro de [k0,k1] (borde libre)
__device__ cd div_grad(const cd* f,int i,int j,int k,const double* w,int k0,int k1){
    cd f0 = f[didx(i,j,k)];
    cd s = w[layer(k)] * (fx(f,i,j,k,1) + fx(f,i,j,k,-1) + fy(f,i,j,k,1) + fy(f,i,j,k,-1) - 4.0 * f0);
    if (k < k1) s += w[layer(k)]     * (f[didx(i,j,k+1)] - f0);
    if (k > k0) s += w[layer(k - 1)] * (f[didx(i,j,k-1)] - f0);
    return s / (d_dx * d_dx);
}


struct energy_density_functor {
    const cd* psi;
    energy_density_functor(const cd* _psi) : psi(_psi) {}

    __device__ double operator()(int id) const {
        int i = id % d_Nx;
        int j = (id / d_Nx) % d_Ny;
        int k = id / (d_Nx * d_Ny);
        int r = layer(k);
        cd psi0 = psi[id];

        //Cinetico k2 |D psi|^2, diferencias hacia adelante (enlace en z solo dentro del dominio)
        double kin = thrust::norm(fx(psi,i,j,k,1) - psi0) + thrust::norm(fy(psi,i,j,k,1) - psi0);
        if (k < d_Nz - 1) kin += thrust::norm(psi[didx(i,j,k+1)] - psi0);

        //Energia potencial
        double psi_sq = thrust::norm(psi0);
        double potential = 0.5 * psi_sq * psi_sq - d_alfa[r] * psi_sq;

        //Término lifshitz lineal
        cd dxpsi = Dx(psi,i,j,k);
        double lifshitz = d_k1[r] * d_By * (thrust::conj(psi0) * dxpsi).real();

        double e = d_k2[r] * kin / (d_dx * d_dx) + potential + lifshitz;

        //k6 |D D^2 psi|^2 - k4 |D^2 psi|^2 - k3 By Re((Dx psi)* D^2 psi), D^2 psi solo donde su stencil cabe en z
        if (interior(k)) {
            cd p0 = D2(psi,i,j,k);
            double kin6 = thrust::norm(D2(psi,(i + 1) % d_Nx,j,k) * link_x(i,j,k,1) - p0)
                        + thrust::norm(D2(psi,i,(j + 1) % d_Ny,k) * link_y(i,1) - p0);
            if (interior(k + 1)) kin6 += thrust::norm(D2(psi,i,j,k+1) - p0);

            e += d_k6[r] * kin6 / (d_dx * d_dx) - d_k4[r] * thrust::norm(p0)
               - d_k3[r] * d_By * (thrust::conj(dxpsi) * p0).real();
        }
        return e;
    }
};


double compute_energy(const cd* d_psi, int Ntot) {
    double total_energy = thrust::transform_reduce(
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(Ntot),
        energy_density_functor(d_psi),
        double(0.0),
        thrust::plus<double>()
    );
    return total_energy * dx * dx * dx;
}


cd dot(const cd* d_a, const cd* d_b, int Ntot)
{
    auto pa = thrust::device_pointer_cast(d_a);
    auto pb = thrust::device_pointer_cast(d_b);

    return thrust::inner_product(
        pa, pa + Ntot,
        pb,
        cd(0.0, 0.0),
        thrust::plus<cd>(),
        [] __device__ (cd a, cd b) {return thrust::conj(a) * b;} );
}


// Derivada funcional dE/dpsi* (sin el factor dx^3), adjunta exacta de la energia discreta:
// con phi = D^2 psi (solo interior en z),
//   chi  = dE/dphi* = -div(k6 grad phi) - k4 phi - (By/2) k3 Dx psi
//   grad = -div(k2 grad psi) - (alfa - |psi|^2) psi + By k1 Dx psi + D^2 chi - (By/2) k3 Dx phi
// En el bulk de cada capa esto es k6 D^6 psi - k4 D^4 psi - k3 By/2 {D^2, Dx} psi + ..., y en la
// interfaz/bordes los coeficientes de cada sitio entran en el stencil igual que k2 en el codigo k1_escalon.
__device__ inline bool site(int& i,int& j,int& k){
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;
    return i < d_Nx && j < d_Ny && k < d_Nz;
}

__global__ void phi_kernel(const cd* psi, cd* phi){
    int i, j, k;
    if (!site(i,j,k)) return;
    phi[didx(i,j,k)] = interior(k) ? D2(psi,i,j,k) : cd(0.0, 0.0);
}

__global__ void chi_kernel(const cd* psi, const cd* phi, cd* chi){
    int i, j, k;
    if (!site(i,j,k)) return;
    int id = didx(i,j,k), r = layer(k);
    chi[id] = interior(k) ? -div_grad(phi,i,j,k,d_k6,1,d_Nz-2) - d_k4[r] * phi[id] - 0.5 * d_By * d_k3[r] * Dx(psi,i,j,k)
                          : cd(0.0, 0.0);
}

__global__ void grad_kernel(const cd* psi, const cd* phi, const cd* chi, cd* grad){
    int i, j, k;
    if (!site(i,j,k)) return;
    int id = didx(i,j,k), r = layer(k);
    cd psi0 = psi[id];
    grad[id] = -div_grad(psi,i,j,k,d_k2,0,d_Nz-1) - (d_alfa[r] - thrust::norm(psi0)) * psi0
             + d_By * d_k1[r] * Dx(psi,i,j,k)
             + D2(chi,i,j,k) - 0.5 * d_By * d_k3[r] * Dx(phi,i,j,k);
}

void compute_gradient(const cd* psi, cd* grad, dim3 Blocks, dim3 Threads){
    phi_kernel<<<Blocks,Threads>>>(psi, d_phi);
    chi_kernel<<<Blocks,Threads>>>(psi, d_phi, d_chi);
    grad_kernel<<<Blocks,Threads>>>(psi, d_phi, d_chi, grad);
    cudaDeviceSynchronize();
}


//Escribe la densidad o la fase en un .dat
void write_field(const std::vector<cd>& psi, const std::string& fname, bool density)
{
    std::ofstream file(fname);
    file<<std::setprecision(14);

    for(int k=0;k<Nz;k++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                int id=hidx3(i,j,k);
                double x=i*dx;
                double y=j*dx;
                double z=k*dx;
                double val = density ? thrust::norm(psi[id]) : thrust::arg(psi[id]);
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
    cd* g_trial,
    dim3 Blocks, dim3 Threads,
    int Ntot)
{
    double c1 = 1e-4;
    double c2 = 0.1;
    double alpha = 1.0;
    double alpha_lo = 0.0, alpha_hi = 10.0;

    auto ppsi = thrust::device_pointer_cast(psi);
    auto ptrial = thrust::device_pointer_cast(trial);
    auto pdir = thrust::device_pointer_cast(dir);

    double E0 = compute_energy(psi, Ntot);
    double slope0 = dot(grad, dir, Ntot).real();

    for(int iter = 0; iter < 30; iter++) {

        thrust::transform(ppsi, ppsi + Ntot,
                  pdir, ptrial,
                  [alpha] __device__ (cd p, cd d){ return p + alpha * d; });

        double E = compute_energy(trial, Ntot);

        if(E > E0 + c1 * alpha * slope0) {
            alpha_hi = alpha;
            alpha = 0.5 * (alpha_lo + alpha_hi);
            continue;
        }

        compute_gradient(trial, g_trial, Blocks, Threads);
        double slope = dot(g_trial, dir, Ntot).real();

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
    __host__ __device__ cd operator()(const cd& g, const cd& g_old) const {
        return thrust::conj(g) * (g - g_old);
    }
};


int main(int argc, char* argv[]) {

    k1[0] = 1.0; k2[0] = 2.0; alfa[0] = 0.2;
    k1[1] = 0.0; k2[1] = 1.0; alfa[1] = 1.0;
    //Valores de ejemplo, ajustar. k6 > 0 si k4 > 0 para que la energia este acotada
    k3[0] = 0.1; k4[0] = 0.1; k6[0] = 0.1;
    k3[1] = 0.0; k4[1] = 0.0; k6[1] = 0.0;

    if (argc < 9 || std::stoi(argv[8]) < 0 || std::stoi(argv[8]) > std::stoi(argv[3])) {
        std::cerr << "Uso: " << argv[0]
                  << " Nx Ny Nz Nv dx By seed.dat kd, kd entre 0 y Nz" << std::endl;
        return 1;
    }

    Nx = std::stoi(argv[1]);
    Ny = std::stoi(argv[2]);
    Nz = std::stoi(argv[3]);
    Nv = std::stoi(argv[4]);
    dx = std::stod(argv[5]);
    By = std::stod(argv[6]);
    std::ifstream seed_file(argv[7]);
    kd = std::stoi(argv[8]);

    Lx = Nx * dx;
    Ly = Ny * dx;
    Bz = 2.0 * M_PI * Nv / (Lx * Ly);  
    int Ntot = Nx * Ny * Nz;

    //Parametro computacionales 
    dim3 Threads(8, 8, 8);
    dim3 Blocks((Nx + 7) / 8, (Ny + 7) / 8, (Nz + 7) / 8);

    int max_iter = 80000;
    double tol = 1e-10 * Ntot;
    int restart_period = 100;

    std::cout << "Nx=" << Nx << " Ny=" << Ny << " Nz=" << Nz << " Nv=" << Nv
              << " dx=" << dx << " Bz(fuera de plano, multiplo del numero de cuantos de flujo magnetico)=" << Bz
              << " By(en plano)=" << By
              << " d=" << kd * dx << " (kd=" << kd << ")"
              << " k1=(" << k1[0] << "," << k1[1] << ")"
              << " k2=(" << k2[0] << "," << k2[1] << ")"
              << " k3=(" << k3[0] << "," << k3[1] << ")"
              << " k4=(" << k4[0] << "," << k4[1] << ")"
              << " k6=(" << k6[0] << "," << k6[1] << ")"
              << " alfa=(" << alfa[0] << "," << alfa[1] << ")\n";

    cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(int));
    cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(int));
    cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(int));
    cudaMemcpyToSymbol(d_dx, &dx, sizeof(double));
    cudaMemcpyToSymbol(d_Lx, &Lx, sizeof(double));
    cudaMemcpyToSymbol(d_Bz, &Bz, sizeof(double));
    cudaMemcpyToSymbol(d_By, &By, sizeof(double));
    cudaMemcpyToSymbol(d_k1, k1, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_k2, k2, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_k3, k3, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_k4, k4, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_k6, k6, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_alfa, alfa, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_kd, &kd, sizeof(int));

    std::vector<cd> psi (Ntot);
    thrust::device_vector<cd> d_psi(Ntot);
    thrust::device_vector<cd> d_grad(Ntot);
    thrust::device_vector<cd> d_grad_old(Ntot);
    thrust::device_vector<cd> d_dir(Ntot);
    thrust::device_vector<cd> trial(Ntot);
    thrust::device_vector<cd> g_trial(Ntot);
    thrust::device_vector<cd> phi_buf(Ntot), chi_buf(Ntot);
    d_phi = thrust::raw_pointer_cast(phi_buf.data());
    d_chi = thrust::raw_pointer_cast(chi_buf.data());

    cd* psi_ptr  = thrust::raw_pointer_cast(d_psi.data());
    cd* grad_ptr = thrust::raw_pointer_cast(d_grad.data());
    cd* dir_ptr = thrust::raw_pointer_cast(d_dir.data());
    cd* grad_old_ptr = thrust::raw_pointer_cast(d_grad_old.data());
    cd* trial_ptr = thrust::raw_pointer_cast(trial.data());
    cd* g_trial_ptr = thrust::raw_pointer_cast(g_trial.data());

    //Chequea que abra bien el seed y que coincida con la cantidad de entradas esperadas (Ntot)
    int nread = 0;
    for (auto& p : psi) {
        double r, im;
        if (!(seed_file >> r >> im)) break;
        p = cd(r, im);
        nread++;
    }
    double extra;
    if (nread != Ntot || (seed_file >> extra)) {
        std::cerr << "Semilla invalida: se leyeron " << nread << " de " << Ntot
                  << " sitios (o sobran datos). Esperado: Nx*Ny*Nz lineas 'Re Im'.\n";
        return 1;
    }
    d_psi = psi; 

    //Chequeo gradiente vs energia: derivada direccional numerica de E en la direccion grad vs 2 dx^3 |grad|^2
    {
        compute_gradient(psi_ptr, grad_ptr, Blocks, Threads);
        double g2 = dot(grad_ptr, grad_ptr, Ntot).real();
        double h = 1e-5 * std::sqrt(dot(psi_ptr, psi_ptr, Ntot).real() / g2);
        double Epm[2];
        for (int s = 0; s < 2; s++) {
            double a = s ? -h : h;
            thrust::transform(d_psi.begin(), d_psi.end(), d_grad.begin(), trial.begin(),
                              [a] __device__ (cd p, cd g){ return p + a * g; });
            Epm[s] = compute_energy(trial_ptr, Ntot);
        }
        double num = (Epm[0] - Epm[1]) / (2.0 * h), ana = 2.0 * dx * dx * dx * g2;
        std::cout << "Chequeo gradiente: numerico=" << num << " analitico=" << ana
                  << " error relativo=" << std::abs(num - ana) / std::abs(ana) << "\n";
    }

    //Log de energía y gradiente cuadrado
    std::ofstream file("run_log.dat");
    file<<std::setprecision(14);


    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);
    cudaEventRecord(t_start);


    //Loop principal
    for(int k = 0; k < max_iter; k++){
        compute_gradient(psi_ptr, grad_ptr, Blocks, Threads);

        double norm2 = dot(grad_ptr, grad_ptr, Ntot).real();

        if(k % 10 == 0){
            double E = compute_energy(psi_ptr, Ntot);
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

            double den = dot(grad_old_ptr, grad_old_ptr, Ntot).real();

            beta = std::max(0.0, num / den);

            thrust::transform(d_grad.begin(), d_grad.end(),
                  d_dir.begin(), d_dir.begin(),
                  [beta] __device__ (cd g, cd d){ return -g + beta * d; });
        }

        double alpha = line_search_wolfe(psi_ptr, dir_ptr, grad_ptr, trial_ptr, g_trial_ptr, Blocks, Threads, Ntot);

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

    std::cout << "Tiempo: " << elapsed_ms / 1000.0f << " s ("
              << elapsed_ms << " ms)\n";

    thrust::copy(d_psi.begin(), d_psi.end(), psi.begin());

    write_field(psi, "density.dat", true);
    write_field(psi, "phase.dat", false);

    return 0;
}
