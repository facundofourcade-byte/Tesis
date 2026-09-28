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
#include <thrust/transform.h>

using cd = thrust::complex<double>;

int Nx, Ny, Nz, Nv;
double dx, Lx, Ly, Bz, By;
// CAMBIO: parametros escalon en z, [0] -> region 1 (0 <= z < d), [1] -> region 2 (d <= z <= Lz)
double k1[2], k2[2], alfa[2];
// CAMBIO: kd = primer indice k con z = k*dx >= d (los nodos k < kd son region 1)
int kd;

__constant__ int d_Nx, d_Ny, d_Nz;
__constant__ double d_dx, d_Lx, d_Bz, d_By;
// CAMBIO: copias en device de los parametros escalon y del indice de la interfase
__constant__ double d_k1[2], d_k2[2], d_alfa[2];
__constant__ int d_kd;

inline int hidx3(int i,int j,int k){ return i + Nx*(j + Ny*k); }


// CAMBIO: el functor recibe la region r (0 o 1) y usa los parametros de esa region
struct energy_density_functor {
    const cd* psi;
    int r;
    energy_density_functor(const cd* _psi, int _r) : psi(_psi), r(_r) {}

    __device__ double operator()(int id) const {
        int i = id % d_Nx;
        int j = (id / d_Nx) % d_Ny;
        int k = id / (d_Nx * d_Ny);

        cd psi0 = psi[id];
        double x = i * d_dx;
        double y = j * d_dx;
        double z = k * d_dx;

        //Link derivada discreta en coordenada x
        int ip = (i + 1) % d_Nx;
        int im = (i - 1 + d_Nx) % d_Nx;

        cd Ux_pos = thrust::exp(cd(0.0, -d_By * z * d_dx));
        cd Ux_neg = thrust::conj(Ux_pos);

        cd psi_xp = psi[ip + d_Nx * (j + d_Ny * k)] * Ux_pos;
        cd psi_xm = psi[im + d_Nx * (j + d_Ny * k)] * Ux_neg;

        // Twist de borde en x
        if (i == d_Nx - 1) psi_xp *= thrust::exp(cd(0.0,  d_Bz * d_Lx * y));
        if (i == 0)        psi_xm *= thrust::exp(cd(0.0, -d_Bz * d_Lx * y));

        cd Dx = (psi_xp - psi0) / d_dx;   

        //Link derivada discreta en y
        int jp = (j + 1) % d_Ny;
        cd Uy = thrust::exp(cd(0.0, - d_Bz * x * d_dx));
        cd psi_yp = Uy * psi[i + d_Nx * (jp + d_Ny * k)];
        cd Dy = (psi_yp - psi0) / d_dx;

        //Derivada neumann en z
        double kinetic_z = 0.0;
        if (k < d_Nz - 1) {
            cd psi_zp = psi[i + d_Nx * (j + d_Ny * (k + 1))];
            cd Dz = (psi_zp - psi0) / d_dx;
            kinetic_z = thrust::norm(Dz);
        }

        //Energia potencial
        double psi_sq = thrust::norm(psi0);
        double potential = 0.5 * psi_sq * psi_sq - d_alfa[r] * psi_sq;   // CAMBIO: alfa(z)

        //Término lifshitz lineal
        cd Dx_centered = (psi_xp - psi_xm) / (2.0 * d_dx);   
        double lifshitz = d_k1[r] * d_By * (thrust::conj(psi0) * Dx_centered).imag();   // CAMBIO: k1(z)

        // CAMBIO: k2(z) multiplica todo |Dpsi|^2. El link en z (k -> k+1) lleva el k2 del nodo k,
        // asi el link que cruza z = d usa k2_1 (region 1) y el siguiente k2_2.
        return d_k2[r] * (thrust::norm(Dx) + thrust::norm(Dy) + kinetic_z) + potential + lifshitz;
    }
};


// CAMBIO: la energia se calcula por separado en los dos sectores:
// region 1 = ids [0, Nx*Ny*kd), region 2 = ids [Nx*Ny*kd, Ntot)
double compute_energy(const cd* d_psi, int Ntot) {
    int lim[3] = {0, Nx * Ny * kd, Ntot};
    double total_energy = 0.0;
    for (int r = 0; r < 2; r++)
        total_energy += thrust::transform_reduce(
            thrust::device,
            thrust::counting_iterator<int>(lim[r]),
            thrust::counting_iterator<int>(lim[r + 1]),
            energy_density_functor(d_psi, r),
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


__global__ void compute_gradient(const cd* psi, cd* grad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= d_Nx || j >= d_Ny || k >= d_Nz) return;

    int id = i + d_Nx * (j + d_Ny * k);
    int r = (k < d_kd) ? 0 : 1;   // CAMBIO: region del nodo
    cd psi0 = psi[id];
    double x = i * d_dx;
    double y = j * d_dx;
    double z = k * d_dx;

    //Laplaciano covariante en x con twist de borde
    int ip = (i + 1) % d_Nx;
    int im = (i - 1 + d_Nx) % d_Nx;

    cd Ux_pos = thrust::exp(cd(0.0, -d_By * z * d_dx));
    cd Ux_neg = thrust::conj(Ux_pos);

    cd psi_xp = psi[ip + d_Nx * (j + d_Ny * k)] * Ux_pos;
    cd psi_xm = psi[im + d_Nx * (j + d_Ny * k)] * Ux_neg;

    if (i == d_Nx - 1) psi_xp *= thrust::exp(cd(0.0,  d_Bz * d_Lx * y));
    if (i == 0)        psi_xm *= thrust::exp(cd(0.0, -d_Bz * d_Lx * y));

    //Laplaciano covariante en y
    int jp = (j + 1) % d_Ny;
    int jm = (j - 1 + d_Ny) % d_Ny;
    cd Uy_pos = thrust::exp(cd(0.0, - d_Bz * x * d_dx));
    cd Uy_neg = thrust::conj(Uy_pos);

    cd psi_yp = Uy_pos * psi[i + d_Nx * (jp + d_Ny * k)];
    cd psi_ym = Uy_neg * psi[i + d_Nx * (jm + d_Ny * k)];

    //Laplaciano en z con condicion Neumann en z = 0, Lz. 
    int kp = (k < d_Nz - 1) ? k + 1 : k;
    int km = (k > 0)        ? k - 1 : k;
    cd psi_zp = psi[i + d_Nx * (j + d_Ny * kp)];
    cd psi_zm = psi[i + d_Nx * (j + d_Ny * km)];

    // CAMBIO: laplaciano en plano (k2 constante dentro del plano k) separado del de z
    cd lap_xy = (psi_xp + psi_xm + psi_yp + psi_ym - 4.0 * psi0) / (d_dx * d_dx);

    // CAMBIO: termino en z en forma de flujo, d/dz (k2 dpsi/dz). El link de arriba (k -> k+1) lleva
    // k2 del nodo k y el de abajo (k-1 -> k) k2 del nodo k-1. Esto impone en forma discreta el empalme
    // k2_- dz psi(d-) = k2_+ dz psi(d+) y sigue siendo Neumann en z = 0, Lz (kp = k o km = k anulan el flujo).
    int rm = (km < d_kd) ? 0 : 1;
    cd div_z = (d_k2[r] * (psi_zp - psi0) - d_k2[rm] * (psi0 - psi_zm)) / (d_dx * d_dx);

    cd grad_val = -d_k2[r] * lap_xy - div_z - (d_alfa[r] - thrust::norm(psi0)) * psi0;   // CAMBIO: k2(z), alfa(z)

    //Aporte termino lineal Lifshitz
    cd Dx_centered = (psi_xp - psi_xm) / (2.0 * d_dx);
    grad_val += cd(0.0, -d_By * d_k1[r]) * Dx_centered;   // CAMBIO: k1(z)

    grad[id] = grad_val;
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

        compute_gradient<<<Blocks,Threads>>>(trial, g_trial);
        cudaDeviceSynchronize();
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


    // CAMBIO: nuevos argumentos d y los parametros de cada region
    if (argc < 15) {
        std::cerr << "Uso: " << argv[0]
                  << " Nx Ny Nz Nv dx By seed.dat d k1_1 k1_2 k2_1 k2_2 alfa_1 alfa_2" << std::endl;
        return 1;
    }

    Nx = std::stoi(argv[1]);
    Ny = std::stoi(argv[2]);
    Nz = std::stoi(argv[3]);
    Nv = std::stoi(argv[4]);
    dx = std::stod(argv[5]);
    By = std::stod(argv[6]);
    std::ifstream seed_file(argv[7]);
    // CAMBIO: d (posicion del escalon) y parametros de cada region
    double d = std::stod(argv[8]);
    k1[0]   = std::stod(argv[9]);   k1[1]   = std::stod(argv[10]);
    k2[0]   = std::stod(argv[11]);  k2[1]   = std::stod(argv[12]);
    alfa[0] = std::stod(argv[13]);  alfa[1] = std::stod(argv[14]);

    Lx = Nx * dx;
    Ly = Ny * dx;
    Bz = 2.0 * M_PI * Nv / (Lx * Ly);  
    int Ntot = Nx * Ny * Nz;
    // CAMBIO: indice de la interfase. El -1e-9 evita que el redondeo mande un nodo con z == d a la region 1.
    kd = std::min(std::max((int)std::ceil(d / dx - 1e-9), 0), Nz);

    //Parametro computacionales 
    dim3 Threads(8, 8, 8);
    dim3 Blocks((Nx + 7) / 8, (Ny + 7) / 8, (Nz + 7) / 8);

    int max_iter = 80000;
    double tol = 1e-10 * Ntot;
    int restart_period = 100;

    std::cout << "Nx=" << Nx << " Ny=" << Ny << " Nz=" << Nz << " Nv=" << Nv
              << " dx=" << dx << " Bz(fuera de plano, multiplo del numero de cuantos de flujo magnetico)=" << Bz
              << " By(en plano)=" << By
              << " d=" << d << " (kd=" << kd << ")"
              << " k1=(" << k1[0] << "," << k1[1] << ")"
              << " k2=(" << k2[0] << "," << k2[1] << ")"
              << " alfa=(" << alfa[0] << "," << alfa[1] << ")\n";   // CAMBIO

    cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(int));
    cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(int));
    cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(int));
    cudaMemcpyToSymbol(d_dx, &dx, sizeof(double));
    cudaMemcpyToSymbol(d_Lx, &Lx, sizeof(double));
    cudaMemcpyToSymbol(d_Bz, &Bz, sizeof(double));
    cudaMemcpyToSymbol(d_By, &By, sizeof(double));
    // CAMBIO: copia de los parametros escalon e indice de interfase
    cudaMemcpyToSymbol(d_k1, k1, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_k2, k2, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_alfa, alfa, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_kd, &kd, sizeof(int));

    std::vector<cd> psi (Ntot);
    thrust::device_vector<cd> d_psi(Ntot);
    thrust::device_vector<cd> d_grad(Ntot);
    thrust::device_vector<cd> d_grad_old(Ntot);
    thrust::device_vector<cd> d_dir(Ntot);
    thrust::device_vector<cd> trial(Ntot);
    thrust::device_vector<cd> g_trial(Ntot);

    cd* psi_ptr  = thrust::raw_pointer_cast(d_psi.data());
    cd* grad_ptr = thrust::raw_pointer_cast(d_grad.data());
    cd* dir_ptr = thrust::raw_pointer_cast(d_dir.data());
    cd* grad_old_ptr = thrust::raw_pointer_cast(d_grad_old.data());
    cd* trial_ptr = thrust::raw_pointer_cast(trial.data());
    cd* g_trial_ptr = thrust::raw_pointer_cast(g_trial.data());

    for(auto& p : psi){
      double r, im;
      if (!(seed_file >> r >> im)) break;
      p = cd(r, im);
    }
    d_psi = psi;

    //Log de energía y gradiente cuadrado
    std::ofstream file("run_log.dat");
    file<<std::setprecision(14);


    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);
    cudaEventRecord(t_start);


    //Loop principal
    for(int k = 0; k < max_iter; k++){
        compute_gradient<<<Blocks,Threads>>>(psi_ptr, grad_ptr);
        cudaDeviceSynchronize();

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
