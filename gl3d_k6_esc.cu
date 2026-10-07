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
double k1[2], k2[2], alfa[2];
double k3[2], k4[2], k6[2];   
int kd;
dim3 Blocks, Threads;        
cd *phi_buf, *chi_buf;     

__constant__ int d_Nx, d_Ny, d_Nz;
__constant__ double d_dx, d_Lx, d_Bz, d_By;
__constant__ double d_k1[2], d_k2[2], d_alfa[2];
__constant__ double d_k3[2], d_k4[2], d_k6[2];  
__constant__ int d_kd;

inline int hidx3(int i,int j,int k){ return i + Nx*(j + Ny*k); }
__device__ inline int didx(int i,int j,int k){ return i + d_Nx*(j + d_Ny*k); }
__device__ inline int region(int k){ return (k < d_kd) ? 0 : 1; }

//Indices del thread; devuelve true si cae fuera de la red
__device__ inline bool site(int &i, int &j, int &k){
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;
    return i >= d_Nx || j >= d_Ny || k >= d_Nz;
}


struct Vecinos {
    cd xp, xm, yp, ym; 

    __device__ Vecinos(){}

    __device__ Vecinos(const cd* f, int i, int j, int k){
        double x = i * d_dx;
        double y = j * d_dx;
        double z = k * d_dx;

        //Link en x
        int ip = (i + 1) % d_Nx;
        int im = (i - 1 + d_Nx) % d_Nx;
        cd Ux_pos = thrust::exp(cd(0.0, -d_By * z * d_dx));
        cd Ux_neg = thrust::conj(Ux_pos);
        xp = f[didx(ip,j,k)] * Ux_pos;
        xm = f[didx(im,j,k)] * Ux_neg;

        // Twist de borde en x
        if (i == d_Nx - 1) xp *= thrust::exp(cd(0.0,  d_Bz * d_Lx * y));
        if (i == 0)        xm *= thrust::exp(cd(0.0, -d_Bz * d_Lx * y));

        //Link en y
        int jp = (j + 1) % d_Ny;
        int jm = (j - 1 + d_Ny) % d_Ny;
        cd Uy_pos = thrust::exp(cd(0.0, - d_Bz * x * d_dx));
        cd Uy_neg = thrust::conj(Uy_pos);
        yp = Uy_pos * f[didx(i,jp,k)];
        ym = Uy_neg * f[didx(i,jm,k)];
    }
};


//Calcula Dx psi
__device__ cd Dx(const cd* f, int i, int j, int k) {
    Vecinos n(f, i, j, k);
    return cd(0.0, -1.0) * (n.xp - n.xm) / (2.0 * d_dx);
}


// D^2 f. En z: centrada en el interior; en los bordes, descentrada (1,-2,1) desde el borde.
// adj = true aplica el adjunto (D^2)^+, que pide el gradiente: en x,y es el mismo
// operador (hermitico); en z es la traspuesta de la matriz de d_z^2, que difiere solo en
// los planos 0..2 y Nz-3..Nz-1 (la columna k recibe de las filas interiores k-1, k, k+1
// y de las filas de borde 0 y Nz-1).
__device__ cd D2(const cd* f, int i, int j, int k, bool adj) {
    const double c[3] = {1.0, -2.0, 1.0};   // stencil de borde: f_b - 2 f_b+-1 + f_b+-2
    int N = d_Nz - 1;
    cd f0 = f[didx(i,j,k)];
    Vecinos n(f, i, j, k);
    cd lap = n.xp + n.xm + n.yp + n.ym - 4.0 * f0;

    if (!adj) {
        if (k == 0)      lap += f[didx(i,j,k+2)] - 2.0 * f[didx(i,j,k+1)] + f0;
        else if (k == N) lap += f[didx(i,j,k-2)] - 2.0 * f[didx(i,j,k-1)] + f0;
        else             lap += f[didx(i,j,k+1)] - 2.0 * f0 + f[didx(i,j,k-1)];
    } else {
        if (k >= 2)          lap += f[didx(i,j,k-1)];              // fila k-1
        if (k >= 1 && k < N) lap -= 2.0 * f0;                      // fila k
        if (k <= N - 2)      lap += f[didx(i,j,k+1)];              // fila k+1
        if (k <= 2)          lap += c[k] * f[didx(i,j,0)];         // fila 0
        if (N - k <= 2)      lap += c[N - k] * f[didx(i,j,N)];     // fila Nz-1
    }
    return -lap / (d_dx * d_dx);
}


// Kc f = d/df^* de sum c |D f|^2, con diferencias adelantadas y c escalon en z.
// Es el -k2 lap_xy - div_z del original con c en lugar de k2: el enlace z (k, k+1)
// lleva el coeficiente de la region de k. En los bordes z falta el enlace hacia
// afuera y no se impone nada.
__device__ cd Kc(const cd* f, const double* c, int i, int j, int k) {
    int r = region(k);
    cd f0 = f[didx(i,j,k)];

    Vecinos n(f, i, j, k);
    cd lap_xy = (n.xp + n.xm + n.yp + n.ym - 4.0 * f0) / (d_dx * d_dx);

    cd div_z = 0.0;
    if (k < d_Nz - 1) div_z += c[r] * (f[didx(i,j,k+1)] - f0);
    if (k > 0)        div_z -= c[region(k - 1)] * (f0 - f[didx(i,j,k-1)]);

    return -c[r] * lap_xy - div_z / (d_dx * d_dx);
}


// phi = D^2 psi en phi_buf
void compute_phi(const cd* psi){
    thrust::transform(thrust::device,
        thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(Nx * Ny * Nz),
        thrust::device_pointer_cast(phi_buf),
        [psi] __device__ (int id){
            int i = id % d_Nx, j = (id / d_Nx) % d_Ny, k = id / (d_Nx * d_Ny);
            return D2(psi, i, j, k, false);
        });
}

// chi = k6 D^4 psi - k4 D^2 psi - k3 By/2 Dx psi en chi_buf, con los coeficientes adentro
// (k6 por enlace via Kc, k4 y k3 por sitio) para que D2(chi) tome el de cada lado en la interfaz
void compute_chi(const cd* psi){
    const cd* phi = phi_buf;   // copia local: la lambda de device no puede leer el global de host
    thrust::transform(thrust::device,
        thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(Nx * Ny * Nz),
        thrust::device_pointer_cast(chi_buf),
        [psi, phi] __device__ (int id){
            int i = id % d_Nx, j = (id / d_Nx) % d_Ny, k = id / (d_Nx * d_Ny);
            int r = region(k);
            return Kc(phi, d_k6, i, j, k) - d_k4[r] * phi[id] - 0.5 * d_k3[r] * d_By * Dx(psi, i, j, k);
        });
}

// grad = k2 D^2 psi + (|psi|^2 - alfa) psi + k1 By Dx psi + k6 D^6 psi - k4 D^4 psi - k3 By/2 {D^2, Dx} psi
// con los terminos de orden superior escritos como (D^2)^+ chi - k3 By/2 Dx D^2 psi
__global__ void kernel_grad(const cd* psi, const cd* phi, const cd* chi, cd* grad){
    int i,j,k;
    if(!site(i,j,k)){
        int r = region(k);
        cd psi0 = psi[didx(i,j,k)];
    
        cd grad_val = Kc(psi, d_k2, i, j, k) - (d_alfa[r] - thrust::norm(psi0)) * psi0;
    
        //Aporte termino lineal Lifshitz
        grad_val += d_k1[r] * d_By * Dx(psi, i, j, k);
    
        //Terminos orden superior
        grad_val += D2(chi, i, j, k, true) - 0.5 * d_k3[r] * d_By * Dx(phi, i, j, k);
    
        grad[didx(i,j,k)] = grad_val;
    }
}

void compute_gradient(const cd* psi, cd* grad){
    compute_phi(psi);
    compute_chi(psi);
    kernel_grad<<<Blocks,Threads>>>(psi, phi_buf, chi_buf, grad);
    cudaDeviceSynchronize();
}

struct energy_density_functor {
    const cd* psi;
    const cd* phi;
    energy_density_functor(const cd* _psi, const cd* _phi) : psi(_psi), phi(_phi){}
    
    __device__ double operator()(int id) const {
        int i = id % d_Nx;
        int j = (id / d_Nx) % d_Ny;
        int k = id / (d_Nx * d_Ny);
        int r = region(k);

        cd psi0 = psi[id];
        cd phi0 = phi[id];
        Vecinos n(psi, i, j, k);
        Vecinos m(phi, i, j, k);

        //|Dpsi|^2 y |D D^2psi|^2. En z solo el enlace (k, k+1) si existe el plano k+1
        double kin_psi = thrust::norm(n.xp - psi0) + thrust::norm(n.yp - psi0);
        double kin_phi = thrust::norm(m.xp - phi0) + thrust::norm(m.yp - phi0);
        if (k < d_Nz - 1) {
            kin_psi += thrust::norm(psi[didx(i,j,k+1)] - psi0);
            kin_phi += thrust::norm(phi[didx(i,j,k+1)] - phi0);
        } 

        //Energia potencial
        double psi_sq = thrust::norm(psi0);
        double potential = 0.5 * psi_sq * psi_sq - d_alfa[r] * psi_sq;

        //Término lifshitz lineal
        cd Dxpsi = Dx(psi, i, j, k);
        double lifshitz = d_k1[r] * d_By * (thrust::conj(psi0) * Dxpsi).real();

        //k6 |D D^2 psi|^2 - k4 |D^2 psi|^2 - k3 By Re((Dx psi)* D^2 psi)
        double orden_sup = d_k6[r] * kin_phi / (d_dx * d_dx)
                         - d_k4[r] * thrust::norm(phi0)
                         - d_k3[r] * d_By * (thrust::conj(Dxpsi) * phi0).real();

        return d_k2[r] * kin_psi / (d_dx * d_dx) + potential + lifshitz + orden_sup;
    }
};


double compute_energy(const cd* d_psi, int Ntot) {
    compute_phi(d_psi);

    double total_energy = 0.0;
    total_energy += thrust::transform_reduce(
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(Ntot),
        energy_density_functor(d_psi, phi_buf),
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
    double slope0 = 2*dx*dx*dx*dot(grad, dir, Ntot).real();

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

        compute_gradient(trial, g_trial);
        double slope = 2*dx*dx*dx*dot(g_trial, dir, Ntot).real();

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

    //Quedan chequear bien los parametros de k1 y los de orden superior
    k1[0] = 1.0; k2[0] = 24.8; alfa[0] = 0.4;
    k3[0] = 0.3; k4[0] = 0.1; k6[0] = 0.05;
    k1[1] = 0.0; k2[1] = 1.0; alfa[1] = 1.0;
    k3[1] = 0.0; k4[1] = 0.0; k6[1] = 0.0;

    if (argc < 9 || std::stoi(argv[3]) < 4 || std::stoi(argv[8]) < 0 || std::stoi(argv[8]) >= std::stoi(argv[3])) {
        std::cerr << "Uso: " << argv[0]
                  << " Nx Ny Nz Nv dx By seed.dat kd [sup_order=1], Nz >= 4, kd entre 0 y Nz. Nz >= 4." << std::endl;
        return 1;
    }

    if(argc == 10 && argv[9][0] == '0'){
        k1[0] = k3[0] = k4[0] = k6[0] = 0.0;
        std::cout << "Sin terminos orden superior" << "\n";
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
    Threads = dim3(8, 8, 8);
    Blocks = dim3((Nx + 7) / 8, (Ny + 7) / 8, (Nz + 7) / 8);
    int max_iter = 80000;
    double tol = 1e-10 * Ntot;
    int restart_period = 100;

    std::cout << "Nx=" << Nx << " Ny=" << Ny << " Nz=" << Nz << " Nv=" << Nv
              << " dx=" << dx << " Bz(fuera de plano, multiplo del numero de cuantos de flujo magnetico)=" << Bz
              << " By(en plano)=" << By
              << " d=" << kd * dx << " (kd=" << kd << ")"
              << " k1=(" << k1[0] << "," << k1[1] << ")"
              << " k2=(" << k2[0] << "," << k2[1] << ")"
              << " alfa=(" << alfa[0] << "," << alfa[1] << ")"
              << " k3=(" << k3[0] << "," << k3[1] << ")"
              << " k4=(" << k4[0] << "," << k4[1] << ")"
              << " k6=(" << k6[0] << "," << k6[1] << ")\n";

    cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(int));
    cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(int));
    cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(int));
    cudaMemcpyToSymbol(d_dx, &dx, sizeof(double));
    cudaMemcpyToSymbol(d_Lx, &Lx, sizeof(double));
    cudaMemcpyToSymbol(d_Bz, &Bz, sizeof(double));
    cudaMemcpyToSymbol(d_By, &By, sizeof(double));
    cudaMemcpyToSymbol(d_k1, k1, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_k2, k2, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_alfa, alfa, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_k3, k3, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_k4, k4, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_k6, k6, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_kd, &kd, sizeof(int));

    std::vector<cd> psi (Ntot);
    thrust::device_vector<cd> d_psi(Ntot);
    thrust::device_vector<cd> d_grad(Ntot);
    thrust::device_vector<cd> d_grad_old(Ntot);
    thrust::device_vector<cd> d_dir(Ntot);
    thrust::device_vector<cd> trial(Ntot);
    thrust::device_vector<cd> g_trial(Ntot);
    thrust::device_vector<cd> d_phi(Ntot);   
    thrust::device_vector<cd> d_chi(Ntot);   

    cd* psi_ptr  = thrust::raw_pointer_cast(d_psi.data());
    cd* grad_ptr = thrust::raw_pointer_cast(d_grad.data());
    cd* dir_ptr = thrust::raw_pointer_cast(d_dir.data());
    cd* grad_old_ptr = thrust::raw_pointer_cast(d_grad_old.data());
    cd* trial_ptr = thrust::raw_pointer_cast(trial.data());
    cd* g_trial_ptr = thrust::raw_pointer_cast(g_trial.data());
    phi_buf = thrust::raw_pointer_cast(d_phi.data());
    chi_buf = thrust::raw_pointer_cast(d_chi.data());

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

    //Log de energía y gradiente cuadrado
    std::ofstream file("run_log.dat");
    file<<std::setprecision(14);

    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);
    cudaEventRecord(t_start);

    //Loop principal
    for(int k = 0; k < max_iter; k++){
        compute_gradient(psi_ptr, grad_ptr);

        double norm2 = dot(grad_ptr, grad_ptr, Ntot).real();

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

        double alpha = line_search_wolfe(psi_ptr, dir_ptr, grad_ptr, trial_ptr, g_trial_ptr, Ntot);

        if(k % 10 == 0){
            double E = compute_energy(psi_ptr, Ntot);
            std::cout << "Iter " << k << " |grad|^2 = " << norm2 << " Energy: " << E << "\n";
            file << k << " " << norm2 << " " << E;
            file << " " << alpha << " " << beta << "\n";
        }

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
