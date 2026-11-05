// Bicapa 2D: minimiza por gradiente conjugado (Polak-Ribiere + Wolfe) la energia libre
//
// F = int d^2r [ K1 |D psi_1|^2 + (-alfa_1 + Vp sum_i exp(-|r-R_i|^2/2w^2)) |psi_1|^2 + 1/2 |psi_1|^4
//              + K2 |(D - A_d) psi_2|^2 + alfa_2 |psi_2|^2 + 1/2 beta_2 |psi_2|^4
//              - J (psi_1^* psi_2 + c.c.) ]
//
// D = -i grad - A_perp, A_perp = (0, Bz x), A_d = (-By d, 0).
// Bordes periodicos con twist: psi(r + L_i e_i) = psi(r) exp(i q_i L_i) exp(i Bz Lx y delta_ix).
// Pinning en el Al: R_i, i = 1..Nv, red triangular (debilita el orden en los R_i, que atraen a los vortices).
//
// Misma estructura que actualizada/minimizador.cu, con el eje z reemplazado por el indice de capa
// k = 0 (psi_1), k = 1 (psi_2): psi[i + Nx*(j + Ny*k)].

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/transform.h>

using cd = thrust::complex<double>;

int Nx, Ny, Nv;
double dx, Lx, Ly, Bz, By, qx, qy, d, J;
double K[2], alfa[2], beta[2], Vp, w, X0, Y0;
dim3 Blocks, Threads;

__constant__ int d_Nx, d_Ny;
__constant__ double d_dx, d_Lx, d_Ly, d_Bz, d_By, d_qx, d_qy, d_d, d_J;
__constant__ double d_K[2], d_a2[2], d_a4[2];   //potencial de la capa k: a2_k |psi|^2 + 1/2 a4_k |psi|^4
__device__ const double* d_pin;                 //Vp sum_i exp(-|r-R_i|^2/2w^2) en (i + Nx j): se suma a a2 de la capa 1

//a2 en el sitio (i, j) de la capa k, con el pinning en la capa 1
__device__ inline double a2(int i, int j, int k){ return d_a2[k] + (k == 0 ? d_pin[i + d_Nx * j] : 0.0); }

inline int hidx(int i,int j,int k){ return i + Nx*(j + Ny*k); }
__device__ inline int didx(int i,int j,int k){ return i + d_Nx*(j + d_Ny*k); }

__device__ inline bool site(int &i, int &j, int &k){
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;
    return i >= d_Nx || j >= d_Ny || k >= 2;
}


struct Vecinos {
    cd xp, xm, yp, ym;

    __device__ Vecinos(const cd* f, int i, int j, int k){
        double x = i * d_dx;
        double y = j * d_dx;

        //Link Ux = exp(-i A_x dx): A_x = 0 en la capa 1 y A_d,x = -By d en la capa 2
        int ip = (i + 1) % d_Nx;
        int im = (i - 1 + d_Nx) % d_Nx;
        cd Ux_pos = thrust::exp(cd(0.0, k * d_By * d_d * d_dx));
        cd Ux_neg = thrust::conj(Ux_pos);
        xp = f[didx(ip,j,k)] * Ux_pos;
        xm = f[didx(im,j,k)] * Ux_neg;

        // Twist de borde en x: psi(x+Lx, y) = psi(x, y) exp(i (Bz Lx y + qx Lx))
        if (i == d_Nx - 1) xp *= thrust::exp(cd(0.0,  d_Bz * d_Lx * y + d_qx * d_Lx));
        if (i == 0)        xm *= thrust::exp(cd(0.0, -d_Bz * d_Lx * y - d_qx * d_Lx));

        //Link Uy = exp(-i A_y dx), A_y = Bz x
        int jp = (j + 1) % d_Ny;
        int jm = (j - 1 + d_Ny) % d_Ny;
        cd Uy_pos = thrust::exp(cd(0.0, - d_Bz * x * d_dx));
        cd Uy_neg = thrust::conj(Uy_pos);
        yp = Uy_pos * f[didx(i,jp,k)];
        ym = Uy_neg * f[didx(i,jm,k)];

        // Twist de borde en y: psi(x, y+Ly) = psi(x, y) exp(i qy Ly)
        if (j == d_Ny - 1) yp *= thrust::exp(cd(0.0,  d_qy * d_Ly));
        if (j == 0)        ym *= thrust::exp(cd(0.0, -d_qy * d_Ly));
    }
};


//H f = (Dx^2 + Dy^2) f con los enlaces de la capa k: d/df^* de sum |D f|^2 (enlaces adelantados)
__device__ cd H(const cd* f, int i, int j, int k) {
    Vecinos n(f, i, j, k);
    return -(n.xp + n.xm + n.yp + n.ym - 4.0 * f[didx(i,j,k)]) / (d_dx * d_dx);
}


// grad_k = dF/dpsi_k^* = K_k H psi_k + (a2_k + a4_k |psi_k|^2) psi_k - J psi_(otra capa), a2 = (-alfa_1 + pin, alfa_2), a4 = (1, beta_2)
__global__ void kernel_grad(const cd* psi, cd* grad){
    int i,j,k;
    if(!site(i,j,k)){
        cd psi0 = psi[didx(i,j,k)];
        grad[didx(i,j,k)] = d_K[k] * H(psi, i, j, k)
                          + (a2(i, j, k) + d_a4[k] * thrust::norm(psi0)) * psi0
                          - d_J * psi[didx(i,j,1-k)];
    }
}

void compute_gradient(const cd* psi, cd* grad){
    kernel_grad<<<Blocks,Threads>>>(psi, grad);
    cudaDeviceSynchronize();
}

struct energy_density_functor {
    const cd* psi;
    energy_density_functor(const cd* _psi) : psi(_psi){}

    __device__ double operator()(int id) const {
        int i = id % d_Nx;
        int j = (id / d_Nx) % d_Ny;
        int k = id / (d_Nx * d_Ny);

        cd psi0 = psi[id];
        Vecinos n(psi, i, j, k);

        //|D psi|^2 sumado sobre enlaces adelantados
        double kin = thrust::norm(n.xp - psi0) + thrust::norm(n.yp - psi0);

        double psi_sq = thrust::norm(psi0);
        double potential = a2(i, j, k) * psi_sq + 0.5 * d_a4[k] * psi_sq * psi_sq;

        //-J (psi_1^* psi_2 + c.c.) = -2 J Re(psi_1^* psi_2): cada capa aporta la mitad
        double josephson = -d_J * (thrust::conj(psi0) * psi[didx(i,j,1-k)]).real();

        return d_K[k] * kin / (d_dx * d_dx) + potential + josephson;
    }
};


double compute_energy(const cd* d_psi, int Ntot) {
    return thrust::transform_reduce(
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(Ntot),
        energy_density_functor(d_psi),
        double(0.0),
        thrust::plus<double>()
    ) * dx * dx;
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


//Escribe la densidad o la fase en un .dat: columnas x y capa val (capa 0 = psi_1, 1 = psi_2)
void write_field(const std::vector<cd>& psi, const std::string& fname, bool density)
{
    std::ofstream file(fname);
    file<<std::setprecision(14);

    for(int k=0;k<2;k++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                int id=hidx(i,j,k);
                double val = density ? thrust::norm(psi[id]) : thrust::arg(psi[id]);
                file<<i*dx<<" "<<j*dx<<" "<<k<<" "<<val<<"\n";
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
    // Arranca de 2x el paso aceptado en la llamada anterior (la primera vez, 1.0)
    static double alpha_prev = 0.5;
    double alpha = std::min(2.0 * alpha_prev, 10.0);
    double alpha_lo = 0.0, alpha_hi = 10.0;
    bool acotado = false;    //true cuando alpha_hi ya es un paso evaluado (intervalo cerrado)

    auto ppsi = thrust::device_pointer_cast(psi);
    auto ptrial = thrust::device_pointer_cast(trial);
    auto pdir = thrust::device_pointer_cast(dir);

    double alpha_ok = 0.0;   //Ultimo alpha que cumplio Armijo (0 = ninguno)

    double E0 = compute_energy(psi, Ntot);
    double slope0 = 2*dx*dx*dot(grad, dir, Ntot).real();

    // Si dir no es de descenso no hay paso valido: devuelve 0 y el loop principal reinicia con -grad.
    if(!(slope0 < 0.0)) return 0.0;

    for(int iter = 0; iter < 30; iter++) {

        thrust::transform(ppsi, ppsi + Ntot,
                  pdir, ptrial,
                  [alpha] __device__ (cd p, cd d){ return p + alpha * d; });

        double E = compute_energy(trial, Ntot);

        // NaN/inf cuentan como rechazo
        if(!std::isfinite(E) || !(E <= E0 + c1 * alpha * slope0)) {
            alpha_hi = alpha; acotado = true;
            alpha = 0.5 * (alpha_lo + alpha_hi);
            continue;
        }

        alpha_ok = alpha;

        compute_gradient(trial, g_trial);
        double slope = 2*dx*dx*dot(g_trial, dir, Ntot).real();

        if(std::abs(slope) <= c2 * std::abs(slope0))
            return alpha_prev = alpha;

        if(slope >= 0) {
            alpha_hi = alpha; acotado = true;
        } else
            alpha_lo = alpha;

        // Sin cota superior evaluada se agranda duplicando
        alpha = acotado ? 0.5 * (alpha_lo + alpha_hi) : std::min(2.0 * alpha, alpha_hi);
    }
    // El ultimo que cumplio Armijo, o 0 si ninguno.
    if(alpha_ok > 0.0) alpha_prev = alpha_ok;
    return alpha_ok;
}


struct polak_ribiere_op {
    __host__ __device__ cd operator()(const cd& g, const cd& g_old) const {
        return thrust::conj(g) * (g - g_old);
    }
};


int main(int argc, char* argv[]) {

    alfa[0] = 1.0;                  //alfa_1: entra como -alfa_1 |psi_1|^2
    alfa[1] = 1.0;                  //alfa_2: entra como +alfa_2 |psi_2|^2
    beta[0] = 1.0; beta[1] = 1.0;   //beta_1 = 1 fijo por el modelo, beta_2
    Vp = 0.5;                       //pinning: Vp > 0 debilita el orden del Al en los R_i
    w = 1.0;                        //ancho de las gaussianas de pinning
    //Origen de la red de pinning: la posicion de un vortice de la solucion con q = 0 y Vp = 0 (de density.dat).
    //Con q = 0 el borde fija donde queda la red de vortices; si los pozos no caen ahi, el minimo de G(q) se va a
    //q != 0 y el barrido sobre los ejes qx, qy no pasa por el.
    X0 = 0.0; Y0 = 0.0;

    if (argc < 11) {
        std::cerr << "Uso: " << argv[0]
                  << " Nx Ny dx Nv By K1 K2 d J seed.dat [qx qy]" << std::endl
                  << " seed.dat: 2*Nx*Ny lineas 'Re Im' (primero psi_1, despues psi_2)."
                  << " q = (qx, qy): twist externo exp(i q.r) en los bordes xy, por defecto 0." << std::endl;
        return 1;
    }

    Nx = std::stoi(argv[1]);
    Ny = std::stoi(argv[2]);
    dx = std::stod(argv[3]);
    Nv = std::stoi(argv[4]);
    By = std::stod(argv[5]);
    K[0] = std::stod(argv[6]);
    K[1] = std::stod(argv[7]);
    d  = std::stod(argv[8]);
    J  = std::stod(argv[9]);
    std::ifstream seed_file(argv[10]);
    qx = (argc > 11) ? std::stod(argv[11]) : 0.0;
    qy = (argc > 12) ? std::stod(argv[12]) : 0.0;

    Lx = Nx * dx;
    Ly = Ny * dx;
    Bz = 2.0 * M_PI * Nv / (Lx * Ly);
    int Ntot = 2 * Nx * Ny;
    double a[2] = {-alfa[0], alfa[1]};

    //Red triangular de Nv sitios de pinning: px * py = Nv, filas alternadas corridas media celda, con py par
    //para que cierre periodica en y. Se elige la factorizacion con ay/ax mas cerca de sqrt(3)/2: si la caja no
    //tiene la proporcion justa la red queda algo deformada (ax = Lx/px, ay = Ly/py).
    int px = 0, py = 0;
    double err = 0.0;
    for (int m = 1; m <= Nv; m++) {
        if (Nv % m || (m % 2 && Nv % 2 == 0)) continue;   //py = m par, salvo Nv impar (no hay)
        double e = std::abs((Ly / m) / (Lx / (Nv / m)) - std::sqrt(3.0) / 2);
        if (!py || e < err) { px = Nv / m; py = m; err = e; }
    }
    if (py % 2 && py > 1)
        std::cerr << "AVISO: Nv impar, la red de pinning (" << px << "x" << py << ") no es triangular en el borde y.\n";

    std::vector<double> pin(Nx * Ny, 0.0);
    for (int m = 0; m < py; m++)
        for (int n = 0; n < px; n++) {
            double X = X0 + (n + 0.5 * (m % 2)) * Lx / px, Y = Y0 + m * Ly / py;
            for (int j = 0; j < Ny; j++)
                for (int i = 0; i < Nx; i++) {
                    // ponytail: solo la imagen periodica mas cercana de cada R_i, vale si w << Lx/2, Ly/2
                    double rx = std::remainder(i * dx - X, Lx), ry = std::remainder(j * dx - Y, Ly);
                    pin[i + Nx * j] += Vp * std::exp(-(rx * rx + ry * ry) / (2 * w * w));
                }
        }
    thrust::device_vector<double> d_pin_vec = pin;
    const double* pin_ptr = thrust::raw_pointer_cast(d_pin_vec.data());
    cudaMemcpyToSymbol(d_pin, &pin_ptr, sizeof(pin_ptr));

    //Mapa del pinning (barrido_q.py lo lee para energia()): x y Vp sum_i exp(-|r-R_i|^2/2w^2)
    std::ofstream pin_file("pinning.dat");
    pin_file << std::setprecision(17);
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) pin_file << i * dx << " " << j * dx << " " << pin[i + Nx * j] << "\n";
        pin_file << "\n";
    }
    pin_file.close();

    //Parametro computacionales
    Threads = dim3(16, 16, 1);
    Blocks = dim3((Nx + 15) / 16, (Ny + 15) / 16, 2);
    int max_iter = 80000;
    double tol = 1e-10 * Ntot;
    int restart_period = 100;

    std::cout << "Nx=" << Nx << " Ny=" << Ny << " Nv=" << Nv
              << " dx=" << dx << " Bz(fuera de plano, multiplo del numero de cuantos de flujo magnetico)=" << Bz
              << " By(en plano)=" << By
              << std::setprecision(10)
              << " d=" << d << " J=" << J
              << " q=(" << qx << "," << qy << ")"
              << " K=(" << K[0] << "," << K[1] << ")"
              << " alfa=(" << alfa[0] << "," << alfa[1] << ")"
              << " beta=(" << beta[0] << "," << beta[1] << ")"
              << " Vp=" << Vp << " w=" << w << " red_pinning=" << px << "x" << py
              << " origen_pinning=(" << X0 << "," << Y0 << ")\n" << std::setprecision(6);

    cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(int));
    cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(int));
    cudaMemcpyToSymbol(d_dx, &dx, sizeof(double));
    cudaMemcpyToSymbol(d_Lx, &Lx, sizeof(double));
    cudaMemcpyToSymbol(d_Ly, &Ly, sizeof(double));
    cudaMemcpyToSymbol(d_qx, &qx, sizeof(double));
    cudaMemcpyToSymbol(d_qy, &qy, sizeof(double));
    cudaMemcpyToSymbol(d_Bz, &Bz, sizeof(double));
    cudaMemcpyToSymbol(d_By, &By, sizeof(double));
    cudaMemcpyToSymbol(d_d, &d, sizeof(double));
    cudaMemcpyToSymbol(d_J, &J, sizeof(double));
    cudaMemcpyToSymbol(d_K, K, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_a2, a, 2 * sizeof(double));
    cudaMemcpyToSymbol(d_a4, beta, 2 * sizeof(double));

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
                  << " sitios (o sobran datos). Esperado: 2*Nx*Ny lineas 'Re Im'.\n";
        return 1;
    }
    d_psi = psi;

    //Chequeo del gradiente contra la energia: dF/dh de F(psi - h grad) por diferencia centrada
    //tiene que dar -2 dx^2 |grad|^2. Si no coinciden, kernel_grad y energy_density_functor no son consistentes.
    compute_gradient(psi_ptr, grad_ptr);
    double g2 = dot(grad_ptr, grad_ptr, Ntot).real();
    if (g2 > 0.0) {
        double h = 1e-5 * std::sqrt(Ntot / g2);   //paso rms 1e-5 por sitio
        double Eh[2];
        for (int s = 0; s < 2; s++) {
            double hs = s ? -h : h;
            thrust::transform(d_psi.begin(), d_psi.end(), d_grad.begin(), trial.begin(),
                              [hs] __device__ (cd p, cd g){ return p - hs * g; });
            Eh[s] = compute_energy(trial_ptr, Ntot);
        }
        double num = (Eh[0] - Eh[1]) / (2.0 * h), ana = -2.0 * dx * dx * g2;
        std::cout << "Chequeo gradiente: numerico=" << num << " analitico=" << ana
                  << " error relativo=" << std::abs(num / ana - 1.0) << "\n";
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
        compute_gradient(psi_ptr, grad_ptr);

        double norm2 = dot(grad_ptr, grad_ptr, Ntot).real();

        //Corta si norm2 es NaN
        if(!std::isfinite(norm2)) {
            std::cerr << "Iter " << k << ": gradiente no finito, se corta.\n";
            break;
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

        double alpha = line_search_wolfe(psi_ptr, dir_ptr, grad_ptr, trial_ptr, g_trial_ptr, Ntot);

        //Si la direccion CG no dio paso valido, reinicia con descenso maximo
        if(alpha == 0.0 && beta != 0.0) {
            beta = 0.0;
            thrust::transform(d_grad.begin(), d_grad.end(), d_dir.begin(), thrust::negate<cd>());
            alpha = line_search_wolfe(psi_ptr, dir_ptr, grad_ptr, trial_ptr, g_trial_ptr, Ntot);
        }
        //Ni -grad baja la energia: estancado (precision o tol muy chica), se corta
        if(alpha == 0.0) {
            std::cerr << "Iter " << k << ": line search sin paso de descenso (|grad|^2 = "
                      << norm2 << "), se corta.\n";
            break;
        }

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

    //f(q) = F[psi] minimizada con este twist
    std::cout << std::setprecision(14) << "E_final= " << compute_energy(psi_ptr, Ntot) << "\n";

    thrust::copy(d_psi.begin(), d_psi.end(), psi.begin());

    write_field(psi, "density.dat", true);
    write_field(psi, "phase.dat", false);

    return 0;
}
