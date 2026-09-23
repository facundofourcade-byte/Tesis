#include <complex>
#include <cmath>
using cd = std::complex<double>;
#define HD
#include "params.inc"   // extraido del .cu por run.sh
#include "ops.inc"
extern "C" {
static Params P;
void setp(int nx,int ny,int nz,double dx,double by,double bz,double k1,double k3,double k4,double k6){
  P.Nx=nx;P.Ny=ny;P.Nz=nz;P.dx=dx;P.Lx=nx*dx;P.Ly=ny*dx;P.By=by;P.Bz=bz;P.k1=k1;P.k3=k3;P.k4=k4;P.k6=k6;}
static int N(){return P.Nx*P.Ny*P.Nz;}
void applyP(const cd* f, cd* out, double qx,double qy){
  for(int k=0;k<P.Nz;k++)for(int j=0;j<P.Ny;j++)for(int i=0;i<P.Nx;i++) out[lin(P,i,j,k)]=apply_P(P,f,i,j,k,qx,qy);}
void applyDx(const cd* f, cd* out){
  for(int k=0;k<P.Nz;k++)for(int j=0;j<P.Ny;j++)for(int i=0;i<P.Nx;i++) out[lin(P,i,j,k)]=cov_Dx(P,f,i,j,k,0);}
double energy(const cd* psi){
  cd* phi=new cd[N()]; applyP(psi,phi,0,0); double e=0;
  for(int k=0;k<P.Nz;k++)for(int j=0;j<P.Ny;j++)for(int i=0;i<P.Nx;i++) e+=energy_density(P,psi,phi,i,j,k,0,0);
  delete[] phi; return e*P.dx*P.dx*P.dx;}
void gradient(const cd* psi, cd* g){
  cd* phi=new cd[N()]; cd* chi=new cd[N()]; applyP(psi,phi,0,0);
  for(int k=0;k<P.Nz;k++)for(int j=0;j<P.Ny;j++)for(int i=0;i<P.Nx;i++) chi[lin(P,i,j,k)]=grad_chi(P,psi,phi,i,j,k);
  for(int k=0;k<P.Nz;k++)for(int j=0;j<P.Ny;j++)for(int i=0;i<P.Nx;i++) g[lin(P,i,j,k)]=grad_final(P,psi,phi,chi,i,j,k);
  delete[] phi; delete[] chi;}
}
