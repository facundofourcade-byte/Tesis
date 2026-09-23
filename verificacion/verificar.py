# 1) gradiente = derivada exacta de la energia discreta (3D, Bz y By != 0)
# 2) en el interior: g = k6 D^6 - k4 D^4 - (k3 By/2){D^2,Dx} + D^2 + k1 By Dx + (|psi|^2-1) psi
# 3) caso 1D (psi uniforme en x, Bz=0): el minimo discreto converge a la solucion
#    espectral del continuo y cumple las condiciones de borde a), b), c) en z=0
import ctypes, numpy as np
from referencia_espectral import spectral, bcs
L=ctypes.CDLL('./libops.so')
L.energy.restype=ctypes.c_double
L.setp.argtypes=[ctypes.c_int]*3+[ctypes.c_double]*7
def ptr(a): return a.ctypes.data_as(ctypes.c_void_p)
def E(psi): return L.energy(ptr(psi))
def G(psi):
    g=np.zeros_like(psi); L.gradient(ptr(psi),ptr(g)); return g
def Pop(f):
    o=np.zeros_like(f); L.applyP(ptr(f),ptr(o),ctypes.c_double(0),ctypes.c_double(0)); return o
def Dx(f):
    o=np.zeros_like(f); L.applyDx(ptr(f),ptr(o)); return o

Nx,Ny,Nz,dx=6,5,24,0.4
By=0.4; Bz=2*np.pi/(Nx*dx*Ny*dx); k1,k3,k4,k6=0.7,0.5,0.8,0.3
L.setp(Nx,Ny,Nz,dx,By,Bz,k1,k3,k4,k6)
rng=np.random.default_rng(0); n=Nx*Ny*Nz
psi=rng.normal(size=n)+1j*rng.normal(size=n)
g=G(psi); err=0
for t in range(20):
    d=rng.normal(size=n)+1j*rng.normal(size=n); eps=1e-6
    fd=(E(psi+eps*d)-E(psi-eps*d))/(2*eps); an=2*dx**3*np.real(np.vdot(g,d))
    err=max(err,abs(fd-an)/abs(an))
print("1) error relativo max. derivada direccional:",err)
P1=Pop(psi); P2=Pop(P1); P3=Pop(P2)
ref=k6*P3-k4*P2-0.5*k3*By*(Pop(Dx(psi))+Dx(P1))+P1+k1*By*Dx(psi)+(abs(psi)**2-1)*psi
idx=np.arange(n).reshape(Nz,Ny,Nx)[10:Nz-10].ravel()
print("2) interior |g - formula|/|g| =",np.abs(g[idx]-ref[idx]).max()/np.abs(g[idx]).max())

Lz=6.0; B=0.5; k1,k3,k4,k6=1.0,0.6,0.5,0.2
r,fref,c,s=spectral(Lz,B,k1,k3,k4,k6,60)
print("3) referencia espectral: E/A =",r.fun,"  BCs a,b,c en z=0:",["%.1e"%abs(v) for v in bcs(c,s,Lz,B,k3,k4,k6)[0][:3]])
W8=np.array([-363/140,7,-21/2,35/3,-35/4,21/5,-7/6,1/7])
def newton(ps,Nz):
    for it in range(30):
        g=G(ps.astype(complex)).real; H=np.zeros((Nz,Nz)); h=1e-6
        for j in range(Nz):
            e=np.zeros(Nz); e[j]=h
            H[:,j]=(G((ps+e).astype(complex)).real-G((ps-e).astype(complex)).real)/(2*h)
        dlt=np.linalg.solve(0.5*(H+H.T),-g); ps=ps+dlt
        if np.abs(dlt).max()<1e-12: break
    return ps
for Nz in (49,97,193):
    dx=Lz/(Nz-1); L.setp(1,1,Nz,dx,B,0.0,k1,k3,k4,k6); z=np.arange(Nz)*dx
    ps=newton(fref(z),Nz); pc=ps.astype(complex)
    phi=Pop(pc); d4=Pop(phi); dxp=Dx(pc); h=0.5*k3*B
    dz=lambda f: sum(W8[m]*f[m] for m in range(8))/dx
    a=k6*dz(phi); b=-k4*phi[0]-h*dxp[0]+k6*d4[0]; cc=dz(pc)-k4*dz(phi)-h*dz(dxp)+k6*dz(d4)
    print(f"   Nz={Nz:4d} dx={dx:.4f}  E/A-ref={E(pc)/dx**2-r.fun:+.2e}  max|psi-ref|={np.abs(ps-fref(z)).max():.2e}"
          f"  BC z=0: |a|={abs(a):.1e} |b|={abs(b):.1e} |c|={abs(cc):.1e}  (escala ~{abs(dz(pc)):.0f})")
