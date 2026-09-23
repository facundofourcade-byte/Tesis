# Referencia continua 1D (psi uniforme en x, Bz=0): Galerkin Chebyshev de la energia,
# minimizada sin imponer ninguna condicion de borde (las naturales emergen).
import numpy as np
from numpy.polynomial import chebyshev as C, legendre as Lg
from scipy.optimize import minimize

def spectral(Lz,B,k1,k3,k4,k6,N=40,x0=None):
    tq,wq=Lg.leggauss(4*N)
    z=(tq+1)*Lz/2; wq=wq*Lz/2; s=2/Lz
    I=np.eye(N)
    V=[C.chebval(tq,I)]  # V[d][n,q]
    for d in range(1,6):
        V.append(np.array([C.chebval(tq,C.chebder(I[n],d))*s**d for n in range(N)]))
    V=[v.T for v in V]  # (q,n)
    b2z2=B*B*z*z
    def fields(c):
        p=[v@c for v in V]
        phi=b2z2*p[0]-p[2]
        dphi=2*B*B*z*p[0]+b2z2*p[1]-p[3]
        return p,phi,dphi
    def Ef(c):
        p,phi,dphi=fields(c); ps=p[0]
        f=(k6*(dphi**2+b2z2*phi**2)-k4*phi**2+k3*B*B*z*phi*ps+p[1]**2+b2z2*ps**2
           -k1*B*B*z*ps**2+0.5*ps**4-ps**2)
        # gradient
        gphi=2*k6*b2z2*phi-2*k4*phi+k3*B*B*z*ps
        gdphi=2*k6*dphi
        gps=k3*B*B*z*phi+2*b2z2*ps-2*k1*B*B*z*ps+2*ps**3-2*ps
        gp1=2*p[1]
        # phi = b2z2 p0 - p2 ; dphi = 2B^2 z p0 + b2z2 p1 - p3
        g0=gps+gphi*b2z2+gdphi*2*B*B*z
        g1=gp1+gdphi*b2z2
        g2=-gphi; g3=-gdphi
        G=V[0].T@(wq*g0)+V[1].T@(wq*g1)+V[2].T@(wq*g2)+V[3].T@(wq*g3)
        return wq@f, G
    if x0 is None:
        x0=np.zeros(N); x0[0]=0.8
    r=minimize(Ef,x0,jac=True,method='BFGS',options=dict(maxiter=100000,gtol=1e-12))
    return r, (lambda zz: C.chebval(zz*s-1,r.x)), r.x, s

def bcs(c,s,Lz,B,k3,k4,k6):
    # evalua a), b), c) en z=0 y z=Lz con derivadas exactas del polinomio
    out=[]
    for zz in (0.0,Lz):
        t=zz*s-1
        d=lambda k: C.chebval(t,C.chebder(c,k))*s**k if k>0 else C.chebval(t,c)
        # funciones de z: psi^(k)
        p=[d(k) for k in range(6)]
        b=B*B
        phi=b*zz*zz*p[0]-p[2]
        dphi=2*b*zz*p[0]+b*zz*zz*p[1]-p[3]
        # D^4 psi = (B^2 z^2 - d^2)(phi)
        d2phi=2*b*p[0]+4*b*zz*p[1]+b*zz*zz*p[2]-p[4]
        d3phi=6*b*p[1]+6*b*zz*p[2]+b*zz*zz*p[3]-p[5]
        D4=b*zz*zz*phi-d2phi
        dD4=2*b*zz*phi+b*zz*zz*dphi-d3phi
        Dxpsi=-B*zz*p[0]; dDxpsi=-B*p[0]-B*zz*p[1]
        h=0.5*k3*B
        a=k6*dphi
        bb=-k4*phi-h*Dxpsi+k6*D4
        cc=p[1]-k4*dphi-h*dDxpsi+k6*dD4
        out.append((a,bb,cc,abs(k4*phi)+abs(h*Dxpsi)+abs(k6*D4)))
    return out

if __name__=="__main__":
    Lz=6.0; B=0.5; k1,k3,k4,k6=1.0,0.6,0.5,0.2
    for N in (30,45,60):
        r,f,c,s=spectral(Lz,B,k1,k3,k4,k6,N)
        print(N,r.fun,r.message,"psi(0..0.5)",f(np.array([0,0.125,0.25,0.375,0.5])))
        print("  BCs",bcs(c,s,Lz,B,k3,k4,k6))
