import numpy as np
import dxchange
import ptychotomo

if __name__ == "__main__":
    
    # read object
    u = -dxchange.read_tiff('data/delta-chip-256.tiff')+1j*dxchange.read_tiff('data/beta-chip-256.tiff')
    u = u+1j*u/2

    n, _, _ = u.shape

    det = 256
    ntheta = 128
    phi = np.pi/3
    gamma = np.pi/4
    theta = np.linspace(0, 2*np.pi, ntheta).astype('float32')

    # simulate data
    with ptychotomo.SolverLam(n, n, n, det, ntheta, phi,gamma,eps=1e-2) as slv:
        data = slv.fwd_lam(u, theta)

    # adjoint test with data padding
    with ptychotomo.SolverLam(n, n, n, det, ntheta, phi,gamma,eps=1e-2) as slv:
        ua = slv.adj_lam(data,theta)

    with ptychotomo.SolverLam(n, n, n, det, ntheta, phi,gamma,eps=1e-2) as slv:
        data1 = slv.fwd_lam(ua, theta)        
        
    print(f'norm data = {np.linalg.norm(data)}')
    print(f'norm object = {np.linalg.norm(ua)}')
    print(
        f'<u,R*Ru>=<Ru,Ru>: {np.sum(u*np.conj(ua)):e} ? {np.sum(data*np.conj(data)):e}')
    ua*= 1/ntheta/n/n/n        
    data1*=1/ntheta/n/n/n        
    print(
        f'<Ru,RR*Ru>=<RR*Ru,RR*Ru>: {np.sum(data1*np.conj(data1)):e} ? {np.sum(data*np.conj(data1)):e}')        

    
    print(
        f'<u,R*Ru>=<R*Ru,R*Ru>: {np.sum(u*np.conj(ua)):e} ? {np.sum(ua*np.conj(ua)):e}')                
