import numpy as np
import ptychotomo
import dxchange

n0 = 256
n1 = 256
n2 = 256
det = 256
ntheta = 256
phi = np.pi/3
gamma = np.pi/4
niter = 32

theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False).astype('float32')
f = -dxchange.read_tiff('data/delta-chip-256.tiff')+1j*dxchange.read_tiff('data/beta-chip-256.tiff')
with ptychotomo.SolverLam(n0, n1, n2, det, ntheta, phi,gamma,eps=1e-1) as slv:
    data = slv.fwd_lam(f, theta)
    dxchange.write_tiff(data.real, 'data/re', overwrite=True)
    dxchange.write_tiff(data.imag, 'data/im', overwrite=True)
    init = np.zeros([n0, n1, n2], dtype='complex64')
    rec = slv.grad_lam(data,data*0+1, init, theta, niter, dbg=True)
    dxchange.write_tiff_stack(rec.real, 'rec/re', overwrite=True)
    dxchange.write_tiff_stack(rec.imag, 'rec/im', overwrite=True)