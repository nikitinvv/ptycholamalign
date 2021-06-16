import numpy as np
import ptychotomo
import dxchange

n0 = 256
n1 = 256
n2 = 256
det = 256
ntheta = 128
phi = np.pi/3
niter = 32
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False).astype('float32')
f = -dxchange.read_tiff('data/delta-chip-256.tiff')
with ptychotomo.SolverLam(n0, n1, n2, det, ntheta, phi) as slv:
    data = slv.fwd_lam(f, theta)
    dxchange.write_tiff(data, 'data/r', overwrite=True)
    init = np.zeros([n0, n1, n2], dtype='float32')
    rec = slv.grad_lam(data,data*0+1, init, theta, niter, dbg=True)
    dxchange.write_tiff_stack(rec, 'rec/r', overwrite=True)