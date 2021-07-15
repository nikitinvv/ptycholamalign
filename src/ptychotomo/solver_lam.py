"""Module for tomography."""

import cupy as cp
import numpy as np
import threading
import concurrent.futures as cf
from .lamusfft import lamusfft
from .utils import chunk
from functools import partial
import matplotlib
matplotlib.use('Agg')

class SolverLam(lamusfft):
    """Base class for tomography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    n0 : int
        Object size in x.
    n1 : int
        Object size in y.
    n2 : int
        Object size in z.
    det : int
        Detector size in one dimension.
    ntheta : int
        The number of projections.
    eps : float
        Accuracy for the USFFT computation. Default: 1e-3.
    """

    def __init__(self, n0, n1, n2, det, ntheta, phi, gamma, eps=1e-3, ngpus=1):
        """Please see help(SolverLam) for more info."""
        # create class for the tomo transform associated with first gpu
        super().__init__(n2, n1, n0, det, ntheta, phi, gamma, eps, ngpus)  # reorder sizes


    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_lam(self, u, theta, gpu_arrays=False):
        """Laminography transform (L)"""
        res = cp.zeros([self.ntheta, self.det, self.det], dtype='complex64')

        u_gpu = cp.asarray(u.astype('complex64'))
        theta_gpu = cp.asarray(theta)

        # C++ wrapper, send pointers to GPU arrays
        self.fwd(res.data.ptr, u_gpu.data.ptr, theta_gpu.data.ptr)

        if(np.isrealobj(u)):
            res = res.real
        if(isinstance(u, np.ndarray)):
            res = res.get()
        return res

    def adj_lam(self, data, theta):
        """Adjoint Laminography transform (L^*)"""
        res = cp.zeros([self.n2, self.n1, self.n0], dtype='complex64')

        data_gpu = cp.asarray(data.astype('complex64'))
        theta_gpu = cp.asarray(theta)

        # C++ wrapper, send pointers to GPU arrays
        self.adj(res.data.ptr, data_gpu.data.ptr, theta_gpu.data.ptr)

        if(cp.isrealobj(data)):
            res = res.real
        if(isinstance(data, np.ndarray)):
            res = res.get()
        return res

    def grad_lam(self, data0, K0, u0, theta0, titer, dbg=False):
        """CG solver for ||KLu-data||_2"""
        u = cp.asarray(u0)
        theta = cp.asarray(theta0)
        data = cp.asarray(data0)
        K = cp.asarray(K0)

        # minimization functional
        def minf(KLu):
            f = cp.linalg.norm(KLu-data)**2
            return f
        for i in range(titer):
            KLu = K*self.fwd_lam(u, theta)
            grad = self.adj_lam(cp.conj(K)*(KLu-data), theta) * 1 / \
                self.ntheta/self.n0/self.n1/self.n2

            u -= 0.5*grad
            if (dbg == True):
                print("%4d, %.7e" %
                      (i,  minf(KLu)))

        if(isinstance(u0, np.ndarray)):
            u = u.get()
        return u

    
    