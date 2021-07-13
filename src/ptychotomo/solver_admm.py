

from .solver_deform import SolverDeform
from .solver_ptycho import SolverPtycho
from .solver_lam import SolverLam
from .flowvis import flow_to_color
from .utils import *
import numpy as np
import signal
import dxchange
import sys
import os
import matplotlib.pyplot as plt
import gc


class SolverAdmm(object):
    def __init__(self, nscan, theta, lamino_angle, ndet, voxelsize, energy, ntheta, nz, n, nprb, ptheta, nmodes, ngpus):

        self.ntheta = ntheta
        self.nz = nz
        self.n = n
        self.nscan = nscan
        self.ndet = ndet
        self.nprb = nprb
        self.ptheta = ptheta
        self.nmodes = nmodes
        self.ngpus = ngpus
        self.theta=theta

        eps=1e-1
        self.tslv = SolverLam(n,n,n,n,ntheta,lamino_angle,eps,ngpus)
        self.pslv = SolverPtycho(
            ntheta, ptheta, nz, n, nscan, ndet, nprb, nmodes, voxelsize, energy, ngpus)
        self.dslv = SolverDeform(ntheta, nz, n, ptheta, ngpus)

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)

    def signal_handler(self, sig, frame):  # Free gpu memory after SIGINT, SIGSTSTP
        self.tslv = []
        self.pslv = []
        self.dslv = []
        sys.exit(0)

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        return

    def update_penalty(self, psi1, h1, h10, psi3, h3, h30, rho1, rho3):
        """Update rho, for a faster convergence"""
        # rho1
        
        r = np.linalg.norm(psi1 - h1)**2
        s = np.linalg.norm(rho1*(h1-h10))**2
        if (r > 10*s):
            rho1 *= 2
        elif (s > 10*r):
            rho1 *= 0.5
        # rho3
        r = np.linalg.norm(psi3 - h3)**2
        s = np.linalg.norm(rho3*(h3-h30))**2
        if (r > 10*s):
            rho3 *= 2
        elif (s > 10*r):
            rho3 *= 0.5

        return rho1, rho3

    def take_lagr(self, psi1, psi3, data, prb, scan, h1, h3, lamd1, lamd3, rho1, rho3):
        """Lagrangian terms for monitoring convergence"""
        lagr = np.zeros(6, dtype="float32")
        lagr[0] = self.pslv.take_error(data, psi1, prb, scan)
        lagr[1] = 2*np.sum(np.real(np.conj(lamd1)*(h1-psi1)))
        lagr[2] = rho1*np.linalg.norm(h1-psi1)**2
        lagr[3] = 2*np.sum(np.real(np.conj(lamd3)*(h3-psi3)))
        lagr[4] = rho3*np.linalg.norm(h3-psi3)**2
        lagr[5] = np.sum(lagr[0:5])

        return lagr

    # ADMM for ptycho-tomography problem
    def admm_lam(self, data, psi1, psi3, flow, prb, scan,
             h1, h3, lamd1, lamd3,
             u, piter, titer, diter, niter, 
             recover_prb, align, start_win, 
             step_flow=1, name='tmp/', dbg_step=8, sptycho=True):

        # data /= (self.ndetx*self.ndety)  # FFT compensation  (should be done for real data)
        pars = [0.5, 1, start_win, 4, 5, 1.1, 4]
        rho1, rho3 = 0.5, 0.5
        t=np.zeros(4)
        for i in range(niter):
            # keep previous iteration for penalty updates
            h10,  h30 = h1,  h3
            
            # solve ptycho
            tic()
            if(sptycho==1):
                psi1, prb = self.pslv.grad_ptycho_batch(
                    data, psi1, prb, scan, h1+lamd1/rho1, rho1, piter, recover_prb)
            t[0]+=toc()
            gc.collect()
            # solve deform
            tic()
            mmin, mmax = find_min_max(np.angle(psi1-lamd1/rho1))
            flow = self.dslv.registration_flow_batch(
                np.angle(psi3), np.angle(psi1-lamd1/rho1), mmin, mmax, flow, pars)*align
            psi3 = self.dslv.grad_deform_gpu_batch(
                psi1-lamd1/rho1, psi3, flow, diter, h3+lamd3/rho3, rho3/rho1)            
            t[1]+=toc()
            gc.collect()

            tic()
            # solve tomo
            xi0, K, pshift = self.pslv.takexi(psi3, lamd3, rho3)
            u = self.tslv.grad_lam(xi0, K, u, self.theta, titer)
            t[2]+=toc() 
            gc.collect()
            
            # update h1, h3
            tic()
            if(sptycho==1):
                h1 = self.dslv.apply_flow_gpu_batch(
                    psi3.real, flow)+1j*self.dslv.apply_flow_gpu_batch(psi3.imag, flow)
            h3 = self.pslv.exptomo(
                self.tslv.fwd_lam(u, self.theta))*np.exp(1j*pshift)
            t[3]+=toc()             
            
            # lamd updates
            if(sptycho==1):
                lamd1 = lamd1 + rho1 * (h1-psi1)
            lamd3 = lamd3 + rho3 * (h3-psi3)
            
            # update rho for a faster convergence
            rho1, rho3 = self.update_penalty(
                psi1, h1, h10, psi3, h3, h30, rho1, rho3)
            gc.collect()
            # decrease the step for optical flow window
            pars[2] -= step_flow
            # Lagrangians difference between two iterations
            if (i % dbg_step == 0):
               # lagr = self.take_lagr(
                #    psi1, psi3, data, prb, scan, h1, h3, lamd1, lamd3, rho1, rho3)
                #print(f"{i}/{niter}) flow:{np.linalg.norm(flow)}, {pars[2]}, {rho1:.2e}, {rho3:.2e}",
                      #"Lagrangian terms: [", *(f"{x:.1e}" for x in lagr), "]")
                print(f"ptycho:{t[0]/60}, deform: {t[1]/60}, lam: {t[2]/60}, update: {t[3]/60}")
                self.msave(psi1,psi3,u,flow,name,i)
                
        return u, psi1, psi3, flow, prb

    def msave(self,psi1,psi3,u,flow,name,i):
        dxchange.write_tiff_stack(np.angle(psi3),
                                    name+'psi3iter'+str(self.n)+'/'+str(i), overwrite=True)
        dxchange.write_tiff_stack(np.abs(psi3),
                                    name+'psi3iterabs'+str(self.n)+'/'+str(i), overwrite=True)
        dxchange.write_tiff_stack(np.angle(psi1),
                                    name+'psi1iter'+str(self.n)+'/'+str(i), overwrite=True)
        dxchange.write_tiff_stack(np.abs(psi1),
                                    name+'psi1iterabs'+str(self.n)+'/'+str(i), overwrite=True)
        dxchange.write_tiff_stack(np.real(u),
                                    name+'ure'+str(self.n)+'/'+str(i), overwrite=True)
        dxchange.write_tiff_stack(np.imag(u),
                                    name+'uim'+str(self.n)+'/'+str(i), overwrite=True)
        
        if not os.path.exists(name+'flow/'):
            os.makedirs(name+'flow/')
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.imshow(flow_to_color(flow[0]))
        plt.subplot(2, 2, 2)
        plt.imshow(flow_to_color(flow[self.ntheta//4]))
        plt.subplot(2, 2, 3)
        plt.imshow(flow_to_color(flow[3*self.ntheta//4]))
        plt.subplot(2, 2, 4)
        plt.imshow(flow_to_color(flow[self.ntheta-1]))
        plt.savefig(name+'flow/'+str(i)+'.png')
        np.save(name+'flow/'+str(i), flow)
