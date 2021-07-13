import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import ptychotomo
import sys
import h5py

if __name__ == "__main__":

   
    data_prefix = sys.argv[1]
    nscan = int(sys.argv[2])
    piter = int(sys.argv[3])
    nmodes = int(sys.argv[4])
    step  = int(sys.argv[5])

    h5file = h5py.File(f'{data_prefix}/catalyst/extracted_scan192.h5', 'r')
    lamino_angle = h5file.attrs.get('lamino_angle')*np.pi/180
    n = 640//step
    det = 640//step
    ngpus = 1
    ntheta = 168    
    theta = np.zeros([ntheta],dtype='float32')
    
    psi = np.zeros([ntheta, n, n], dtype='complex64', order='C')
    for k in range(ntheta):        
        psiangle = dxchange.read_tiff(data_prefix+'rec_crop3/psiangle'+str(nmodes)+str(nscan)+'/r'+str(k)+'.tiff')        
        psi[k] = psiangle[0,::step,::step]
        theta[k] = np.load(data_prefix+'datanpy/thetasorted_'+str(k)+'.npy')    
    #psi = dxchange.read_tiff(data_prefix+'/matlab-recon.tif')[:,::2,::2].astype('complex64')
    
    center = n//2
    u = np.zeros([n,n,n],dtype='complex64')#(self, n0, n1, n2, det, ntheta, phi, eps=1e-3):
    with ptychotomo.SolverLam(n,n,n,det,ntheta,lamino_angle,eps=1e-2) as tslv:
        u = tslv.grad_lam(psi, psi*0+1, u,theta, piter,dbg=True)        
        dxchange.write_tiff_stack(u.real,data_prefix+'rec_lam'+str(n)+str(nscan)+'/r'+str(center)+'.tiff',overwrite=True)
        

    