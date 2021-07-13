import numpy as np
import dxchange
import ptychotomo
from random import sample
import matplotlib.pyplot as plt
import sys
import h5py
import cupy
import scipy.ndimage as ndimage
#cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)
if __name__ == "__main__":    

    # read object
    n = 544  # object size n x,y
    ntheta = 168  # number of angles
    ptheta = 1  # partial size for ntheta
    voxelsize = 7.5e-5  # cm
    energy = 8.8
    ndet = 128  # detector size
    nprb = 128  # probe size
    nmodes = 7  # number of probe modes
    ngpus = 1  # number of GPUs

    data_prefix = sys.argv[1]
    nscan = int(sys.argv[2])
    align = int(sys.argv[3])    
    sptycho = int(sys.argv[4])
    initptycho = int(sys.argv[5])

    # reconstruction paramters
    recover_prb = bool(sys.argv[6])
    piter = 32  # ptycho iterations
    titer = 32 # tomo iterations
    diter = 32  # deform iterations
    niter = 256  # admm iterations

    dbg_step = 4
    step_flow = 2
    start_win = 544

    h5file = h5py.File(f'{data_prefix}/catalyst/extracted_scan192.h5', 'r')
    tilt_angle = h5file.attrs.get('tilt_angle')
    lamino_angle = h5file.attrs.get('lamino_angle')*np.pi/180        
    # Load probe
    prb = np.zeros([ntheta, nmodes, nprb, nprb], dtype='complex64')
    prb[:] = np.load(f'{data_prefix}datanpy/probessorted_0.npy')[:nmodes]
    for j in range(ntheta):
        for k in range(7):
            prb[j,k].real = ndimage.rotate(prb[j,k].real,-tilt_angle,reshape=False,order=1)
            prb[j,k].imag = ndimage.rotate(prb[j,k].imag,-tilt_angle,reshape=False,order=1)

    theta = np.zeros(ntheta, dtype='float32')
    data = np.zeros([ntheta, nscan, ndet, ndet], dtype='float32')
    scan = -np.ones([2, ntheta, nscan], dtype='float32')

    for k in range(ntheta):
        print('read angle', k)
        data0 = np.load(data_prefix+'datanpy/datasorted_'+str(k)+'.npy')
        scan0 = np.load(data_prefix+'datanpy/scansorted_'+str(k)+'.npy')
        shifts0 = np.load(data_prefix+'/datanpy/shifts.npy')[k]
        shifts1 = np.load(data_prefix+'/datanpy/shifts_sift.npy')[k]
        shifts2 = np.load(data_prefix+'/datanpy/shifts_cm.npy')[k]
        scan0[0] -= shifts0[0] + shifts1[0] + shifts2[0]
        scan0[1] -= shifts0[1] + shifts1[1] + shifts2[1]
        scan0[0] -= 64+64+32+16
        scan0[1] -= 320+32+16
        # ignore position out of field of view            
        ids = np.where((scan0[0,0]<n-nprb)*(scan0[1,0]<n-nprb)*(scan0[0,0]>=0)*(scan0[1,0]>=0))[0]    
        ids = ids[sample(range(len(ids)), min(len(ids),nscan))]
        print(f'{len(ids)}')
        scan[:,k,:min(len(ids),nscan)] = scan0[:, 0, ids]
        data[k,:min(len(ids),nscan)] = data0[ids]#,::2,::2]
        theta[k] = np.load(data_prefix+'datanpy/thetasorted_'+str(k)+'.npy')

    # normaliza data to the detector size and fftshift it
    data = np.fft.fftshift(data, axes=(2, 3))/ndet/ndet
    # Initial guess
    # variable index: 1 - ptycho problem, 2 - regularization (not implemented), 3 - tomo
    # used as in the pdf documents
    h1 = np.ones([ntheta, n, n], dtype='complex64')
    h3 = np.ones([ntheta, n, n], dtype='complex64')
    psi1 = np.ones([ntheta, n, n], dtype='complex64')    
    psi3 = np.ones([ntheta, n, n], dtype='complex64')
    lamd1 = np.zeros([ntheta, n, n], dtype='complex64')
    lamd3 = np.zeros([ntheta, n, n], dtype='complex64')
    u = np.zeros([n, n, n], dtype='complex64')
    flow = np.zeros([ntheta, n, n, 2], dtype='float32')


    # init with correct values
    for k in range(ntheta):
        if(initptycho==1):
            psiangle = dxchange.read_tiff(data_prefix+'rec_crop3/psiangle'+str(nmodes)+'800/r'+str(k)+'.tiff')[:,16:-16,16:-16]
            psiamp = dxchange.read_tiff(data_prefix+'rec_crop3/psiamp'+str(nmodes)+'800/r'+str(k)+'.tiff')[:,16:-16,16:-16]
            psi1[k] = psiamp*np.exp(1j*psiangle) 
            psi3[k] = psiamp*np.exp(1j*psiangle) 
            h1[k] = psiamp*np.exp(1j*psiangle) 
            h3[k] = psiamp*np.exp(1j*psiangle) 
        if(recover_prb==False):
            for m in range(nmodes):
                prbangle = dxchange.read_tiff(data_prefix+'rec_crop3/prbangle/r'+str(m)+'_'+str(k)+'.tiff')
                prbamp = dxchange.read_tiff(data_prefix+'rec_crop3/prbamp/r'+str(m)+'_'+str(k)+'.tiff')
                prb[k,m] = prbamp*np.exp(1j*prbangle) 
                

    data_prefix += 'rec_test/'+str(nscan)+'align'+str(align)+str(sptycho)+str(initptycho)+str(step_flow)+str(recover_prb)+'/'
    with ptychotomo.SolverAdmm(nscan, theta, lamino_angle, ndet, voxelsize, energy,
                               ntheta, n, n, nprb, ptheta, nmodes, ngpus) as aslv:
        u, psi1, psi3, flow, prb = aslv.admm_lam(
            data, psi1, psi3, flow, prb, scan,
            h1, h3, lamd1, lamd3,
            u, piter, titer, diter, niter, recover_prb, align, start_win=start_win,
            step_flow=step_flow, name=data_prefix+'tmp/', dbg_step=dbg_step, sptycho=sptycho)

    dxchange.write_tiff_stack(
        np.angle(psi1), data_prefix+'rec_admm/psiangle/p', overwrite=True)
    dxchange.write_tiff_stack(
        np.abs(psi1), data_prefix+'rec_admm/psiamp/p', overwrite=True)
    dxchange.write_tiff_stack(
        np.angle(psi3), data_prefix+'rec_admm/psi3angle/p', overwrite=True)
    dxchange.write_tiff_stack(
        np.abs(psi3), data_prefix+'rec_admm/psi3amp/p', overwrite=True)

    dxchange.write_tiff_stack(u.real,data_prefix+'rec_admm/ure/u', overwrite=True)
    dxchange.write_tiff_stack(u.imag,data_prefix+'rec_admm/uim/u', overwrite=True)
