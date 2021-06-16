import os
import signal
import sys
import h5py
import cupy as cp
import dxchange
import numpy as np
import matplotlib.pyplot as plt
import ptychotomo as pt
from random import sample 

data_prefix = '/data/staff/tomograms/vviknik/nanomax/'

if __name__ == "__main__":
    
   
    n = 512-128
    nz = 512-192
    det = [128, 128]
    voxelsize = 18.03*1e-7  # cm
    energy = 12.4
    nprb = 128  # probe size
    recover_prb = True
    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 7*1e-14  # tv regularization penalty coefficient
    piter = 256  # ptychography iterations
    nmodes = 4
    ngpus = 1
    nscan = 13689
    
    id_theta = int(sys.argv[1])
    
    # Load a 3D object
    prb = np.zeros([1, nmodes, nprb, nprb], dtype='complex64',order='C')
    prb[:] = np.load(data_prefix+'datanpy/prb128.npy')

    scan0 = np.load(data_prefix+'datanpy/scan128sorted_'+str(id_theta)+'.npy')   
    shifts = np.load(data_prefix+'/datanpy/shifts.npy')[id_theta]
    # shiftspart = np.load(data_prefix+'/datanpy/shiftscrop.npy')[id_theta]
    shiftssum = np.load(data_prefix+'/datanpy/shiftssum.npy')[id_theta]
    scan0[1]-=np.round(shifts[1]+shiftssum[1])
    scan0[0]-=np.round(shifts[0]+shiftssum[0])
    scan0[1]-=64+30
    scan0[0]-=160


    ids = np.where((scan0[1,0]<n-nprb)*(scan0[0,0]<nz-nprb)*(scan0[0,0]>=0)*(scan0[1,0]>=0))[0]
    print(len(ids))

    #ids = sample(ids,nsncan)


    data = np.zeros([1, nscan, det[0], det[1]], dtype='float32')
    scan = np.zeros([2, 1, nscan], dtype='float32')-1            
    theta = np.zeros([1],dtype='float32')    
    scan[0,0,:len(ids)] = scan0[1,0,ids]
    scan[1,0,:len(ids)] = scan0[0,0,ids]
    data[0,:len(ids)] = np.load(data_prefix+'datanpy/data128sorted_'+str(id_theta)+'.npy')[ids]

    # Load a 3D object
    prb = np.zeros([1, nmodes, nprb, nprb], dtype='complex64',order='C')
    prb[:] = np.load(data_prefix+'datanpy/prb128.npy')
    
    psi = np.ones([1, nz, n], dtype='complex64', order='C')*1    
    data = np.fft.fftshift(data, axes=(2, 3))
    # Class gpu solver
    slv = pt.Solver(nscan, theta, n/2, det, voxelsize,
                    energy, 1, nz, n, nprb, 1, 1, nmodes, ngpus)
    name = data_prefix+'rec'+str(recover_prb)+str(nmodes)+str(scan.shape[2])

    psi, prb = slv.cg_ptycho_batch(
        data/det[0]/det[1], psi, prb, scan, None, -1, piter, model, recover_prb)

    # Save result
    dxchange.write_tiff(np.angle(psi),  data_prefix+'rec_crop_sum/psiangle'+str(nmodes)+str(nscan)+'/r'+str(id_theta), overwrite=True)
    dxchange.write_tiff(np.abs(psi),   data_prefix+'rec_crop_sum/psiamp'+str(nmodes)+str(nscan)+'/r'+str(id_theta), overwrite=True)
    for m in range(nmodes):
        dxchange.write_tiff(np.angle(prb[:,m]),   data_prefix+'rec_crop_sum/prbangle/r'+str(id_theta), overwrite=True)
        dxchange.write_tiff(np.abs(prb[:,m]),   data_prefix+'rec_crop_sum/prbamp/r'+str(id_theta), overwrite=True)
        
        