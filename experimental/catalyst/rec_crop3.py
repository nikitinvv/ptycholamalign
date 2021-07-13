import dxchange
import numpy as np
import sys
import ptychotomo 
from random import sample
import matplotlib.pyplot as plt
# data_prefix = '/gdata/RAVEN/vnikitin/nanomax/'

if __name__ == "__main__":
    
    nz = 576
    n = 576
    ndet = 128
    ntheta = 1
    ptheta = 1 
    voxelsize = 7.5e-5  # cm
    energy = 8.8
    nprb = 128  # probe size
    recover_prb = True
    
    # Reconstrucion parameters
    
    
    ngpus = 1
    
    data_prefix = sys.argv[1]
    id_theta = int(sys.argv[2])
    nscan = int(sys.argv[3])
    step = int(sys.argv[4])    
    piter = int(sys.argv[5])
    nmodes = int(sys.argv[6])
    
    data = np.zeros([1, nscan, ndet, ndet], dtype='float32')
    scan = np.zeros([2, 1, nscan], dtype='float32')-1
    theta = np.zeros([1],dtype='float32')
        
    data0 = np.load(data_prefix+'datanpy/datasorted_'+str(id_theta)+'.npy')
    #data0 = data0[:,::2,::2]+data0[:,1::2,1::2]
    scan0 = np.load(data_prefix+'datanpy/scansorted_'+str(id_theta)+'.npy')
    shifts0 = np.load(data_prefix+'/datanpy/shifts.npy')[id_theta]
    shifts1 = np.load(data_prefix+'/datanpy/shifts_sift.npy')[id_theta]
    shifts2 = np.load(data_prefix+'/datanpy/shifts_cm.npy')[id_theta]
    scan0[0] -= shifts0[0] + shifts1[0] + shifts2[0]
    scan0[1] -= shifts0[1] + shifts1[1] + shifts2[1]
    scan0[0] -= 64+64
    scan0[1] -= 320
    #scan0 /= 2


    # ignore position out of field of view            
    ids = np.where((scan0[0,0]<nz-nprb)*(scan0[1,0]<n-nprb)*(scan0[0,0]>=0)*(scan0[1,0]>=0))[0]
    
    plt.savefig(data_prefix+'tmp/fig'+str(id_theta)+'.png')    
    print(f'{len(ids)}')
    ids = ids[sample(range(len(ids)), min(len(ids),nscan))]
    
    scan[:,:,:min(len(ids),nscan)] = scan0[:, :, ids]
    data[0,:min(len(ids),nscan)] = data0[ids]    
    print(nscan)
    plt.plot(scan[1,0],scan[0,0],'r.')
    plt.xlim([-2,1280])
    plt.ylim([-2,768])
    plt.savefig(data_prefix+'tmp/fig'+str(id_theta)+'.png')
    # init probes
    prb = np.zeros([1, nmodes, nprb, nprb], dtype='complex64')
    prb0 = np.load(f'{data_prefix}datanpy/probessorted_{id_theta}.npy')[:nmodes]#,::2,::2]
    #prb0 = prb0[:,::2,::2]+prb0[:,1::2,1::2]
    prb[:]=prb0
    
    # Initial guess
    psi = np.ones([1, nz, n], dtype='complex64')    

    # data sh
    data = np.fft.fftshift(data, axes=(2, 3))/ndet/ndet
    
    
    name = data_prefix+'rec'+str(recover_prb)+str(nmodes)+str(scan.shape[2])

    with ptychotomo.SolverPtycho(ntheta, ptheta, nz, n, nscan, ndet, nprb, nmodes, voxelsize, energy, ngpus) as pslv:
        psi, prb = pslv.grad_ptycho_batch(
            data, psi, prb, scan, psi*0, -1, piter, recover_prb)   
    
    # Save result
    dxchange.write_tiff(np.angle(psi),  data_prefix+'rec_crop3/psiangle'+str(nmodes)+str(nscan)+'/r'+str(id_theta), overwrite=True)
    dxchange.write_tiff(np.abs(psi),   data_prefix+'rec_crop3/psiamp'+str(nmodes)+str(nscan)+'/r'+str(id_theta), overwrite=True)
    for m in range(nmodes):
        dxchange.write_tiff(np.angle(prb[:,m]),   data_prefix+'rec_crop3/prbangle/r'+str(m)+'_'+str(id_theta), overwrite=True)
        dxchange.write_tiff(np.abs(prb[:,m]),   data_prefix+'rec_crop3/prbamp/r'+str(m)+'_'+str(id_theta), overwrite=True)
        