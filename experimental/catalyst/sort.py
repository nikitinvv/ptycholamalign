import matplotlib.pyplot as plt
import numpy as np
import dxchange
import os
import sys

if __name__ == "__main__":
    
    data_prefix = sys.argv[1]
    nscan = int(sys.argv[2])
    ntheta = int(sys.argv[3])
    nmodes = int(sys.argv[4])
    
    ndet = 128
    theta = np.zeros(ntheta, dtype='float32')
    scan = np.zeros([2,ntheta,nscan], dtype='float32')-1
    
    for k in range(0,ntheta):        
        # Load a 3D object
        print(k)
        theta[k] = np.load(data_prefix+'datanpy/theta_'+str(k)+'.npy')
        scan0 = np.load(data_prefix+'datanpy/scan_'+str(k)+'.npy')
        scan[:,k:k+1,:scan0.shape[2]] = scan0
    
    ids = np.argsort(theta)
    theta = theta[ids]
    scan = scan[:,ids]
    
    
    for k in range(0,ntheta):        
        print(k)        
        data = np.load(data_prefix+'datanpy/data_'+str(ids[k])+'.npy')        
        probes = np.load(data_prefix+'datanpy/probes_'+str(ids[k])+'.npy')        
        np.save(data_prefix+'/datanpy/thetasorted_'+str(k)+'.npy',theta[k])    
        np.save(data_prefix+'/datanpy/datasorted_'+str(k)+'.npy',data)
        np.save(data_prefix+'/datanpy/scansorted_'+str(k)+'.npy',scan[:,k:k+1])
        np.save(data_prefix+'/datanpy/probessorted_'+str(k)+'.npy',probes)        