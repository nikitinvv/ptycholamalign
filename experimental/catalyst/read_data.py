import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.ndimage as ndimage
data_prefix = sys.argv[1]
def read_data(id_data):
    try:
        h5file = h5py.File(f'{data_prefix}/catalyst/extracted_scan{id_data}.h5', 'r')
        data = h5file['data'][:].astype('float32')
        positions = h5file['positions_1'][:].astype('float32')
        probes = h5file['recprobe'][:].astype('complex64')
        detector_pixel_size = h5file.attrs.get('detector_pixel_size')
        detector_distance = h5file.attrs.get('detector_distance')
        incident_wavelength = h5file.attrs.get('incident_wavelength')
        lamino_angle = h5file.attrs.get('lamino_angle')
        tilt_angle = h5file.attrs.get('tilt_angle')
        skewness_angle = h5file.attrs.get('skewness_angle')
        #print(f'{lamino_angle},{tilt_angle},{skewness_angle}')
        theta = h5file.attrs.get('rotation_angle')*np.pi/180
        pos2det_const = (detector_pixel_size*probes.shape[-1]/(detector_distance*1e-10*incident_wavelength))
        positions *= pos2det_const
        #print(positions.shape)
        #exit()
        positions = positions[:,::-1]
        positionsrot = positions.copy()
        positionsrot[:,1] = positions[:,1]*np.cos(-tilt_angle*np.pi/180)+positions[:,0]*np.sin(-tilt_angle*np.pi/180)
        positionsrot[:,0] = -positions[:,1]*np.sin(-tilt_angle*np.pi/180)+positions[:,0]*np.cos(-tilt_angle*np.pi/180)
        for k in range(7):
            probes[k].real = ndimage.rotate(probes[k].real,-tilt_angle,reshape=False,order=1)
            probes[k].imag = ndimage.rotate(probes[k].imag,-tilt_angle,reshape=False,order=1)
        data = ndimage.rotate(data, -tilt_angle, reshape=False, axes=(2,1), order=1).astype('float32')
        data[data<0]=0
        positionsrot[:,0] -= np.amin(positionsrot[:,0])
        positionsrot[:,1] -= np.amin(positionsrot[:,1])    
        scan = positionsrot.swapaxes(0,1)[:,np.newaxis,:].astype('float32')        
    except:
        scan = None
        theta = None
        data = None
        probes = None
    return data, scan, theta, probes


if __name__ == "__main__":   
    kk = 0
    for k in range(192,380):
        print(kk, k)
        data,scan,theta,probes = read_data(k)
        if(scan is not None):            
            plt.clf()
            plt.plot(scan[1],scan[0],'r.',markersize=1)            
            plt.axis('equal')
            plt.savefig(data_prefix+'/scan_pos/'+str(kk)+'.png',dpi=450)
            print('theta',theta)
            print('scan',scan.shape,np.min(scan[0]),np.min(scan[1]),np.max(scan[0]),np.max(scan[1]))
            print('data',data.shape,np.linalg.norm(data))
            np.save(data_prefix+'datanpy/theta_'+str(kk),theta)
            np.save(data_prefix+'datanpy/scan_'+str(kk),scan)
            np.save(data_prefix+'datanpy/data_'+str(kk),data)            
            np.save(data_prefix+'datanpy/probes_'+str(kk),probes)            
            kk += 1                   