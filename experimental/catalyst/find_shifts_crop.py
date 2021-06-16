import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import sys

if __name__ == "__main__":

    data_prefix = sys.argv[1]
    nscan = int(sys.argv[2])
    ntheta = int(sys.argv[3])
    nmodes = int(sys.argv[4])
    
    nz = 640
    n = 640
    ndet = 128
    #tilt_angle = -72.035
    data = np.zeros([ntheta,nz,n],dtype='float32')
    for k in range(ntheta):
        data[k] = dxchange.read_tiff(data_prefix+'rec_crop/psiangle'+str(nmodes)+str(nscan)+'/r'+str(k)+'.tiff').astype('float32')       
    shift = np.zeros((ntheta,2),dtype='float32')
    for k in range(ntheta):
        #data[k] = ndimage.rotate(data[k],-tilt_angle,reshape=False)
        data0 = data[k]*(data[k]>0)
        cm = ndimage.measurements.center_of_mass(data0[64:-64,64:-64])
        #print(cm)
        shift0 = [cm[0]-nz/2+64,cm[1]-n/2+64]
        print(shift0)
        data[k] = np.roll(data[k],(-int(shift0[0]),-int(shift0[1])),axis=(0,1))
        shift[k] = shift0        
        dxchange.write_tiff(data[k],data_prefix+'rec_crop_aligned/psiangle'+str(nmodes)+str(nscan)+'/r'+str(k)+'.tiff',overwrite=True)
    np.save(data_prefix+'/datanpy/shiftscrop.npy',shift)

