# read in dataset and convert to .mat format for use in MATLAB

import numpy as np
import scipy.io as sio
import h5py 


data = np.load('E:/sPIV_PLIF_ProcessedData/PIV/old/piv_baseline_u.npy')
data = np.nan_to_num(data)
print('Data shape:', data.shape)
print('Data dtype:', data.dtype)

data2 = np.load('E:/sPIV_PLIF_ProcessedData/PIV/old/piv_baseline_v.npy')
data2 = np.nan_to_num(data2)
print('Data2 shape:', data2.shape)
print('Data2 dtype:', data2.dtype)

with h5py.File('E:/sPIV_PLIF_ProcessedData/PIV/old/MATLAB/piv_baseline.h5', 'w') as hf:
    hf.create_dataset('u_velocity', data=data)
    hf.create_dataset('v_velocity', data=data2)

#sio.savemat('E:/sPIV_PLIF_ProcessedData/PLIF/MATLAB/plif_baseline_smoothed.mat', {'plif_baseline_smoothed': data})
# np.savetxt('E:/sPIV_PLIF_ProcessedData/PLIF/MATLAB/plif_baseline_smoothed.csv', data.reshape(-1, data.shape[-1]), delimiter=',')