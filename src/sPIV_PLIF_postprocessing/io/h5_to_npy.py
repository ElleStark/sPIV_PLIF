# read in .h5 files and save as .npy for easier loading in Python

import h5py
import numpy as np
import matplotlib.pyplot as plt

# file paths
# h5_path = "E:/sPIV_PLIF_ProcessedData/PLIF/Baseline_PLIF.h5"
# npy_path = "E:/sPIV_PLIF_ProcessedData/PLIF/baseline_ygrid.npy"

# # read .h5 file
# with h5py.File(h5_path, "r") as f:

#     # read dataset 
#     # dataset_key = "DataSets/Baseline_Data"  
#     dataset_key = "Mapping/y_grid" 
#     if dataset_key not in f:
#         raise KeyError(f"Dataset key '{dataset_key}' not found in {h5_path}")
    
#     data = f[dataset_key][()]
#     print(f"Loaded data shape: {data.shape}, dtype: {data.dtype}")

# # save as .npy
# np.save(npy_path, data)
# print(f"Saved data to {npy_path}")


# data = np.load("E:/sPIV_PLIF_ProcessedData/PIV/Interpolated_to_PLIF/piv_baseline_v.npy")
# print(f"Loaded data shape: {data.shape}, dtype: {data.dtype}")
# data = np.transpose(data, (2, 0, 1))  # reorder axes to (frames, y, x)
# print(f"Transposed data shape: {data.shape}, dtype: {data.dtype}")
# np.save("E:/sPIV_PLIF_ProcessedData/PIV/Interpolated_to_PLIF/piv_baseline_v.npy", data)
# print(f"Saved transposed data to E:/sPIV_PLIF_ProcessedData/PIV/Interpolated_to_PLIF/piv_baseline_v.npy")


data = np.load("E:/sPIV_PLIF_ProcessedData/PLIF/Baseline_PLIF.npy")

# data = np.flip(data, axis=1)  # flip vertically to match PIV orientation
plt.pcolormesh(data[0, :, :])
plt.show()
# np.save("E:/sPIV_PLIF_ProcessedData/PLIF/baseline_PLIF_flipped.npy", data)


