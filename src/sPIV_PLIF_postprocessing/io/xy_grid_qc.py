import numpy as np

xvec = np.load('E:/sPIV_PLIF_ProcessedData/x_coords.npy')
yvec = np.load('E:/sPIV_PLIF_ProcessedData/y_coords.npy')

# yvec = np.float32(yvec)
# np.save('E:/sPIV_PLIF_ProcessedData/y_coords.npy', yvec)

xgrid, ygrid = np.meshgrid(xvec, yvec, indexing='xy')

print(f'x grid shape: {xgrid.shape}')
print(f'y grid shape: {ygrid.shape}')   

print(f'x grid min/max: {np.min(xgrid)}/{np.max(xgrid)}')
print(f'y grid min/max: {np.min(ygrid)}/{np.max(ygrid)}')
print(f'x grid sample: {xgrid[0,0:5]}')
print(f'y grid sample: {ygrid[0:5,0]}')

print(f"x vec datatype: {xvec.dtype}")
print(f"y vec datatype: {yvec.dtype}")