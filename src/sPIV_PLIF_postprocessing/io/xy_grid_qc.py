import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

y1 = np.load('E:/y_v1.npy')
x1 = np.load('E:/x_v1.npy')

xvec = np.load('E:/sPIV_PLIF_ProcessedData/x_coords.npy')
yvec = np.load('E:/sPIV_PLIF_ProcessedData/y_coords.npy')

# yvec = np.float32(yvec)
# np.save('E:/sPIV_PLIF_ProcessedData/y_coords.npy', yvec)

xgrid, ygrid = np.meshgrid(xvec, yvec, indexing='xy')
xgrid = np.flipud(xgrid)
ygrid = np.flipud(ygrid)


print(f'x grid shape: {xgrid.shape}')
print(f'y grid shape: {ygrid.shape}')   

print(f'x grid min/max: {np.min(xgrid)}/{np.max(xgrid)}')
print(f'xv1 grid min/max: {np.min(x1)}/{np.max(x1)}')
print(f'y grid min/max: {np.min(ygrid)}/{np.max(ygrid)}')
print(f'yv1 grid min/max: {np.min(y1)}/{np.max(y1)}')
print(f'x grid sample: {xgrid[0,0:5]}')
print(f'xv1 grid sample: {x1[0,0:5]}')
print(f'y grid sample: {ygrid[0:5,0]}')
print(f'yv1 grid sample: {y1[0:5,0]}')

print(f"x vec datatype: {xvec.dtype}")
print(f"y vec datatype: {yvec.dtype}")


ftle = np.load('data/FTLE_t50.0to50.05s_fractalCasev2.npy')
ftle = np.squeeze(ftle, axis=0)
ftle = np.rot90(ftle, k=2)
plt.figure()
levels = np.linspace(-5, np.max(ftle), 100)
ftle[ftle < -5] = -5
plt.contourf(xgrid, ygrid, ftle, levels=levels, cmap=cmr.amber)
plt.colorbar()  
plt.show()
