""" read in original head-specific .vc7 files from DaVis and combine into single .npy ndarray, 
    along with x and y coordinate arrays. Save as .npy file for later use"""

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib import rcParams
import logging
import lvpyio as lv  # LaVision's package for importing .imx and .vc7 files to Python
# import pivpy as pp
# from pivpy import io
import scipy.interpolate as interp
# import xarray as xr
import os

save_dir = 'E:/sPIV_PLIF_processedData/'
save_name = '8.29_30cmsPWM2.25_smTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_'
piv_dir = 'D:/PIV_20Hz_data/'

piv_path1 = piv_dir + '8.29.2025_20Hz_BuoyancyEffects_L1/8.29_30cmsPWM2.25_smTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso\CopySelected_L1\StereoPIV_MPd(2x12x12_75%ov)_01.set'
piv_path2 = piv_dir + '8.29.2025_20Hz_BuoyancyEffects_L2/8.29_30cmsPWM2.25_smTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso\CopySelected_L2\StereoPIV_MPd(2x12x12_75%ov).set'

# read first frame to define grid in real-world coordinates
vec_set1 = lv.read_set(piv_path1)
vec_set2 = lv.read_set(piv_path2)
n_frames = len(vec_set1) + len(vec_set2)

vec_firstbuffer1 = vec_set1[0]
vec_firstframe1 = vec_firstbuffer1[0].as_masked_array()
vec_firstbuffer2 = vec_set2[0]
vec_firstframe2 = vec_firstbuffer2[0].as_masked_array()

# extract x, y grid data
x, y = vec_firstbuffer1[0].grid.x, vec_firstbuffer1[0].grid.y
# print(f'x grid shape: {x.shape}, y grid shape: {y.shape}')
print(f'x grid {x}; y grid {y}')


# get scales and offset to compute real-world coordinates
# check scales first
x_scale = vec_firstbuffer1[0].scales.x
y_scale = vec_firstbuffer1[0].scales.y
print(f'FIRST vector set:      x scale: slope={x_scale.slope}, offset={x_scale.offset}, unit={x_scale.unit}'
      f'; y scale: slope={y_scale.slope}, offset={y_scale.offset}, unit={y_scale.unit}')
x_scale2 = vec_firstbuffer2[0].scales.x
y_scale2 = vec_firstbuffer2[0].scales.y
print(f'SECOND vector set:     x scale: slope={x_scale2.slope}, offset={x_scale2.offset}, unit={x_scale2.unit}'
      f'; y scale: slope={y_scale2.slope}, offset={y_scale2.offset}, unit={y_scale2.unit}')
# looks like y dims are 2 px off, x scale 1 px off between two heads; assume first head is correct

u1 = vec_firstframe1['u']
u2 = vec_firstframe2['u']

# initialize arrays for storing collated data
all_u = np.zeros((u1.shape[0], u2.shape[1], n_frames), dtype=np.float32)
all_v = np.zeros((u1.shape[0], u2.shape[1], n_frames), dtype=np.float32)
all_w = np.zeros((u1.shape[0], u2.shape[1], n_frames), dtype=np.float32)

# loop over all frames to collate data
for i in range(n_frames):
    if i%2==0:  # even frames from first head
        vec_data = vec_set1[int(i/2)]
        vec_data = vec_data[0].as_masked_array()
        all_u[:, :, i] = vec_data['u'][:, :-1]
        all_v[:, :, i] = vec_data['v'][:, :-1]
        all_w[:, :, i] = vec_data['w'][:, :-1]
    else:  # odd frames from second head
        vec_data = vec_set2[int((i-1)/2)]
        vec_data = vec_data[0].as_masked_array()
        all_u[:, :, i] = vec_data['u'][:-2, :]
        all_v[:, :, i] = vec_data['v'][:-2, :]
        all_w[:, :, i] = vec_data['w'][:-2, :]


np.save(save_dir + save_name +'u.npy', all_u)
np.save(save_dir + save_name +'v.npy', all_v)
np.save(save_dir + save_name +'w.npy', all_w)

