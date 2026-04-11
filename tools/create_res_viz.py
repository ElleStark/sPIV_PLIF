# plot velocity and concentration fields for zoomed in subset of single frame of data

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from src.sPIV_PLIF_postprocessing.io import load_frame
import cmasher as cmr

################## USER INPUTS ###################

im7_path = "D:\SimulData_20Hz\PLIF_proc_8.29.2025_Buoyancy\data_8.29_30cms_FractalTG15cm_He0.897_air0.917_PIV0.02_iso_L3\Subtract_bgL4\divide_ffmbg_L4\Divide_C0_instantaneous\AddCameraAttributes\calibrated_L4\Resize\Median filter_01.set"
# save_name = "I:/MOXLIF_processing/results/r60_uppersensor_ts_25to35sec.png"
vc7_path = "D:/SimulData_20Hz/8.29.2025_20Hz_BuoyancyEffects_L2/8.29_30cmsPWM2.25_FractalTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_L2/StereoPIV_MPd(3x12x12_75%ov).set"
# start_frame_no = int(500)  # start time = 25 sec
# end_frame_no = int(701)  # end time = 35 sec
frame_no = int(107)
sensor_xbounds = [494, 506]  # limits for upper sensor based on FF image
sensor_ybounds = [84, 102]  # limits for upper sensor based on FF image

#cmap = "turbo"
cmap = cmr.cosmic
vec_stride = 2
vec_stride_large = 18
arrow_color = cmr.chroma
xlim = [-20, 0]
ylim = [5, 25]

# Velocity magnitude color scaling
vmin_vel = 0.0
vmax_vel = 0.5
cmap_vel = cmr.chroma_r
norm_vel = colors.Normalize(vmin=vmin_vel, vmax=vmax_vel)

################ END USER INPUTS #################

########### Line plot of area subset #############
# initialize time series array
# avg_sensor_ts = np.zeros(end_frame_no-start_frame_no)

# for i in range(end_frame_no - start_frame_no):
#     c, plif_x, plif_y = load_frame.load_im7_frame_as_npy(im7_path, start_frame_no + i)

#     # first frame: set up arrays and QC sensor coordinates 
#     if i==20:
#         sensor_mask = np.zeros_like(c, dtype=bool)
#         sensor_mask[sensor_xbounds[0]:sensor_xbounds[1], sensor_ybounds[0]:sensor_ybounds[1]] = True
#         overlay = np.zeros((c.shape[0], c.shape[1], 4))  # RGBA format
#         overlay[sensor_mask] = [1, 0, 0, 0.5]  # Red with 50% transparency
#         vmin = 100
#         vmax = 2000
#         plt.imshow(c, cmap='turbo', vmin=vmin, vmax=vmax)
#         plt.colorbar()
#         plt.imshow(overlay)
#         plt.show()

#     # average (or otherwise summarize) data over sensing area and comparison area
#     avg_sensor_val = np.mean(c[sensor_xbounds[0]:sensor_xbounds[1], sensor_ybounds[0]:sensor_ybounds[1]])
#     avg_sensor_ts[i] = avg_sensor_val

# fig, ax = plt.subplots(figsize=(14, 4))
# plt.plot(avg_sensor_ts)
# plt.savefig(save_name, dpi=300)
# plt.show()


u, v, piv_x, piv_y = load_frame.load_vc7_frame_as_npy(vc7_path, frame_no)
c, plif_x, plif_y = load_frame.load_im7_frame_as_npy(im7_path, frame_no)

piv_x_grid, piv_y_grid = np.meshgrid(piv_x, piv_y)
plif_x_grid, plif_y_grid = np.meshgrid(plif_x, plif_y)

u_mag = np.sqrt(u**2+v**2)

fig, ax = plt.subplots()
plt.pcolormesh(plif_x_grid, plif_y_grid, c, cmap=cmap, norm=colors.LogNorm(vmin=0.012, vmax=1), shading="gouraud")
plt.colorbar()

plt.quiver(piv_x_grid[::vec_stride, ::vec_stride], piv_y_grid[::vec_stride, ::vec_stride], u[::vec_stride, ::vec_stride], v[::vec_stride, ::vec_stride], u_mag[::vec_stride, ::vec_stride], scale=13,width=0.003, norm = norm_vel, cmap=arrow_color)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(xlim[0], xlim[1])
ax.set_ylim(ylim[0], ylim[1])
plt.savefig("C:/Users/Lavision/Documents/sPIV_PLIF_processing/res_plot_update.png", dpi=600)
plt.show()

fig, ax = plt.subplots()
plt.pcolormesh(plif_x_grid, plif_y_grid, c, cmap=cmap, norm=colors.LogNorm(vmin=0.012, vmax=1), shading="gouraud")
plt.quiver(piv_x_grid[::vec_stride_large, ::vec_stride_large], piv_y_grid[::vec_stride_large, ::vec_stride_large], u[::vec_stride_large, ::vec_stride_large], v[::vec_stride_large, ::vec_stride_large], u_mag[::vec_stride_large, ::vec_stride_large], scale=11, headlength=3.75, headaxislength=3, width=0.004, norm=norm_vel, cmap=arrow_color)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-100, 100)
plt.savefig("C:/Users/Lavision/Documents/sPIV_PLIF_processing/full_plot_216_fractal.png", dpi=600)
plt.show()
