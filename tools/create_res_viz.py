# plot velocity and concentration fields for zoomed in subset of single frame of data

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
# from src.sPIV_PLIF_postprocessing.io import load_frame
import cmasher as cmr

################## USER INPUTS ###################

# PIV read paths
u_path = "E:/sPIV_PLIF_ProcessedData/PIV/Native_grid/piv_fractal_u.npz"
v_path = "E:/sPIV_PLIF_ProcessedData/PIV/Native_grid/piv_fractal_v.npz"
piv_x_path = "E:/sPIV_PLIF_ProcessedData/PIV/Native_grid/piv_fractal_xgrid_stack.npz"
piv_y_path = "E:/sPIV_PLIF_ProcessedData/PIV/Native_grid/piv_fractal_ygrid_stack.npz"

# PLIF read paths
c_path = "E:/sPIV_PLIF_ProcessedData/PLIF/fractal_PLIF.npy"
plif_x_path = "E:/sPIV_PLIF_ProcessedData/PLIF/fractal_xgrid.npy"
plif_y_path = "E:/sPIV_PLIF_ProcessedData/PLIF/fractal_ygrid.npy"

# time and space subsets
frame_no = 175
ylim = [88, 118]  # limits in mm
xlim = [-7, 23]  # limits in mm
ylim = [117, 137]  # limits in mm
xlim = [-66, -46]  # limits in mm
# ylim = [0, 300]
# xlim = [-100, 100]

# write paths
plif_save_path = "E:/sPIV_PLIF_ProcessedData/Plots/Instantaneous/resolution/plif_fractal_" + str(frame_no) + "AltLoc.png"
piv_save_path = "E:/sPIV_PLIF_ProcessedData/Plots/Instantaneous/resolution/piv_fractal_" + str(frame_no) + "AltLoc.png"

# plot settings
plif_cmap = cmr.rainforest_r
# plif_cmap = cmr.get_sub_cmap(plif_cmap, 0, 0.65)
plif_vmin = 0.00
plif_vmax = 0.20

vec_stride = 1 # SET TO 1 FOR FINAL RESOLUTION PLOT!!!!
# piv_tailwidth = 0.0025
# piv_scale = 10
# piv_headwidth = 3.75
# piv_headaxislength = 3.5
# piv_headlength = 5.0
piv_tailwidth = 0.003
piv_scale = 6.5
piv_headwidth = 4.5
piv_headaxislength = 3.25
piv_headlength = 3.75
piv_color = '#000000'


##################################################

### PLIF spatial resolution plot ###

# load data
c = np.load(c_path, mmap_mode="r")
c = np.array(c[frame_no, :, :], copy=True)
plif_x = np.load(plif_x_path)
plif_x = plif_x[0, :]
plif_y = np.load(plif_y_path)
plif_y = plif_y[:, 0]

# data subsetting for zoomed in plot
mask_x = (plif_x >= xlim[0]) & (plif_x <= xlim[1])
mask_y = (plif_y >= ylim[0]) & (plif_y <= ylim[1])
print(mask_x.shape)
print(mask_y.shape)

# Apply masks to select the zoomed-in region
y_idx = np.where(mask_y)[0]
x_idx = np.where(mask_x)[0]
c = c[np.ix_(y_idx, x_idx)]
plif_x = plif_x[x_idx]
plif_y = plif_y[y_idx]

# plotting
fig, ax = plt.subplots(figsize=(8, 6))
plt.pcolormesh(plif_x, plif_y, c, cmap=plif_cmap, norm=colors.Normalize(vmin=plif_vmin, vmax=plif_vmax), shading="gouraud")
plt.colorbar(label="Relative Concentration")
plt.xlim(xlim[0], xlim[1])
plt.ylim(ylim[0], ylim[1])
ax.set_aspect('equal', adjustable='box')
plt.savefig(plif_save_path, dpi=600)
plt.show()

### PIV spatial resolution plot ###

# load data
v = np.load(v_path, mmap_mode="r")
v = np.array(v['arr_' + str(frame_no)], copy=True)
u = np.load(u_path, mmap_mode="r")
u = np.array(u['arr_' + str(frame_no)], copy=True)
piv_x = np.load(piv_x_path, mmap_mode="r")
piv_x = np.array(piv_x['arr_' + str(frame_no)], copy=True)
piv_y = np.load(piv_y_path, mmap_mode="r")
piv_y = np.array(piv_y['arr_' + str(frame_no)], copy=True)
piv_y += 150

# data subsetting for zoomed in plot
mask_x = (piv_x[:] >= xlim[0]) & (piv_x[:] <= xlim[1])
mask_y = (piv_y[:] >= ylim[0]) & (piv_y[:] <= ylim[1])
print(mask_x.shape)     
print(mask_y.shape)
# Apply masks to select the zoomed-in region
y_idx = np.where(mask_y)[0]
x_idx = np.where(mask_x)[0]
u = u[np.ix_(y_idx, x_idx)]
v = v[np.ix_(y_idx, x_idx)]
piv_x = piv_x[x_idx]
piv_y = piv_y[y_idx]

# sample for vector stride
u = u[::vec_stride, ::vec_stride]
v = v[::vec_stride, ::vec_stride]
piv_x = piv_x[::vec_stride]
piv_y = piv_y[::vec_stride]

# plotting
fig, ax = plt.subplots(figsize=(8, 6))
# plt.pcolormesh(plif_x, plif_y, c, cmap=plif_cmap, norm=colors.Normalize(vmin=plif_vmin, vmax=plif_vmax), shading="gouraud")
# plt.colorbar(label="Relative Concentration")
plt.quiver(piv_x, piv_y, u, v, scale=piv_scale, width=piv_tailwidth, headwidth=piv_headwidth, headaxislength=piv_headaxislength, headlength=piv_headlength, color=piv_color)
ax.set_aspect('equal', adjustable='box')
plt.xlim(xlim[0], xlim[1])
plt.ylim(ylim[0], ylim[1])
plt.savefig(piv_save_path, dpi=600)
plt.show()


# # Plot PLIF with  white to jet cmap for comparison
# CMAP_NAME = "jet"  # jet for concentration
# CMAP_SLICE = (0.0, 1.0)
# C_UNDER: str | None = "white"  # fade in from white
# C_UNDER_TRANSITION: float | None = 0.1  # fraction of cmap for white->jet blend

# cmap = plt.get_cmap(CMAP_NAME)
# cmap = cmap.copy()
# cmap.set_under(C_UNDER)
# transition_fraction = C_UNDER_TRANSITION
# color_list = cmap(np.linspace(0, 1, 256))
# n_under = max(2, int(len(color_list) * min(transition_fraction, 0.5)))
# white = np.array([1.0, 1.0, 1.0, 1.0])
# first_color = color_list[0]
# under_grad = np.stack(
#     [
#         white * (1 - t) + first_color * t
#         for t in np.linspace(0, 1, n_under, endpoint=True)
#     ],
#     axis=0,
# )
# color_list = np.vstack([under_grad, color_list])
# cmap = colors.ListedColormap(color_list)

# vmin=0.006
# vmax=1

# fig, ax = plt.subplots(figsize=(8, 6))
# plt.pcolormesh(plif_x, plif_y, c, cmap=cmap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading="gouraud")
# plt.colorbar(label="Relative Concentration")
# plt.xlim(xlim[0], xlim[1])
# plt.ylim(ylim[0], ylim[1])
# ax.set_aspect('equal', adjustable='box')
# plt.savefig("E:/sPIV_PLIF_ProcessedData/Plots/Instantaneous/resolution/plif_fractal_" + str(frame_no) + "LogJet.png", dpi=600)
# plt.show()




########### VERSION FOR DIRECT READING OF IM7 AND VC7 FILES #############
########### ALSO SUBVERSION FOR MOXLIF DATA #############

# im7_path = "D:\SimulData_20Hz\PLIF_proc_8.29.2025_Buoyancy\data_8.29_30cms_FractalTG15cm_He0.897_air0.917_PIV0.02_iso_L3\Subtract_bgL4\divide_ffmbg_L4\Divide_C0_instantaneous\AddCameraAttributes\calibrated_L4\Resize\Median filter_01.set"
# # save_name = "I:/MOXLIF_processing/results/r60_uppersensor_ts_25to35sec.png"
# vc7_path = "D:/SimulData_20Hz/8.29.2025_20Hz_BuoyancyEffects_L2/8.29_30cmsPWM2.25_FractalTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_L2/StereoPIV_MPd(3x12x12_75%ov).set"
# # start_frame_no = int(500)  # start time = 25 sec
# # end_frame_no = int(701)  # end time = 35 sec
# frame_no = int(107)
# # sensor_xbounds = [494, 506]  # limits for upper sensor based on FF image
# # sensor_ybounds = [84, 102]  # limits for upper sensor based on FF image

# #cmap = "turbo"
# cmap = cmr.cosmic
# vec_stride = 2
# vec_stride_large = 18
# arrow_color = cmr.chroma
# xlim = [-20, 0]
# ylim = [5, 25]

# # Velocity magnitude color scaling
# vmin_vel = 0.0
# vmax_vel = 0.5
# cmap_vel = cmr.chroma_r
# norm_vel = colors.Normalize(vmin=vmin_vel, vmax=vmax_vel)

# ################ END USER INPUTS #################

# ########### Line plot of area subset #############
# # initialize time series array
# # avg_sensor_ts = np.zeros(end_frame_no-start_frame_no)

# # for i in range(end_frame_no - start_frame_no):
# #     c, plif_x, plif_y = load_frame.load_im7_frame_as_npy(im7_path, start_frame_no + i)

# #     # first frame: set up arrays and QC sensor coordinates 
# #     if i==20:
# #         sensor_mask = np.zeros_like(c, dtype=bool)
# #         sensor_mask[sensor_xbounds[0]:sensor_xbounds[1], sensor_ybounds[0]:sensor_ybounds[1]] = True
# #         overlay = np.zeros((c.shape[0], c.shape[1], 4))  # RGBA format
# #         overlay[sensor_mask] = [1, 0, 0, 0.5]  # Red with 50% transparency
# #         vmin = 100
# #         vmax = 2000
# #         plt.imshow(c, cmap='turbo', vmin=vmin, vmax=vmax)
# #         plt.colorbar()
# #         plt.imshow(overlay)
# #         plt.show()

# #     # average (or otherwise summarize) data over sensing area and comparison area
# #     avg_sensor_val = np.mean(c[sensor_xbounds[0]:sensor_xbounds[1], sensor_ybounds[0]:sensor_ybounds[1]])
# #     avg_sensor_ts[i] = avg_sensor_val

# # fig, ax = plt.subplots(figsize=(14, 4))
# # plt.plot(avg_sensor_ts)
# # plt.savefig(save_name, dpi=300)
# # plt.show()


# u, v, piv_x, piv_y = load_frame.load_vc7_frame_as_npy(vc7_path, frame_no)
# c, plif_x, plif_y = load_frame.load_im7_frame_as_npy(im7_path, frame_no)

# piv_x_grid, piv_y_grid = np.meshgrid(piv_x, piv_y)
# plif_x_grid, plif_y_grid = np.meshgrid(plif_x, plif_y)

# u_mag = np.sqrt(u**2+v**2)

# fig, ax = plt.subplots()
# plt.pcolormesh(plif_x_grid, plif_y_grid, c, cmap=cmap, norm=colors.LogNorm(vmin=0.012, vmax=1), shading="gouraud")
# plt.colorbar()

# plt.quiver(piv_x_grid[::vec_stride, ::vec_stride], piv_y_grid[::vec_stride, ::vec_stride], u[::vec_stride, ::vec_stride], v[::vec_stride, ::vec_stride], u_mag[::vec_stride, ::vec_stride], scale=13,width=0.003, norm = norm_vel, cmap=arrow_color)
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlim(xlim[0], xlim[1])
# ax.set_ylim(ylim[0], ylim[1])
# plt.savefig("C:/Users/Lavision/Documents/sPIV_PLIF_processing/res_plot_update.png", dpi=600)
# plt.show()

# fig, ax = plt.subplots()
# plt.pcolormesh(plif_x_grid, plif_y_grid, c, cmap=cmap, norm=colors.LogNorm(vmin=0.012, vmax=1), shading="gouraud")
# plt.quiver(piv_x_grid[::vec_stride_large, ::vec_stride_large], piv_y_grid[::vec_stride_large, ::vec_stride_large], u[::vec_stride_large, ::vec_stride_large], v[::vec_stride_large, ::vec_stride_large], u_mag[::vec_stride_large, ::vec_stride_large], scale=11, headlength=3.75, headaxislength=3, width=0.004, norm=norm_vel, cmap=arrow_color)
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlim(-100, 100)
# plt.savefig("C:/Users/Lavision/Documents/sPIV_PLIF_processing/full_plot_216_fractal.png", dpi=600)
# plt.show()
