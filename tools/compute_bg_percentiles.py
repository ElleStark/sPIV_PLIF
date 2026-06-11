"""
read in PLIF data and compute distribution of background intensities. 
Elle Stark June 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cmasher as cmr


####### USER INPUTS ########
CASE_NAME = "nearbed"

c_path = Path(f"E:/sPIV_PLIF_ProcessedData/PLIF/{CASE_NAME}_PLIF.npy")
xpath = Path(f"E:/sPIV_PLIF_ProcessedData/PLIF/{CASE_NAME}_xgrid.npy")
ypath = Path(f"E:/sPIV_PLIF_ProcessedData/PLIF/{CASE_NAME}_ygrid.npy")

save_path_overlay = Path(f"E:/sPIV_PLIF_ProcessedData/Plots/Noise_plots/{CASE_NAME}_noiseAreas.png")
save_path_hist = Path(f"E:/sPIV_PLIF_ProcessedData/Plots/Noise_plots/{CASE_NAME}_noiseHist.png")

xlim = [-100, -80]  # background area limits in mm
xlim2 = [80, 100]  # background area limits in mm
ylim = [0, 300]  # background area limits in mm
tlim = [0, 6000]  # frame limits for background analysis

#####################################

# load data
c = np.load(c_path, mmap_mode="r")
c_sub = np.array(c[tlim[0]:tlim[1], :, :], copy=True)  # take frame subset for background analysis
x = np.load(xpath)
x = x[0, :]
y = np.load(ypath)
y = y[:, 0]
# subset to background area
x_mask = (x >= xlim[0]) & (x <= xlim[1])
x_mask2 = (x >= xlim2[0]) & (x <= xlim2[1])
y_mask = (y >= ylim[0]) & (y <= ylim[1])

# y_idx = np.where(y_mask)[0]
# print("y_idx:", y_idx.shape)
# x_idx = np.where(x_mask)[0]
# print("x_idx:", x_idx.shape)
# # c = c[:, np.ix_(y_idx, x_idx)]
# plif_x = x[x_idx]
# plif_y = y[y_idx]
# bg_c = c_sub[:, y_idx, x_idx]

bg_c = c_sub[:, y_mask, :][:, :, x_mask]
bg_c2 = c_sub[:, y_mask, :][:, :, x_mask2]
# combine values and compute percentiles
bg_values = np.concatenate([bg_c.ravel(), bg_c2.ravel()])
percentiles = np.percentile(bg_values, [0.1, 5, 25, 50, 90, 95, 99, 99.9, 100])
percentiles = np.round(percentiles, 4)
rms_noise = np.sqrt(np.mean(bg_values**2))
mean_noise = np.mean(bg_values)
var_noise = np.var(bg_values)
std_noise = np.std(bg_values)
print(f"noise percentiles: 0.1% {percentiles[0]}, 5% {percentiles[1]}, 25% {percentiles[2]}, 50% {percentiles[3]}, 90% {percentiles[4]}, 95% {percentiles[5]}, 99% {percentiles[6]}, 99.9% {percentiles[7]}, 100% {percentiles[8]}")
print(percentiles)
print(f"\n RMS Noise: {rms_noise}")
print(f"\n Mean Noise: {mean_noise}")
print(f"\n Variance Noise: {var_noise}")
print(f"\n Standard Deviation Noise: {std_noise}")

# plot bg area over PLIF image
fig, ax = plt.subplots(figsize=(6, 8))
plt.pcolormesh(x, y, c[0], shading='auto', cmap=cmr.rainforest_r, norm=plt.Normalize(vmin=0, vmax=1))
plt.colorbar(label='PLIF Intensity')
plt.vlines(xlim, ylim[0], ylim[1], color='red', linestyle='--', linewidth=3, label='Background Area')
plt.vlines(xlim2, ylim[0], ylim[1], color='red', linestyle='--', linewidth=3, label='Background Area 2')
plt.hlines(ylim, xlim[0], xlim[1], color='red', linestyle='--', linewidth=3)
plt.hlines(ylim, xlim2[0], xlim2[1], color='red', linestyle='--', linewidth=3)
plt.xlim(-101, 101)
plt.ylim(0, 300)
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title('PLIF image with non-plume areas')
ax.set_aspect('equal', adjustable='box')
plt.savefig(save_path_overlay, dpi=600)
plt.show()

# plot histogram of bg values
plt.figure(figsize=(10, 6))
bins = [-0.001, -0.0005, 0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01, 0.0105, 0.011, 0.0115, 0.012, 0.0125, 0.013, 0.0135, 0.014, 0.0145, 0.015, 0.0155, 0.016, 0.0165, 0.017, 0.0175, 0.018, 0.0185, 0.019, 0.0195, 0.02, 0.0205, 0.021, 0.0215, 0.022, 0.0225, 0.023, 0.0235, 0.024, 0.0245, 0.025]
plt.hist(bg_values, bins=bins, color='#ECECEC', edgecolor='#3f3f3f', alpha=0.7, density=True)
plt.axvline(percentiles[1], color='red', linestyle='--', linewidth=3, label=f'5th Percentile: {percentiles[1]}')
# plt.axvline(rms_noise, color='orange', linestyle='--', linewidth=3, label=f'RMS Noise: {rms_noise:.3f}')
plt.axvline(percentiles[3], color='orange', linestyle='--', linewidth=3, label=f'50th Percentile: {round(percentiles[3], 3)}')
# plt.axvline(percentiles[3], color='purple', linestyle='--', label=f'90th Percentile: {percentiles[3]}')
plt.axvline(percentiles[5], color='blue', linestyle='--', linewidth=3, label=f'95th Percentile: {round(percentiles[5], 3)}')
# plt.axvline(percentiles[6], color='blue', linestyle='--', label=f'99th Percentile: {round(percentiles[6], 3)}')
plt.xlim(bins[0], bins[-1])
plt.title('histogram of noise levels in non-plume areas')
plt.legend()
plt.xlabel('Concentration C/C0')
plt.ylabel('Frequency')
plt.savefig(save_path_hist, dpi=600)
plt.show()
