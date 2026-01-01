import numpy as np
import matplotlib.pyplot as plt

xstart = 100
xend = 400
ystart = 0
yend = 600
tstart = 0
tend = 100

case_name = 'baseline'
u = np.load(f'E:/sPIV_PLIF_ProcessedData/PIV/piv_{case_name}_u.npy')[xstart:xend, ystart:yend, tstart:tend]
v = np.load(f'E:/sPIV_PLIF_ProcessedData/PIV/piv_{case_name}_v.npy')[xstart:xend, ystart:yend, tstart:tend]
w = np.load(f'E:/sPIV_PLIF_ProcessedData/PIV/piv_{case_name}_w.npy')[xstart:xend, ystart:yend, tstart:tend]

u_mean = np.load(f'E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_{case_name}.npz')['u']
v_mean = np.load(f'E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_{case_name}.npz')['v']
w_mean = np.load(f'E:/sPIV_PLIF_ProcessedData/mean_fields/mean_fields_{case_name}.npz')['w']

# Decompose velocity fields into mean and fluctuating components of u and v
u_flx = u - u_mean
v_flx = v - v_mean
w_flx = w - w_mean

np.save('E:/sPIV_PLIF_ProcessedData/flow_properties/flx_u_v_w/u_flx.npy', u_flx)
np.save('E:/sPIV_PLIF_ProcessedData/flow_properties/flx_u_v_w/v_flx.npy', v_flx)
np.save('E:/sPIV_PLIF_ProcessedData/flow_properties/flx_u_v_w/w_flx.npy', w_flx)

# Compute fluctuating strain rates 
dx = 0.0005  # m
dt = 0.02  # sec
duflx = np.gradient(u_flx, dx, dx, dt)
duflx = np.asarray(duflx)
duflx_dx = duflx[0, :, :, :]
duflx_dy = duflx[1, :, :, :]
duflx_dw = duflx[2, :, :, :]
np.save(f'E:/sPIV_PLIF_ProcessedData/flow_properties/flx_StrainRates/duflx_{case_name}.npy', duflx)
dvflx = np.gradient(v_flx, dx, dx, dt)
dvflx_dx = np.asarray(dvflx[0, :, :, :])
dvflx_dy = np.asarray(dvflx[1, :, :, :])
dvflx_dw = np.asarray(dvflx[2, :, :, :])
np.save(f'E:/sPIV_PLIF_ProcessedData/flow_properties/flx_StrainRates/dvflx_{case_name}.npy', dvflx)
dwflx = np.gradient(w_flx, dx, dx, dt)
dwflx_dw = np.asarray(dwflx[2, :, :, :])
np.save(f'E:/sPIV_PLIF_ProcessedData/flow_properties/flx_StrainRates/dwflx_{case_name}.npy', dwflx)




# duflx_dy = np.load('ignore/duflx_dyv2.npy')
# dvflx_dx = np.load('ignore/dvflx_dxv2.npy')

nu = 1.5 * 10**(-5) # kinematic viscosity 
# Compute viscous energy dissipation rate
epsilon = 2 * nu * np.mean((duflx_dx**2 + dvflx_dy**2 + dwflx_dw**2 + 2 * duflx_dy**2 + 2 *dvflx_dx**2 + 2 * dvflx_dw**2), axis=2)
print(f'avg viscous dissipation: {np.mean(epsilon)}')
print(f'avg viscous dissipation, x=[0, 0.05] and y=[-0.2, 0.2]: {np.mean(epsilon[200:1000, 0:100])}')
print(f'avg viscous dissipation, x=[0.7, 0.75] and y=[-0.2, 0.2]: {np.mean(epsilon[200:1000, 1400:1500])}')
plt.close()
plt.pcolormesh(epsilon)
plt.colorbar()
plt.show()

# Compute Taylor microscale and Taylor Reynolds number
avg_rms = np.sqrt((1/3)*(np.mean(u_flx**2, axis=2) + np.mean(v_flx**2, axis=2) + np.mean(w_flx**2, axis=2)))
Taylor_microscale = np.sqrt(15 * nu / epsilon) * avg_rms  # homogeneous isotropic turbulence assumption
kolmogorov_length_scale = (nu**3 / epsilon)**0.25
kolmogorov_time_scale = (nu / epsilon)**0.5
Taylor_Re = avg_rms * Taylor_microscale / nu
np.save(f'E:/sPIV_PLIF_ProcessedData/flow_properties/Taylor_microscale_{case_name}.npy', Taylor_microscale)
np.save(f'E:/sPIV_PLIF_ProcessedData/flow_properties/Taylor_Re_{case_name}.npy', Taylor_Re)
np.save(f'E:/sPIV_PLIF_ProcessedData/flow_properties/kolmogorov_length_scale_{case_name}.npy', kolmogorov_length_scale)
np.save(f'E:/sPIV_PLIF_ProcessedData/flow_properties/kolmogorov_time_scale_{case_name}.npy', kolmogorov_time_scale)
# PLOT: Taylor microscale, computed per Carbone & Wilczek, 2024 (https://doi.org/10.1017/jfm.2024.165)
print(f'avg Taylor microscale: {np.mean(Taylor_microscale)}')
print(f'avg Taylor microscale from x=[0, 0.05] and y=[-0.15, 0.15]: {np.mean(Taylor_microscale[123:723, 0:100])}')
print(f'avg Taylor microscale from x=[0, 0.05] and y=[-0.15, 0.15]: {np.mean(Taylor_microscale[123:723, 1400:1500])}')
plt.close()
plt.pcolormesh(Taylor_microscale)
plt.colorbar()
plt.savefig(f'E:/sPIV_PLIF_ProcessedData/flow_properties/Taylor_microscale_{case_name}.png', dpi=600)
plt.show()

# PLOT: Taylor Reynolds number using methods from Carbone & Wilczek, 2024 (https://doi.org/10.1017/jfm.2024.165)
print(f'avg Taylor Re: {np.mean(Taylor_Re)}')
print(f'avg Taylor Re from x=[0, 0.05] and y=[-0.15, 0.15]: {np.mean(Taylor_Re[123:723, 0:100])}')
print(f'avg Taylor Re from x=[0, 0.05] and y=[-0.15, 0.15]: {np.mean(Taylor_Re[123:723, 1400:1500])}')
plt.close()
plt.pcolormesh(Taylor_Re)
plt.colorbar()
plt.savefig(f'E:/sPIV_PLIF_ProcessedData/flow_properties/Taylor_Re_{case_name}.png', dpi=600)
plt.show()

# Compute turbulent kinetic energy
tke = 0.5 * (np.mean(u_flx**2, axis=2) + np.mean(v_flx**2, axis=2) + np.mean(w_flx**2, axis=2))
np.save(f'E:/sPIV_PLIF_ProcessedData/flow_properties/tke_{case_name}.npy', tke)
u_rms = np.load(f'E:/sPIV_PLIF_ProcessedData/rms_fields/{case_name}_u_rms.npy')
v_rms = np.load(f'E:/sPIV_PLIF_ProcessedData/rms_fields/{case_name}_v_rms.npy')
w_rms = np.load(f'E:/sPIV_PLIF_ProcessedData/rms_fields/{case_name}_w_rms.npy')
t_intensity_avg = np.sqrt((1/3) * (u_rms + v_rms + w_rms)) / u_mean
np.save(f'E:/sPIV_PLIF_ProcessedData/flow_properties/turbulence_intensity_{case_name}.npy', t_intensity_avg)

# PLOT: turbulent intensity and turbulent kinetic energy
cmap = cmr.ember
utils.plot_field_xy(x_grid, y_grid, tke, title='turbulent kinetic energy', cmap=cmap, filepath=f'E:/sPIV_PLIF_ProcessedData/flow_properties/tke_{case_name}.png', dpi=600, trimmed=False)
utils.plot_field_xy(x_grid, y_grid, t_intensity_avg, title='turbulence intensity', cmap=cmap, range=[0, 0.8], filepath=f'E:/sPIV_PLIF_ProcessedData/flow_properties/t_intensity_{case_name}.png', dpi=600, trimmed=True)