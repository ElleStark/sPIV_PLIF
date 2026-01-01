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
duflx_dy = np.gradient(u_flx, dt, dx, dx)
duflx_dy = np.asarray(duflx_dy)
duflx_dy = duflx_dy[1, :, :, :]
np.save(f'E:/sPIV_PLIF_ProcessedData/flow_properties/flx_StrainRates/duflx_dy_{case_name}.npy', duflx_dy)
dvflx_dx = np.gradient(v_flx, dt, dx, dx)
dvflx_dx = np.asarray(dvflx_dx)
dvflx_dx = dvflx_dx[2, :, :, :]
np.save(f'E:/sPIV_PLIF_ProcessedData/flow_properties/flx_StrainRates/dvflx_dx_{case_name}.npy', dvflx_dx)
# duflx_dy = np.load('ignore/duflx_dyv2.npy')
# dvflx_dx = np.load('ignore/dvflx_dxv2.npy')

nu = 1.5 * 10**(-5) # kinematic viscosity 
# Compute viscous energy dissipation rate
epsilon = np.mean((duflx_dy**2 + dvflx_dx**2), axis=0)
print(f'avg viscous dissipation: {np.mean(epsilon)}')
print(f'avg viscous dissipation, x=[0, 0.05] and y=[-0.2, 0.2]: {np.mean(epsilon[200:1000, 0:100])}')
print(f'avg viscous dissipation, x=[0.7, 0.75] and y=[-0.2, 0.2]: {np.mean(epsilon[200:1000, 1400:1500])}')
plt.close()
plt.pcolormesh(epsilon)
plt.colorbar()
plt.show()

Taylor_microscale = np.sqrt(15 / epsilon) * np.sqrt(0.5 * (np.mean(u_flx**2, axis=0) + np.mean(v_flx**2, axis=0)))
# Taylor_microscale = np.sqrt(10 / epsilon) * np.sqrt((np.mean(u_flx**2, axis=0)))





uprime = np.sqrt(0.5*(np.mean(u_flx**2, axis=0) + np.mean(v_flx**2, axis=0)))  # root mean square velocity
# K_tscale = 1 / np.sqrt(np.mean(np.sqrt((duflx_dy**2 + dvflx_dx**2))**2, axis=0))
# Taylor_microscale = np.sqrt(15)*uprime*K_tscale
Taylor_Re = uprime * Taylor_microscale / nu
np.save('ignore/extended_sim/Taylor_microscale_v1.npy', Taylor_microscale)
np.save('ignore/extended_sim/Taylor_Re_v1.npy', Taylor_Re)

# PLOT: Taylor microscale, computed per Carbone & Wilczek, 2024 (https://doi.org/10.1017/jfm.2024.165)
print(f'avg Taylor microscale: {np.mean(Taylor_microscale)}')
print(f'avg Taylor microscale from x=[0, 0.05] and y=[-0.15, 0.15]: {np.mean(Taylor_microscale[123:723, 0:100])}')
print(f'avg Taylor microscale from x=[0, 0.05] and y=[-0.15, 0.15]: {np.mean(Taylor_microscale[123:723, 1400:1500])}')
plt.close()
plt.pcolormesh(Taylor_microscale)
plt.colorbar()
plt.savefig('ignore/extended_sim/Taylor_microscale_v1.png', dpi=600)
plt.show()

# PLOT: Taylor Reynolds number using methods from Carbone & Wilczek, 2024 (https://doi.org/10.1017/jfm.2024.165)
print(f'avg Taylor Re: {np.mean(Taylor_Re)}')
print(f'avg Taylor Re from x=[0, 0.05] and y=[-0.15, 0.15]: {np.mean(Taylor_Re[123:723, 0:100])}')
print(f'avg Taylor Re from x=[0, 0.05] and y=[-0.15, 0.15]: {np.mean(Taylor_Re[123:723, 1400:1500])}')
plt.close()
plt.pcolormesh(Taylor_Re)
plt.colorbar()
plt.savefig('ignore/extended_sim/Taylor_Re_v1.png', dpi=600)
plt.show()








# Compute turbulent kinetic energy
tke = 0.5 * (np.mean(u_flx**2, axis=0) + np.mean(v_flx**2, axis=0))
np.save('ignore/extended_sim/tke_extendedsim.npy', tke)
# t_intensity = np.sqrt(tke) / np.sqrt(u_mean**2 + v_mean**2)

# PLOT: turbulent intensity and turbulent kinetic energy
cmap = cmr.ember
utils.plot_field_xy(x_grid, y_grid, tke, title='turbulent kinetic energy', cmap=cmap, filepath='ignore/extended_sim/tke_fullDomain.png', dpi=600, trimmed=False)
# utils.plot_field_xy(x_grid, y_grid, t_intensity, title='turbulence intensity', cmap=cmap, range=[0, 0.8], filepath='ignore/t_intensity_trimmed.png', dpi=600, trimmed=True)
