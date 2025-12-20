"""
Script to compute and plot flow field statistics, including mean velocities, turbulence intensity, and turbulence isotropy.
Separate script included in repository to compute integral length scales using the HPC resources (Blanca compute node at CU Boulder).
Elle Stark, May 2024
"""

import h5py
import matplotlib.pyplot as plt
from matplotlib import colors as colors
import numpy as np
# import utils
import cmasher as cmr
import scipy.io
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def main():
    # # Read in data from HDF5 file
    # filename = 'D:/singlesource_2d_extended/Restride0_0_5mm_50Hz_singlesource_2d.h5'
    # with h5py.File(filename, 'r') as f:
    #     # x and y grids for plotting
    #     x_grid = f.get(f'Model Metadata/xGrid')[:, :].T
    #     y_grid = f.get(f'Model Metadata/yGrid')[:, :].T

    #     # u and v velocity field data
    #     # u_flx = f.get('Flow Data/u')[1500:1502, :, :].transpose(0, 2, 1)  # original dimensions (time, x, y) = (9001, 1501, 1201)
    #     # v_flx = f.get('Flow Data/v')[1500:1502, :, :].transpose(0, 2, 1)

    #     # odor concentration field data
    #     # odor = f.get('Odor Data/c')[1500:1502, :, :].transpose(0, 2, 1)

    #     # Spatial resolution
    #     # dx = f.get('Model Metadata/spatialResolution')[0].item()
    #     dx = 0.0005  #m

    v_data = np.load('E:/sPIV_PLIF_processedData/8.29_30cmsPWM2.25_smTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_v.npy')[10:-10, 100:-100, :1000]
    # # v_data = np.flipud(v_data)
    # u_data = np.load('E:/sPIV_PLIF_processedData/8.29_30cmsPWM2.25_smTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_u.npy')[10:-10, 100:-100, :1000]
    # # u_data = np.flipud(u_data)
    # xg = np.load('E:/sPIV_PLIF_processedData/PIV/xgrid.npy')[10:-10, 100:-101]
    # yg = np.load('E:/sPIV_PLIF_processedData/PIV/ygrid.npy')[10:-10, 100:-101]
    # # PLOT: example instantaneous velocity field
    # cmap = cmr.fall_r
    # vmin = -0.7
    # vmax = 0
    # stride = 30

    # uq = u_data[:, :, 150]
    # uq = uq[::stride, ::stride]
    # vq = v_data[:, :, 150]
    # vq = vq[::stride, ::stride]
    # xq = xg[::stride, ::stride]
    # yq = yg[::stride, ::stride]
    
    # plt.figure()
    # plt.pcolor(xg, yg, np.fliplr(v_data[:, :, 150]), cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.colorbar()
    # plt.quiver(xq, yq, uq, vq, headwidth=3, headlength=4)
    # plt.title('instantaneous v velocity')
    # plt.savefig('E:/sPIV_PLIF_processedData/PIV/instant_v_example.png', dpi=600)
    # plt.show()
    # # utils.plot_field_xy(x_grid, y_grid, u_flx[0, :, :], cmap=cmap, range=[-0.25, 0.25], title='instantaneous velocity', arrows=True, u=u_flx[0, :, :], v=v_flx[0, :, :], filepath='ignore/extended_sim_u_instant.png', dpi=600)

    # mean_v = v_data.mean(axis=2)
    # mean_u = u_data.mean(axis=2)

    # vmq = mean_v[::stride, ::stride]
    # umq = mean_u[::stride, ::stride]

    # plt.figure()
    # plt.pcolor(xg, yg, np.fliplr(mean_v[:, :]), cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.colorbar()
    # plt.quiver(xq, yq, umq, vmq, headwidth=3, headlength=4)
    # plt.title('mean v velocity')
    # plt.savefig('E:/sPIV_PLIF_processedData/PIV/mean_v_example.png', dpi=600)
    # plt.show()

    rms_v = np.sqrt(np.mean(v_data**2, axis=2))

    cmap = cmr.sunburst
    plt.figure()
    plt.pcolor(np.flipud(rms_v[:, :]), cmap=cmap)
    plt.colorbar()
    plt.title('rms v velocity')
    plt.savefig('E:/sPIV_PLIF_processedData/PIV/rms_v_example.png', dpi=600)
    plt.show()



    # PLOT: example instantaneous concentration field
    
    # odor = np.load('E:/sPIV_PLIF_processedData/PLIF/PLIF_8.29_30cmsPWM2.25_smTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso.npy')[:, 100:-100, :1000]
    # odor = np.flipud(odor)
    # # utils.plot_field_xy(x_grid, y_grid, odor[0, :, :], cmap=cmap, range=[0.001, 1], title='instantaneous concentration', filepath='ignore/extended_sim/odor_instant.png', dpi=600)
    # # mean concentration field
    # # odor_mean = np.mean(odor, axis=2)
    # # utils.plot_field_xy(x_grid, y_grid, odor, cmap=cmap, range=[0.001, 1], title='time-avg concentration', filepath='ignore/extended_sim/odor_mean.png', dpi=600)
    # rms_odor = np.sqrt(np.mean(odor**2, axis=2))
    



    # cmap = cmr.rainforest
    # vmin = 0.02
    # vmax = 0.75
    # norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    
    # plt.figure()
    # plt.pcolor(rms_odor, cmap=cmap, norm=norm)
    # plt.colorbar()
    # plt.savefig('E:/sPIV_PLIF_processedData/PLIF/rms_odor_example.png', dpi=600)
    # plt.show()

    # plt.figure()
    # plot_odor = odor[:, :, 150]
    # plot_odor[plot_odor < vmin] = vmin
    # plt.pcolor(plot_odor, cmap=cmap, norm=norm)
    # plt.axis('equal')
    # plt.colorbar()
    # plt.savefig('E:/sPIV_PLIF_processedData/PLIF/instant_odor_example.png', dpi=600)
    # plt.show()

    # plt.figure()
    # plt.pcolor(odor_mean, cmap=cmap, norm=norm)
    # plt.colorbar()
    # plt.savefig('E:/sPIV_PLIF_processedData/PLIF/mean_odor_example.png', dpi=600)
    # plt.show()

    # # # PLOT: load and plot mean velocity field (consistent axes version)
    # u_mean = np.load('D:/singlesource_2d_extended/mean_u_0to180s.npy')
    # # u_mean = u_mean.astype(np.float32)
    # # print(f'udims: {u_mean.shape}')
    # # print(f'xgrid dims: {x_grid.shape}')
    # v_mean = np.load('D:/singlesource_2d_extended/mean_v_0to180s.npy')
    # v_mean = v_mean.astype(np.float32)
    # # u_tot = np.sqrt(u_mean**2+v_mean**2)

    # cmap = cmr.waterlily_r
    # utils.plot_field_xy(x_grid, y_grid, u_mean[:-1, :-1].T, cmap=cmap, range=[-0.25, 0.25], title='mean u', filepath='ignore/extended_sim/u_mean_arrows.png', arrows=True, u=u_mean.T, v=v_mean.T, dpi=600)
    # # # utils.plot_field_xy(x_grid, y_grid, v_mean[:-1, :-1].T, cmap=cmap, range=[-0.15, 0.15], title='mean v', filepath='ignore/extended_sim/v_mean_v2.png', dpi=600)


    # # Decompose velocity fields into mean and fluctuating components of u and v
    # u_flx = u_flx - u_mean.T
    # v_flx = v_flx - v_mean.T
    
    
    # # u_flx = u_flx[:, stride0:-stride0, :]
    # # v_flx = v_flx[:, stride0:-stride0, :]
    # # u_flx = utils.reynolds_decomp(u_flx, time_ax=0)
    # # v_flx = utils.reynolds_decomp(v_flx, time_ax=0)[1]
    # # u_flx = u_flx - np.mean(u_flx, axis=0, keepdims=False)
    # # v_flx = v_flx - np.mean(v_flx, axis=0, keepdims=False)

    # # Compute fluctuating strain rate 
    # dt = 0.02  # sec
    # duflx_dy = np.gradient(u_flx, dt, dx, dx)
    # duflx_dy = np.asarray(duflx_dy)
    # duflx_dy = duflx_dy[1, :, :, :]
    # np.save('ignore/duflx_dy_extendedSim.npy', duflx_dy)
    # dvflx_dx = np.gradient(v_flx, dt, dx, dx)
    # dvflx_dx = np.asarray(dvflx_dx)
    # dvflx_dx = dvflx_dx[2, :, :, :]
    # np.save('ignore/dvflc_dx_extendedsim.npy', dvflx_dx)

    # # duflx_dy = np.load('ignore/duflx_dyv2.npy')
    # # dvflx_dx = np.load('ignore/dvflc_dxv2.npy')
    # # # flx_strain = np.mean(1/2 * (duflx_dy + dvflx_dx), axis=0)
    # # # flx_strain = np.mean(duflx_dy, axis=0)

    # nu = 1.5 * stride**(-5) # kinematic viscosity 
    # # Compute viscous energy dissipation rate
    # # epsilon = 2 * flx_strain * flx_strain
    # # epsilon = np.mean(duflx_dy * duflx_dy, axis=0)
    # epsilon = np.mean((duflx_dy**2 + dvflx_dx**2), axis=0)
    # print(f'avg viscous dissipation: {np.mean(epsilon)}')
    # print(f'avg viscous dissipation, x=[0, 0.05] and y=[-0.2, 0.2]: {np.mean(epsilon[200:stride00, 0:stride0])}')
    # print(f'avg viscous dissipation, x=[0.7, 0.75] and y=[-0.2, 0.2]: {np.mean(epsilon[200:stride00, 1400:1500])}')
    # plt.close()
    # plt.pcolormesh(epsilon)
    # plt.colorbar()
    # plt.show()

    # Taylor_microscale = np.sqrt(15 / epsilon) * np.sqrt(0.5 * (np.mean(u_flx**2, axis=0) + np.mean(v_flx**2, axis=0)))
    # # Taylor_microscale = np.sqrt(stride / epsilon) * np.sqrt((np.mean(u_flx**2, axis=0)))





    # uprime = np.sqrt(0.5*(np.mean(u_flx**2, axis=0) + np.mean(v_flx**2, axis=0)))  # root mean square velocity
    # # K_tscale = 1 / np.sqrt(np.mean(np.sqrt((duflx_dy**2 + dvflx_dx**2))**2, axis=0))
    # # Taylor_microscale = np.sqrt(15)*uprime*K_tscale
    # Taylor_Re = uprime * Taylor_microscale / nu
    # np.save('ignore/extended_sim/Taylor_microscale_v1.npy', Taylor_microscale)
    # np.save('ignore/extended_sim/Taylor_Re_v1.npy', Taylor_Re)
    
    # # PLOT: Taylor microscale, computed per Carbone & Wilczek, 2024 (https://doi.org/stride.stride17/jfm.2024.165)
    # print(f'avg Taylor microscale: {np.mean(Taylor_microscale)}')
    # print(f'avg Taylor microscale from x=[0, 0.05] and y=[-0.15, 0.15]: {np.mean(Taylor_microscale[123:723, 0:stride0])}')
    # print(f'avg Taylor microscale from x=[0, 0.05] and y=[-0.15, 0.15]: {np.mean(Taylor_microscale[123:723, 1400:1500])}')
    # plt.close()
    # plt.pcolormesh(Taylor_microscale)
    # plt.colorbar()
    # plt.savefig('ignore/extended_sim/Taylor_microscale_v1.png', dpi=600)
    # plt.show()

    # # PLOT: Taylor Reynolds number using methods from Carbone & Wilczek, 2024 (https://doi.org/stride.stride17/jfm.2024.165)
    # print(f'avg Taylor Re: {np.mean(Taylor_Re)}')
    # print(f'avg Taylor Re from x=[0, 0.05] and y=[-0.15, 0.15]: {np.mean(Taylor_Re[123:723, 0:stride0])}')
    # print(f'avg Taylor Re from x=[0, 0.05] and y=[-0.15, 0.15]: {np.mean(Taylor_Re[123:723, 1400:1500])}')
    # plt.close()
    # plt.pcolormesh(Taylor_Re)
    # plt.colorbar()
    # plt.savefig('ignore/extended_sim/Taylor_Re_v1.png', dpi=600)
    # plt.show()








    # # Compute turbulent kinetic energy
    # tke = 0.5 * (np.mean(u_flx**2, axis=0) + np.mean(v_flx**2, axis=0))
    # np.save('ignore/extended_sim/tke_extendedsim.npy', tke)
    # # t_intensity = np.sqrt(tke) / np.sqrt(u_mean**2 + v_mean**2)

    # # PLOT: turbulent intensity and turbulent kinetic energy
    # cmap = cmr.ember
    # utils.plot_field_xy(x_grid, y_grid, tke, title='turbulent kinetic energy', cmap=cmap, filepath='ignore/extended_sim/tke_fullDomain.png', dpi=600, trimmed=False)
    # # utils.plot_field_xy(x_grid, y_grid, t_intensity, title='turbulence intensity', cmap=cmap, range=[0, 0.8], filepath='ignore/t_intensity_trimmed.png', dpi=600, trimmed=True)


    # # # PLOTS: integral scales
    # # ils_uxstream = np.load('ignore/ILS_u_cross_stream.npy')
    # # ils_vstream = np.load('ignore/ILS_vstreamwise.npy')
    # ils_ustream = np.load('ignore/ILS_ustreamwise.npy')
    # ils_vxstream = np.load('ignore/ILS_v_cross_stream.npy')

    # cmap = cmr.gothic_r
    # # utils.plot_field_xy(x_grid, y_grid, np.flipud(ils_uxstream),  title='Integral Length Scale, u cross-stream', cmap=cmap, range=[0, 0.15], filepath='ignore/ils_u_cross_stream_trimmed_alt.png', dpi=600, trimmed=True)
    # # utils.plot_field_xy(x_grid, y_grid, np.flipud(ils_vstream),  title='Integral Length Scale, v streamwise', cmap=cmap, range=[0, 0.15], filepath='ignore/ils_v_streamwise_trimmed_alt.png', dpi=600, trimmed=True)
    # utils.plot_field_xy(x_grid, y_grid, np.flipud(ils_ustream),  title='Integral Length Scale, u streamwise', cmap=cmap, range=[0, 0.15], filepath='ignore/ils_u_streamwise_flipped.png', dpi=600, trimmed=False)
    # utils.plot_field_xy(x_grid, y_grid, np.flipud(ils_vxstream),  title='Integral Length Scale, v cross-stream', cmap=cmap, range=[0, 0.15], filepath='ignore/ils_v_cross_stream_flipped.png', dpi=600, trimmed=False)
 
    # its_u = scipy.io.loadmat('ignore/Tux_array_u_update.mat')
    # its_u = np.array(its_u['Tux_array'])
    # its_v = scipy.io.loadmat('ignore/Tux_array_v.mat')
    # its_v = np.array(its_v['Tux_array'])

    # cmap = cmr.jungle_r
    # utils.plot_field_xy(x_grid, y_grid, its_u, title='Integral Time Scale, u', cmap=cmap, range=[0, 1.5], filepath='ignore/its_u_trimmed.png', dpi=600, smooth=True, trimmed=True)
    # utils.plot_field_xy(x_grid, y_grid, its_v, title='Integral Time Scale, v', cmap=cmap, range=[0, 1.5], filepath='ignore/its_v_trimmed_alt.png', dpi=600, smooth=True, trimmed=True)


    # # Compute Reynolds Stress Tensor
    # u_prime_sq = np.mean(u_flx**2, axis=0)
    # v_prime_sq = np.mean(v_flx**2, axis=0)
    # uv_prime = np.mean(u_flx*v_flx, axis=0)
    # aspect_ratio = np.minimum(u_prime_sq, v_prime_sq)/np.maximum(u_prime_sq, v_prime_sq)
    # print(aspect_ratio.shape)
    # mean_ar = np.mean(aspect_ratio[123:723, :])
    # print(f"mean aspect ratio: {mean_ar}")

    # # PLOTS: components of Reynolds stress
    # cmap = cmr.dusk_r
    # utils.plot_field_xy(x_grid, y_grid, u_prime_sq, title='Reynolds Stress: u normal', cmap=cmap, filepath='ignore/rs_uprimesq.png', dpi=600)
    # utils.plot_field_xy(x_grid, y_grid, v_prime_sq, title='Reynolds Stress: v normal', cmap=cmap, filepath='ignore/rs_vprimesq.png', dpi=600)
    # utils.plot_field_xy(x_grid, y_grid, uv_prime, title='Reynolds Stress: uv shear', cmap=cmap, filepath='ignore/rs_uvprime.png', dpi=600)
    # utils.plot_field_xy(x_grid, y_grid, u_prime_sq/v_prime_sq, title='Reynolds Stress Anisotropy: u_stress/v_stress', cmap=cmap, range=[1, 5], filepath='ignore/rs_uvanisotropy.png', dpi=600)
    # utils.plot_field_xy(x_grid, y_grid, aspect_ratio, title='Reynolds Stress Anisotropy: Aspect Ratio', cmap=cmap, filepath='ignore/rs_normalaspectratio.png', dpi=600)

if __name__ == '__main__':
    main()

