import numpy as np
from scipy.interpolate import griddata

def interp_to_shared_grid(im7_data, vec_df, xg, yg):
    # scalar interpolation
    h_raw = im7_data.as_masked_array()
    h_raw=h_raw.data
    
    # Get scale and offset from the .scales attributes
    dx = im7_data.scales.x.slope
    dy = im7_data.scales.y.slope
    x0 = im7_data.scales.x.offset
    y0 = im7_data.scales.y.offset

    ny, nx = h_raw.shape
    x_scl = x0 + np.arange(nx) * dx
    y_scl = y0 + np.arange(ny) * dy
    Xs, Ys = np.meshgrid(x_scl, y_scl)

    h_interp = griddata((Xs.ravel(), Ys.ravel()), h_raw.ravel(), (xg, yg), method='linear')

    print("xg shape:", xg.shape)
    print("yg shape:", yg.shape)

    print("vec_df['x'].shape:", vec_df['x'].shape)
    print("vec_df['y'].shape:", vec_df['y'].shape)
    print("vec_df['u'].shape:", vec_df['u'].shape)
    xvec, yvec = np.meshgrid(vec_df['x'].values, vec_df['y'].values)

    # vector interpolation
    u_interp = griddata((xvec.ravel(), yvec.ravel()), 
                        vec_df['u'].values.ravel(), (xg, yg), method='linear')
    v_interp = griddata((xvec.ravel(), yvec.ravel()), 
                        vec_df['v'].values.ravel(), (xg, yg), method='linear')
    
    return h_interp, u_interp, v_interp