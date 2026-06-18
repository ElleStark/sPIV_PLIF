"""read in original velocity fields, 
    create copy with specified lag, 
    save as new .npy array. 
    Elle Stark, June 2026
"""

import numpy as np
from pathlib import Path

# user inputs
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData/PIV/Interpolated_to_PLIF")
U_PATH = BASE_PATH / "piv_baseline_u.npy"
V_PATH = BASE_PATH / "piv_baseline_v.npy"
LAG_FRAMES = 100
U_OUT_PATH = BASE_PATH / "Decorrelated_vel_arrays" / f"lag{LAG_FRAMES}_baseline_u.npy"
V_OUT_PATH = BASE_PATH / "Decorrelated_vel_arrays" / f"lag{LAG_FRAMES}_baseline_v.npy"


# read in velocity data
u_orig = np.load(U_PATH)
v_orig = np.load(V_PATH)
t_size = u_orig.shape[0]
print(f"time steps: {t_size} \n")

# create lagged version
u_lagged = np.zeros_like(u_orig)
v_lagged = np.zeros_like(v_orig)
u_lagged[:t_size-LAG_FRAMES, :, :] = u_orig[LAG_FRAMES:, :, :]
u_lagged[t_size-LAG_FRAMES:, :, :] = u_orig[:LAG_FRAMES, :, :]
v_lagged[:t_size-LAG_FRAMES, :, :] = v_orig[LAG_FRAMES:, :, :]
v_lagged[t_size-LAG_FRAMES:, :, :] = v_orig[:LAG_FRAMES, :, :]

# save lagged versions
np.save(U_OUT_PATH, u_lagged)
print(f"saved lagged u to {U_OUT_PATH}. \n")
np.save(V_OUT_PATH, v_lagged)
print(f"saved lagged v to {V_OUT_PATH}. \n")

