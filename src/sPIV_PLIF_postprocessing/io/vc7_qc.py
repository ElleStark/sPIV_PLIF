from pathlib import Path
import numpy as np

# Make imports work when run as a standalone script (python src/.../vc7_qc.py)
try:
    from src.sPIV_PLIF_postprocessing.visualization.viz import quiver_from_npy
    from src.sPIV_PLIF_postprocessing.io.readwrite_vc7_to_npy import TARGET_X, TARGET_Y
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.sPIV_PLIF_postprocessing.visualization.viz import quiver_from_npy
    from src.sPIV_PLIF_postprocessing.io.readwrite_vc7_to_npy import TARGET_X, TARGET_Y

# paths to the u and v velocity fields
u_path = Path('E:/sPIV_PLIF_ProcessedData/PIV/8.29_30cmsPWM2.25_smTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_u.npy')
v_path = Path('E:/sPIV_PLIF_ProcessedData/PIV/8.29_30cmsPWM2.25_smTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_v.npy')

v = np.load(v_path)
v = np.fliplr(v)  # flip the v velocity field 
np.save(v_path, v)  # save the flipped v velocity field back to the same file

# plotting parameters
frame_idx = 1
out_path = Path('E:/sPIV_PLIF_ProcessedData/PIV/Baseline/quiver_frame_' + str(frame_idx) + '.png')
vecstride = 15  # optional, arrow spacing
scale = 0.04  # optional, scaling of arrow length
x_coords = TARGET_X  # optional, x coordinates of the velocity field
y_coords = TARGET_Y  # optional, y coordinates of the velocity field

quiver_from_npy(
    u_path,
    v_path,
    out_path,
    frame_idx=frame_idx,
    stride=vecstride,
    scale=scale,
    x_coords=x_coords,
    y_coords=y_coords,
)
