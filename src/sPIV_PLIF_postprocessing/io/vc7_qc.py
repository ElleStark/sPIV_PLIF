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
u_path = Path('E:/sPIV_PLIF_ProcessedData/PIV/10.01.2025_30cms_nearbed_smTG15cm_neuHe0.875_air0.952_PIV0.01_iso_u.npy')
v_path = Path('E:/sPIV_PLIF_ProcessedData/PIV/10.01.2025_30cms_nearbed_smTG15cm_neuHe0.875_air0.952_PIV0.01_iso_v.npy')

# v = np.load(v_path)
# v = np.flipud(v)  # flip the v velocity field 
# np.save(v_path, v)  # save the flipped v velocity field back to the same file

# plotting parameters
frame_idx = 1
out_path = Path('E:/sPIV_PLIF_ProcessedData/PIV/Baseline/quiver_frame_' + str(frame_idx) + '.png')
vecstride_x = 8  # optional, arrow spacing
vecstride_y = 15  # optional, arrow spacing
scale = 0.04  # optional, scaling of arrow length
headwidth = 5  # optional, width of the arrow head
headlength = 5  # optional, length of the arrow head
tailwidth = 0.001  # optional, width of the arrow tail
x_coords = TARGET_X  # optional, x coordinates of the velocity field
y_coords = TARGET_Y  # optional, y coordinates of the velocity field

quiver_from_npy(
    u_path,
    v_path,
    out_path,
    frame_idx=frame_idx,
    stride_x=vecstride_x,
    stride_y=vecstride_y,
    scale=scale,
    headwidth=headwidth,
    headlength=headlength,
    tail_width=tailwidth,
    x_coords=x_coords,
    y_coords=y_coords,
)
