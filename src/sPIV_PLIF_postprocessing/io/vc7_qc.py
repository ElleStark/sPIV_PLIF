from pathlib import Path
import numpy as np

# # Make imports work when run as a standalone script (python src/.../vc7_qc.py)
# try:
#     from src.sPIV_PLIF_postprocessing.visualization.viz import quiver_from_npy
#     from src.sPIV_PLIF_postprocessing.io.readwrite_vc7_to_npy import TARGET_X, TARGET_Y
# except ImportError:
#     import sys
#     sys.path.append(str(Path(__file__).resolve().parents[2]))
#     from src.sPIV_PLIF_postprocessing.visualization.viz import quiver_from_npy
#     from src.sPIV_PLIF_postprocessing.io.readwrite_vc7_to_npy import TARGET_X, TARGET_Y

# paths to the u and v velocity fields
# u_path = Path('E:/sPIV_PLIF_ProcessedData/PIV/Interpolated_to_PLIF/piv_diffusive_u.npy')
# v_path = Path('E:/sPIV_PLIF_ProcessedData/PIV/Interpolated_to_PLIF/piv_diffusive_v.npy')
w_path = Path('E:/sPIV_PLIF_ProcessedData/PIV/Interpolated_to_PLIF/piv_smSource_w.npy')
# c_path = Path('I:/Processed_Data/PLIF/plif_baseline.npy')

# print(f'u dimensions: {np.load(u_path).shape}')
# print(f'v dimensions: {np.load(v_path).shape}')
print(f'w dimensions: {np.load(w_path).shape}')

# u = np.load(u_path)
# u = np.transpose(u, [2, 0, 1])  # flip the u velocity field 
# np.save(u_path, u)  # save the flipped velocity field back to the same file

# v = np.load(v_path)
# v = np.transpose(v, [2, 0, 1]) # flip the v velocity field 
# np.save(v_path, v)  # save the flipped velocity field back to the same file

w = np.load(w_path)
w = np.transpose(w, [2, 0, 1])  # flip the w velocity field 
np.save(w_path, w)  # save the flipped velocity field back to the same file


# # plotting parameters
# frame_idx = 2
# out_path = Path('I:/Processed_Data/PIV/QC/Baseline/quiver_frame_' + str(frame_idx) + '.png')
# vecstride_x = 8  # optional, arrow spacing
# vecstride_y = 15  # optional, arrow spacing
# scale = 0.04  # optional, scaling of arrow length
# headwidth = 5  # optional, width of the arrow head
# headlength = 5  # optional, length of the arrow head
# tailwidth = 0.001  # optional, width of the arrow tail
# x_coords = TARGET_X  # optional, x coordinates of the velocity field
# y_coords = TARGET_Y  # optional, y coordinates of the velocity field

# quiver_from_npy(
#     u_path,
#     v_path,
#     out_path,
#     frame_idx=frame_idx,
#     stride_x=vecstride_x,
#     stride_y=vecstride_y,
#     scale=scale,
#     headwidth=headwidth,
#     headlength=headlength,
#     tailwidth=tailwidth,
#     x_coords=x_coords,
#     y_coords=y_coords,
# )
