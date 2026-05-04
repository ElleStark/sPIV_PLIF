# interrogate and plot plume data from .npy file

import numpy as np
import matplotlib.pyplot as plt
import pickle
import av
from scipy.interpolate import RegularGridInterpolator


METADATA_FILE_PATH = 'E:/sPIV_PLIF_ProcessedData/Emonet_smoke/new_smoke_2a_metadata.pkl'
READ_METADATA = False
CONVERT_AVI_DATA = False
INTERPOLATE_TO_GRID = False
COMPUTE_FLX_NORM = True
GRID_SPACE_MM = 0.5
FPS = 180  # frames per second
SIZE = (1164, 902, 4200)  # (height, width, time) of the plume data
MM_PER_PX_X = 0.2927
MM_PER_PX_Y = 0.1428

DATA_FILE_PATH = 'E:/sPIV_PLIF_ProcessedData/Emonet_smoke/new_smoke_2a.avi'
SAVE_DIR = 'E:/sPIV_PLIF_ProcessedData/Emonet_smoke/'
SAVE_FILE_NAME = 'HalfmmGrid_new_smoke_2a.npy'
FRAMES_TO_PLOT = [0, 1000, 2000, 3000, 4000] 

if READ_METADATA:
# read metadata from .pkl file
    with open(METADATA_FILE_PATH, 'rb') as file:
        # Load the contents
        data = pickle.load(file)
    # If 'data' is a dictionary, check its keys for metadata
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{key}: {value}")
    else:
        print("Data is not a dictionary. Type:", type(data))

if CONVERT_AVI_DATA:
    container = av.open(DATA_FILE_PATH)
    plume = np.zeros(SIZE, dtype=np.float32)
    i=0
    for frame in container.decode(video=0):
        plume[:, :, i] = frame.to_ndarray(format='gray')
        i += 1
    print('Number of frames read: ', i)

    np.save(f'{SAVE_DIR}{SAVE_FILE_NAME}', plume)
else:
    plume = np.load(f'{SAVE_DIR}{SAVE_FILE_NAME}')

# plume data attributes
print('Original Plume data shape:', plume.shape)
print('Original Plume data dtype:', plume.dtype)
print('Original Plume data min:', np.nanmin(plume))
print('Original Plume data max:', np.nanmax(plume))



if INTERPOLATE_TO_GRID:
    # original grid
    x_mm = SIZE[1] * MM_PER_PX_X
    y_mm = SIZE[0] * MM_PER_PX_Y
    print('Original grid size (mm): ', y_mm, ', ', x_mm)
    x_orig = np.arange(SIZE[1]) * MM_PER_PX_X
    y_orig = np.arange(SIZE[0]) * MM_PER_PX_Y
    x = np.arange(0, int(np.floor(x_mm)), GRID_SPACE_MM)
    y = np.arange(0, int(np.floor(y_mm)), GRID_SPACE_MM)
    grid_y, grid_x = np.meshgrid(y, x, indexing='ij')

    interpolated_plume = np.zeros((len(y), len(x), SIZE[2]), dtype=np.float32)
    for i in range(SIZE[2]):
        interpolator = RegularGridInterpolator((y_orig, x_orig), plume[:, :, i])
        interpolated_plume[:, :, i] = interpolator((grid_y.flatten(), grid_x.flatten())).reshape(len(y), len(x))
    
    plume = interpolated_plume

    np.save(f'{SAVE_DIR}HalfmmGrid_{SAVE_FILE_NAME}', plume)


if COMPUTE_FLX_NORM:
    # normalize concentration values to max value (255)
    plume = plume / np.nanmax(plume)

    # compute fluctuating concentration by subtracting time-averaged concentration at each pixel
    time_avg_conc = np.nanmean(plume, axis=2)
    plume = plume - time_avg_conc[:, :, np.newaxis]

    np.save(f'{SAVE_DIR}FluctuatingNorm_{SAVE_FILE_NAME}', plume)

# plume data attributes
print('Plume data shape:', plume.shape)
print('Plume data dtype:', plume.dtype)
print('Plume data min:', np.nanmin(plume))
print('Plume data max:', np.nanmax(plume))

# plot the plume data
for frame in FRAMES_TO_PLOT:
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(plume[:, :, frame], cmap='gray')
    plt.colorbar()
    plt.title(f'Plume Data: Frame {frame}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

