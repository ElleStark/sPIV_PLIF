#!/usr/bin/env python
"""Original dataset runner (archived copy).

Kept as a backup of the user-provided non-.py runner. This file is not
used by the package but preserved for reference.
"""
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib import rcParams
import lvpyio as lv  # LaVision's package for importing .imx files to Python
import logging
import pivpy as pp
from pivpy import io
import scipy.interpolate as interp
import xarray as xr
import os
import utils
import utils.interp_shared_grid

# Set up logging for convenient messages
logger = logging.getLogger('sPIV_PLIF')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
logger.addHandler(handler)
INFO = logger.info
WARN = logger.warn
DEBUG = logger.debug


def main():
    # Directory of data locations. assumes PIV data is in .vc7 format and PLIF data is .im7 format
    piv_dir = 'D:/Elle/SimulData_7.23.2025_PIVprocessing/7.23_45pctHe0.403_55pctair0.493_PIV0.02_15cms_TG25cm/StereoPIV_MPd(3x32x32_75%ov)/PostProc_frame1650to1750/'
    plif_dir = 'D:/Elle/SimulData_7.23.2025_PLIF/data_7.23_45pctHe0.403_55pctair0.493_PIV0.02_15cms_TG25cm/Subtract_bg/Divide_ffmbg/Divide_C0_22/AddCameraAttributes/ImageCorrection/Median filter/5x5 smoothing/Resize_01/CopySelected/'

    snapshot = True
    frame_to_plot = 63  # index of frame for snapshot
    make_movie = True
    movie_file = 'ignore/plots/NeuHe55pct_15cms_LgTG25cm_10s.mp4'

    # load file lists
    im7_files = sorted(glob(os.path.join(plif_dir, '*.im7')))
    vc7_files = sorted(glob(os.path.join(piv_dir, '*.vc7')))
    n_frames = min(len(im7_files), len(vc7_files))

    # read first frame to define grid in real-world coordinates
    scalar_data = lv.read_buffer(im7_files[0])
    scalar_data = scalar_data[0]
    vec_data = io.load_vc7(vc7_files[0])

    # extract vector data
    x_vec, y_vec = vec_data['x'].values, vec_data['y'].values
    u_vec, v_vec = vec_data['u'].values, vec_data['v'].values

    # set up real-world coordinate grid
    nx, ny = 540, 640
    x_min = min(x_vec.min(), 0)
    x_min = -130
    x_max = x_vec.max()
    x_max = 130
    y_min = min(y_vec.min(), 0)
    y_max = y_vec.max()
    xg, yg = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))

    # Subsample for quiver
    vec_stride = 50
    xq = xg[::vec_stride, ::vec_stride]
    yq = yg[::vec_stride, ::vec_stride]

    # ... rest of original script omitted in archived copy


if __name__ == "__main__":
    main()
