"""Turbulent scalar-flux, gradient-diffusion, and effective-diffusivity analysis for fixed-plane PLIF/sPIV data.

Conventions (match the processed data on disk):
    arrays are (t, y, x); y (rows) is STREAMWISE with velocity v, x (columns) is CROSS-STREAM with velocity u,
    w is out-of-plane. The source is at the high-row end of the grid; downstream distance s = y_max - y.

Assumptions stated throughout:
    * Fixed-frame (Eulerian) Reynolds decomposition: the mean is the time mean at each fixed pixel, so plume
      meander is folded into the "turbulent" fluxes.
    * Single-plane measurement: only in-plane gradients dC/dx, dC/dy exist. No dC/dz and no K_z are estimated;
      <w'c'> is reported as a standalone transport diagnostic only.

See METHODS.md in this folder for definitions and limitations.
"""
