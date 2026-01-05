# Source code to run on CU Boulder HPC resources: Blanca cluster
# Computes integral length scales for multisource plume dataset for u and v in both x-stream and streamwise directions
# v2 uses distributed data loading to load chunks of u_data into different processes
# Elle Stark May 2024

# import h5py
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import numpy.linalg as LA
from math import log, sqrt
from scipy.interpolate import RegularGridInterpolator
import time

# Set up logging for convenient messages
logger = logging.getLogger('ftlempipy')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
logger.addHandler(handler)
INFO = logger.info
WARN = logger.warning
DEBUG = logger.debug

# GLOBAL REFERENCE FOR FINDING FILES
file_tag = '_fractalCase.npy'


def get_vfield(filename, t0, y, dt, xmesh, ymesh):

    # Convert from time to frame
    frame = int(t0 / dt)
    # DEBUG(f'frame for u interp: {frame}')
    u_data = np.load(filename+'u.npy')[:, :, frame]
    print(f"frame = {frame}, u mean = {np.mean(u_data):.3e}, std = {np.std(u_data):.3e}")
    v_data = np.load(filename+'v.npy')[:, :, frame]
    u_data = np.squeeze(u_data)
    v_data = np.squeeze(v_data)

    ymesh_vec = np.flipud(ymesh)[:, 0]
    # ymesh_vec = ymesh[:, 0]
    xmesh_vec = xmesh[0, :]

    x_grid = xmesh
    # x_offset = xmesh_vec[-1] / 2
    x_offset = 0
    y_offset = 0
    x_grid = x_grid - x_offset  # Center x coordinates on zero for velocity field extension
    y_grid = ymesh + y_offset

    # Set up interpolation functions
    # can use cubic interpolation for continuity between the segments (improve smoothness)
    # set bounds_error=False to allow particles to go outside the domain by extrapolation
    u_interp = RegularGridInterpolator((ymesh_vec, xmesh_vec), u_data,
                                        method='cubic', bounds_error=False, fill_value=None)
    v_interp = RegularGridInterpolator((ymesh_vec, xmesh_vec), v_data,
                                        method='cubic', bounds_error=False, fill_value=None)

    
    # Define length of transition zone and boundaries
    Delta = 0.0005 * 30
    x_max = np.max(np.abs(x_grid))
    y_max = np.max(np.abs(y_grid))

    # Spatial average of velocity field over grid for this time
    avg_u = np.mean(u_data)
    avg_v = np.mean(v_data)
    
    # Tensor for linear velocity field computation
    v_l_tensor = np.empty((2, 2), dtype=np.float32)
    v_l_tensor[0, 0] = np.mean(x_grid * u_data - y_grid * v_data) / np.mean(x_grid ** 2 + y_grid ** 2)
    v_l_tensor[0, 1] = np.mean(y_grid * u_data) / np.mean(y_grid ** 2)
    v_l_tensor[1, 0] = np.mean(x_grid * v_data) / np.mean(x_grid ** 2)
    v_l_tensor[1, 1] = np.mean(y_grid * v_data - x_grid * u_data) / np.mean(x_grid ** 2 + y_grid ** 2)
    
    # Get x and y points
    x_pts = y[0, :] - x_offset
    y_pts = y[1, :] + y_offset

    # Calculate distance from center for x and y
    x_abs = np.abs(x_pts)
    y_abs = np.abs(y_pts)

    # Outside the grid condition
    outside_cond = (x_abs >= x_max) | (y_abs >= y_max)

    # Inside grid condition
    inside_cond = (x_abs <= (x_max - Delta)) & (y_abs <= (y_max - Delta))

    # Transition zone condition
    transition_cond = ~outside_cond & ~inside_cond

    # Interpolate u and v for points inside the grid
    u_inside = u_interp((y_pts - y_offset, x_pts + x_offset))
    v_inside = v_interp((y_pts - y_offset, x_pts + x_offset))

    # For points outside, use linear extension of velocity field
    u_outside = v_l_tensor[0, 0] * x_pts + v_l_tensor[0, 1] * y_pts + avg_u
    v_outside = v_l_tensor[1, 0] * x_pts + v_l_tensor[1, 1] * y_pts + avg_v

    # Delta functions for transition zone
    delta_x = np.where(x_abs <= (x_max - Delta), Delta ** 3,
                    2 * x_abs ** 3 + 3 * (Delta - 2 * x_max) * x_abs ** 2 + 6 * x_max * (x_max - Delta) * x_abs + x_max ** 2 * (3 * Delta - 2 * x_max))
    delta_y = np.where(y_abs <= (y_max - Delta), Delta ** 3,
                    2 * y_abs ** 3 + 3 * (Delta - 2 * y_max) * y_abs ** 2 + 6 * y_max * (y_max - Delta) * y_abs + y_max ** 2 * (3 * Delta - 2 * y_max))

    # Compute velocity for transition zone
    v_l_u = v_l_tensor[0, 0] * x_pts + v_l_tensor[0, 1] * y_pts + avg_u
    v_l_v = v_l_tensor[1, 0] * x_pts + v_l_tensor[1, 1] * y_pts + avg_v

    u_orig = u_interp((y_pts - y_offset, x_pts + x_offset))
    v_orig = v_interp((y_pts - y_offset, x_pts + x_offset))

    u_transition = v_l_u + (u_orig - v_l_u) * delta_x * delta_y / Delta ** 6
    v_transition = v_l_v + (v_orig - v_l_v) * delta_x * delta_y / Delta ** 6

    # Combine results using np.where
    u = np.where(outside_cond, u_outside, np.where(inside_cond, u_inside, u_transition))
    v = np.where(outside_cond, v_outside, np.where(inside_cond, v_inside, v_transition))

    vfield = np.array([u, v])

    plot_times = [0.5, 1, 2, 4]

    if t0 in plot_times:
        # Plot u: horizontal component of velocity
        plt.pcolormesh(u.reshape(791, 790))
        plt.colorbar()
        plt.savefig(f'/projects/elst4602/sPIV_PLIF/QC/u_interp_t{t0}.png')

        # Plot v: vertical component of velocity
        plt.pcolor(v.reshape(791, 790))
        plt.colorbar()
        plt.savefig(f'/projects/elst4602/sPIV_PLIF/QC/v_interp_t{t0}.png')

    return vfield


# def advect_improvedEuler(filename, t0, y0, dt, ftle_dt, xmesh, ymesh):
#     # get the slopes at the initial and end points
#     f1 = get_vfield(filename, t0, y0, dt, xmesh, ymesh)
#     f2 = get_vfield(filename, t0 + ftle_dt, y0 + ftle_dt * f1, dt, xmesh, ymesh)
#     y_out = y0 + ftle_dt / 2 * (f1 + f2)

#     return y_out 

def rk4singlestep(filename, t0, y0, dt, ftle_dt, xmesh, ymesh):
    """
    Single step of 4th-order Runge-Kutta integration. Use instead of scipy.integrate.solve_ivp to allow for
    vectorized computation of bundle of initial conditions. Reference: https://www.youtube.com/watch?v=LRF4dGP4xeo
    Note that self.vfield must be a function that returns an array of [u, v] values
    :param dt: scalar value of desired time step
    :param t0: start time for integration
    :param y0: starting position of particles
    :return: final position of particles
    """
    # RK4 first computes velocity at full steps and partial steps
    f1 = get_vfield(filename, t0, y0, dt, xmesh, ymesh)
    f2 = get_vfield(filename, t0 + dt / 2, y0 + (dt / 2) * f1, dt, xmesh, ymesh)
    f3 = get_vfield(filename, t0 + dt / 2, y0 + (dt / 2) * f2, dt, xmesh, ymesh)
    f4 = get_vfield(filename, t0 + dt, y0 + dt * f3, dt, xmesh, ymesh)
    # RK4 then takes a weighted average to move the particle
    y_out = y0 + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
    return y_out



def find_max_eigval(A):
    """The function computes the eigenvalues and eigenvectors of a two-dimensional symmetric matrix.
    from TBarrier repository by Encinas Bartos, Kaszas, Haller 2023: https://github.com/EncinasBartos/TBarrier
    Parameters:
        A: array(2,2), input matrix


    Returns:
        lambda_min: float, minimal eigenvalue
        lambda_max: float, maximal eigenvalue
        v_min: array(2,), minimal eigenvector
        v_max: array(2,), maximal eigenvector
    """
    A11 = A[0, 0]  # float
    A12 = A[0, 1]  # float
    A22 = A[1, 1]  # float

    discriminant = (A11 + A22) ** 2 / 4 - (A11 * A22 - A12 ** 2)  # float

    if discriminant < 0 or np.isnan(discriminant):
        return np.nan, np.nan, np.zeros((1, 2)) * np.nan, np.zeros((1, 2)) * np.nan

    lambda_max = (A11 + A22) / 2 + np.sqrt(discriminant)  # float
    # lambda_min = (A11 + A22) / 2 - np.sqrt(discriminant)  # float

    # v_max = np.array([-A12, A11 - lambda_max])  # array (2,)
    # v_max = v_max / np.sqrt(v_max[0] ** 2 + v_max[1] ** 2)  # array (2,)

    # v_min = np.array([-v_max[1], v_max[0]])  # array (2,)

    return lambda_max


def compute_flow_map(filename, start_t, integration_t, dt, ftle_dt, nx, ny, xmesh_ftle, ymesh_ftle, xmesh_uv, ymesh_uv):
    
    n_steps = abs(int(integration_t / ftle_dt))  # number of timesteps in integration time
    if start_t == integration_t:
        DEBUG(f'Timesteps in integration time: {n_steps}.')
    
    # Set up initial conditions
    yIC = np.zeros((2, nx * ny))
    yIC[0, :] = xmesh_ftle.reshape(nx * ny)
    yIC[1, :] = ymesh_ftle.reshape(nx * ny)

    y_in = yIC

    for step in range(n_steps):
        tstep = step * ftle_dt + start_t
        DEBUG(f'timesteps: {tstep}')
        y_out = rk4singlestep(filename, tstep, y_in, dt, ftle_dt, xmesh_uv, ymesh_uv)
        y_in = y_out

    y_out = np.squeeze(y_out)

    return y_out


def compute_ftle(filename, xmesh_ftle, ymesh_ftle, start_t, integration_t, dt, ftle_dt, spatial_res, xmesh_uv, ymesh_uv):
    # Extract grid dimensions
    grid_height = len(ymesh_ftle[:, 0])
    grid_width = len(xmesh_ftle[0, :])
    
    # Compute flow map (final positions of particles - initial positions already stored in mesh_ftle arrays)
    final_pos = compute_flow_map(filename, start_t, integration_t, dt, ftle_dt, grid_width, grid_height, xmesh_ftle, ymesh_ftle, xmesh_uv, ymesh_uv)
    x_final = final_pos[0]
    x_final = x_final.reshape(grid_height, grid_width)
    DEBUG(f'xfinal (300, 300): {x_final[300, 300]}')
    y_final = final_pos[1]
    y_final = y_final.reshape(grid_height, grid_width)

    # Initialize arrays for jacobian approximation and ftle
    jacobian = np.empty([2, 2], float)
    ftle = np.zeros([grid_height, grid_width], float)

    # Loop through positions and calculate ftle at each point
    # Leave borders equal to zero (central differencing needs adjacent points for calculation)
    for i in range(1, grid_width - 1):
        for j in range(1, grid_height - 1):
            jacobian[0][0] = (x_final[j, i + 1] - x_final[j, i - 1]) / (2 * spatial_res)
            jacobian[0][1] = (x_final[j + 1, i] - x_final[j - 1, i]) / (2 * spatial_res)
            jacobian[1][0] = (y_final[j, i + 1] - y_final[j, i - 1]) / (2 * spatial_res)
            jacobian[1][1] = (y_final[j + 1, i] - y_final[j - 1, i]) / (2 * spatial_res)

            # Cauchy-Green tensor
            gc_tensor = np.dot(np.transpose(jacobian), jacobian)
    
            # compute largest eigenvalue of CG tensor
            lamda = LA.eigvals(gc_tensor)
            max_eig = np.max(lamda)

            # Compute FTLE at each location
            ftle[j][i] = 1 / (abs(integration_t)) * log(sqrt(abs(max_eig)))

    return ftle


def plot_ftle_snapshot(ftle_part, xmesh, ymesh, odor=False, fname=None, frame=None, odor_xmesh=None, odor_ymesh=None):
    fig, ax = plt.subplots()

    ftle_plot = np.squeeze(ftle_part[0, :, :])

    # Get desired FTLE snapshot data
    plt.contourf(xmesh, ymesh, ftle_plot, 100, cmap=plt.cm.Greys)
    plt.title(f'FTLE (gray lines). value at 300, 300: {ftle_plot[300, 300]}')
    plt.colorbar()

    ax.set_aspect('equal', adjustable='box')

    return fig


def main():

    # MPI setup and related data
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # number of processes will be determined from ntasks listed on Slurm job script (.sh file) 
    num_procs = comm.Get_size()

    # Define common variables on all processes
    filename = '/scratch/alpine/elst4602/sPIV_PLIF/PIV/8.29_30cmsPWM2.25_FractalTG15cm_noHC_PIVairQ0.02_Neu49pctHe0.897_51pctair0.917_Iso_'
    integration_time = 0.4  # seconds

    # Define dataset-based variables on process 0
    if rank==0:
        DEBUG("starting Python script")
        start = time.time()
        spatial_res = 0.0005  # spatial resolution in m
        dt_freq = 20
        dt = 1 / dt_freq  # convert from Hz to seconds
        ymesh_uv = np.load('/scratch/alpine/elst4602/sPIV_PLIF/PIV/y_coords.npy')
        # ymesh_uv = ymesh_uv.T
        ymesh_uv = ymesh_uv / 1000
        xmesh_uv = np.load('/scratch/alpine/elst4602/sPIV_PLIF/PIV/x_coords.npy')
        # xmesh_uv = xmesh_uv.T
        xmesh_uv = xmesh_uv / 1000
        u_grid_dims = np.shape(xmesh_uv)
        x_min = xmesh_uv[0][0]
        x_max = xmesh_uv[0][-1]
        y_min = ymesh_uv[0][0]
        y_max = ymesh_uv[-1][0]
        ymesh_uv = ymesh_uv.flatten()
        xmesh_uv = xmesh_uv.flatten()
        duration = 0.5  # s
        duration = (duration - integration_time) / dt  # adjust duration to account for advection time at first FTLE step
        # duration = 200 # SPECIAL CASE: COMPUTING FOR PARTIAL DATA!!!

        # Create grid of particles with desired spacing
        particle_spacing = spatial_res  # can determine visually if dx is appropriate based on smooth contours for FTLE

        # x and y vectors based on velocity mesh limits and particle spacing
        xmesh_ftle = np.linspace(x_min, x_max, int(u_grid_dims[1] * spatial_res/particle_spacing))
        ymesh_ftle = np.linspace(y_min, y_max, int(u_grid_dims[0] * spatial_res/particle_spacing))
        xmesh_ftle, ymesh_ftle = np.meshgrid(xmesh_ftle, ymesh_ftle, indexing='xy')
        grid_dims = np.shape(xmesh_ftle)
        ymesh_ftle = ymesh_ftle.flatten()
        xmesh_ftle = xmesh_ftle.flatten()
        DEBUG(f'Grid dimensions: {grid_dims}.')
        DEBUG(f"Time to load metadata on process 0: {time.time() - start} s.")

    else:
        # These variables will be broadcast from process 0 based on file contents
        grid_dims = None
        u_grid_dims = None
        dt = None
        duration = None  # total timesteps (idxs) for FTLE calcs
        particle_spacing = None
        spatial_res = None

    # Broadcast dimensions of x and y grids to each process for pre-allocating arrays  
    grid_dims = comm.bcast(grid_dims, root=0) # note, use bcast for Python objects, Bcast for Numpy arrays
    u_grid_dims = comm.bcast(u_grid_dims, root=0)
    dt = comm.bcast(dt, root=0)
    ftle_dt = -dt   # negative for backward-time FTLE
    particle_spacing = comm.bcast(particle_spacing, root=0) 
    duration = comm.bcast(duration, root=0) 
    spatial_res = comm.bcast(spatial_res, root=0)
    
    if rank != 0:
        xmesh_ftle = np.empty([grid_dims[0]*grid_dims[1]], dtype='d')
        ymesh_ftle = np.empty([grid_dims[0]*grid_dims[1]], dtype='d')
        xmesh_uv = np.empty([u_grid_dims[0]*u_grid_dims[1]], dtype='d')
        ymesh_uv = np.empty([u_grid_dims[0]*u_grid_dims[1]], dtype='d')

    comm.Bcast([xmesh_ftle, MPI.DOUBLE], root=0)
    comm.Bcast([ymesh_ftle, MPI.DOUBLE], root=0)
    comm.Bcast([xmesh_uv, MPI.DOUBLE], root=0)
    comm.Bcast([ymesh_uv, MPI.DOUBLE], root=0)

    # Reshape x and y meshes
    xmesh_ftle = xmesh_ftle.reshape(grid_dims)
    ymesh_ftle = ymesh_ftle.reshape(grid_dims)
    xmesh_uv = xmesh_uv.reshape(u_grid_dims)
    ymesh_uv = ymesh_uv.reshape(u_grid_dims)

    # Compute chunk sizes for each process 
    chunk_size = duration // num_procs
    DEBUG(f'Chunk size: {chunk_size}')
    remainder = duration % num_procs

    # Find start and end time index for each process
    start_idx = int(integration_time / abs(ftle_dt) + rank * chunk_size + min(rank, remainder))
    end_idx = int(start_idx + chunk_size + (1 if rank < remainder else 0))
    DEBUG(f'Process {rank} start idx: {start_idx}; end idx: {end_idx}')

    # Compute FTLE and save to .npy on each process for each timestep
    ftle_chunk = np.zeros([(end_idx - start_idx), grid_dims[0], grid_dims[1]], dtype='d')
    timesteps = range(end_idx - start_idx)
    # timesteps = [0]  # TEST WITH SINGLE TIMESTEP
    # ftle_chunk = np.zeros([1, grid_dims[0], grid_dims[1]], dtype='d')
    ftle_chunk = np.zeros([len(timesteps), grid_dims[0], grid_dims[1]], dtype='d')

    start = time.time()
    offset_idx = 1000
    DEBUG(f'Began FTLE computation on process {rank} at {start}.')
    for idx in timesteps:
        start_t = (offset_idx +start_idx + idx) * dt
        ftle_field = compute_ftle(filename, xmesh_ftle, ymesh_ftle, start_t, integration_time, dt, ftle_dt, spatial_res, xmesh_uv, ymesh_uv)
        ftle_chunk[idx, :, :] = ftle_field
    DEBUG(f'Ended FTLE computation on process {rank} after {(time.time()-start)/60} min.')

    # dynamic file name in /scratch based on rank/idxs
    data_fname = f'/projects/elst4602/sPIV_PLIF/ftle_data/{(rank) : 04d}_t{round(start_idx*dt, 2)}to{round(end_idx*dt, 2)}s_{file_tag}.npy'
    np.save(data_fname, ftle_chunk)
    
    # Plot and save figure at final timestep of each process in /rc_scratch
    ftle_fig = plot_ftle_snapshot(ftle_chunk, xmesh_ftle, ymesh_ftle, odor=False)
    plot_fname = f'/projects/elst4602/sPIV_PLIF/ftle_plots/{(rank) : 04d}_t{round(start_idx*dt, 2)}s_{file_tag}.png'
    ftle_fig.savefig(plot_fname, dpi=300)

    DEBUG(f"Process {rank} completed with result size {ftle_chunk.shape}")

if __name__=="__main__":
    main()


