# create animations of processed data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ffmpeg
import lvpyio as lv
from matplotlib import colors 

# npy_file = 'I:/MOXLIF_processing/results/r60_ffmbg_test.npy' 
t_lims = [1200, 1300]  # time limits for the animation
x_lims = [0, 599]  # x-axis limits
y_lims = [0, 599]  # y-axis limits

vmin = 100
vmax = 2000

fmin = 500
fmax = 701
read_path = 'I:/MOXLIF_processing/MOX_LIF_data/bgSubtract_testing/data_r60_L3.set'
write_path = 'I:/MOXLIF_processing/results/r60_raw_animation_25to35s_range2000Log.mp4'

# data = np.load(npy_file)[:, :, t_lims[0]:t_lims[1]]

# set up buffer objects (see lavision pyio manual)
s1 = lv.read_set(read_path)
# check size and n frames
buffer = s1[fmin]
arr = buffer.as_masked_array()
data = np.array(arr.filled(np.nan), dtype=np.float32)

frameData = buffer[0]
x_scale = frameData.scales.x
y_scale = frameData.scales.y
x_axis = x_scale.offset + x_scale.slope * np.arange(data.shape[1])
y_axis = y_scale.offset + y_scale.slope * np.arange(data.shape[0])

fig, ax = plt.subplots()
# im = ax.pcolormesh(data[:, :, 0], cmap='viridis', vmin=vmin, vmax=vmax)
im = ax.pcolormesh(data, cmap='turbo', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
ax.set_title('Raw Data Animation')
fig.colorbar(im)    

def update_frame(frame):
    s1 = lv.read_set(read_path)
    buffer = s1[fmin+frame]
    arr = buffer.as_masked_array()
    data = np.array(arr.filled(np.nan), dtype=np.float32)
    im.set_array(data.flatten())
    # im.set_array(data[:, :, frame].flatten())
    ax.set_xlabel(f'Time Frame: {fmin+frame}')
    return im, 
# ani = animation.FuncAnimation(fig, update_frame, frames=data.shape[2], blit=False, interval=50)
ani = animation.FuncAnimation(fig, update_frame, frames=fmax-fmin, blit=False, interval=50)
ani.save(write_path, writer=animation.FFMpegWriter(fps=10))
# ffmpeg.input('I:/MOXLIF_processing/results/ffmbg_animation.mp4').output('I:/MOXLIF_processing/results/ffmbg_animation_compressed.mp4', vcodec='libx264', crf=23).run()
plt.show()

