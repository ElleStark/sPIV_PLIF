# create animations of processed data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ffmpeg
import lvpyio as lv
from matplotlib import colors 

# npy_file = 'I:/MOXLIF_processing/results/r60_ffmbg_test.npy' 
t_lims = [0, 600]  # time limits for the animation
x_lims = [0, 599]  # x-axis limits
y_lims = [100, 500]  # y-axis limits

vmin = 0
vmax = 0.5

# fmin = 0
# fmax = 100
read_path = 'I:/Processed_Data/PLIF/Smoothed/plif_baseline_smoothed.npy'
write_path = 'I:/Processed_Data/PLIF/QC/plif_animation_baseline.mp4'

data = np.load(read_path)[x_lims[0]:x_lims[1], y_lims[0]:y_lims[1], t_lims[0]:t_lims[1]]

# # set up buffer objects (see lavision pyio manual)
# s1 = lv.read_set(read_path)
# # check size and n frames
# buffer = s1[fmin]
# arr = buffer.as_masked_array()
# data = np.array(arr.filled(np.nan), dtype=np.float32)

# frameData = buffer[0]
# x_scale = frameData.scales.x
# y_scale = frameData.scales.y
# x_axis = x_scale.offset + x_scale.slope * np.arange(data.shape[1])
# y_axis = y_scale.offset + y_scale.slope * np.arange(data.shape[0])

# fig, ax = plt.subplots()
# # im = ax.pcolormesh(data[:, :, 0], cmap='viridis', vmin=vmin, vmax=vmax)
# im = ax.pcolormesh(data, cmap='turbo', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
# ax.set_title('Raw Data Animation')
# fig.colorbar(im)    

# def update_frame(frame):
#     s1 = lv.read_set(read_path)
#     buffer = s1[fmin+frame]
#     arr = buffer.as_masked_array()
#     data = np.array(arr.filled(np.nan), dtype=np.float32)
#     im.set_array(data.flatten())
#     # im.set_array(data[:, :, frame].flatten())
#     ax.set_xlabel(f'Time Frame: {fmin+frame}')
#     return im, 
# ani = animation.FuncAnimation(fig, update_frame, frames=data.shape[2], blit=False, interval=50)


fig, ax = plt.subplots()
im = ax.pcolormesh(data[:, :, 0], cmap='gray', norm=colors.Normalize(vmin=vmin, vmax=vmax))
ax.set_aspect('equal')
ax.set_title('Baseline PLIF Animation') 
def update_frame(frame):
    data_temp =data[:, :, frame]
    im.set_array(data_temp.flatten())
    ax.set_xlabel(f'Time Frame: {t_lims[0]+frame}')
    return im,


ani = animation.FuncAnimation(fig, update_frame, frames=t_lims[1]-t_lims[0], blit=False, interval=50)
writer = animation.FFMpegWriter(fps=20)
ani.save(write_path, writer=writer, dpi=600)
# ffmpeg.input('I:/MOXLIF_processing/results/ffmbg_animation.mp4').output('I:/MOXLIF_processing/results/ffmbg_animation_compressed.mp4', vcodec='libx264', crf=23).run()
plt.show()

