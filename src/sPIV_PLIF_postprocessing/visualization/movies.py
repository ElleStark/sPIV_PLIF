# create animations of processed data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ffmpeg

npy_file = 'I:/MOXLIF_processing/results/r60_ffmbg_test.npy' 
t_lims = [1200, 1300]  # time limits for the animation
x_lims = [0, 599]  # x-axis limits
y_lims = [0, 599]  # y-axis limits

vmin = 0
vmax = 1

data = np.load(npy_file)[:, :, t_lims[0]:t_lims[1]]

fig, ax = plt.subplots()
im = ax.pcolormesh(data[:, :, 0], cmap='viridis', vmin=vmin, vmax=vmax)
ax.set_title('Processed Data Animation')    

def update_frame(frame):
    
    im.set_array(data[:, :, frame].flatten())
    ax.set_xlabel(f'Time Frame: {frame + t_lims[0]}')
    return im, 
ani = animation.FuncAnimation(fig, update_frame, frames=data.shape[2], blit=False, interval=50)
ani.save('I:/MOXLIF_processing/results/ffmbg_animation.mp4', writer=animation.FFMpegWriter(fps=20))
# ffmpeg.input('I:/MOXLIF_processing/results/ffmbg_animation.mp4').output('I:/MOXLIF_processing/results/ffmbg_animation_compressed.mp4', vcodec='libx264', crf=23).run()
plt.show()

