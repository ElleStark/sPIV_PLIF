# read in .imx file from DaVis and write to .npy file
import lvpyio as lv
import numpy as np
import matplotlib.pyplot as plt

save_dir = 'I:/PLIF_20Hz_data/processing/'
save_name = 'data_8.29_30cms_FractalTG15cm_He0.897_air0.917_PIV0.02_iso_L3_combined.npy'
data_dir = 'I:/PLIF_20Hz_data/data_8.29_30cms_FractalTG15cm_He0.897_air0.917_PIV0.02_iso_L3/'

# specify list of file paths and new file names
path1 = data_dir + 'L3.set'
path2 = data_dir + 'L4.set'
offset = 0

# set up buffer objects (see lavision pyio manual)
s1 = lv.read_set(path1)
s2 = lv.read_set(path2)
# check size and n frames
buffer = s1[0]
arr = buffer.as_masked_array()
dims = np.shape(arr.data)
total_frames = len(s1) + len(s2)  
total_frames = total_frames - offset  # manually select subset in time if needed
# total_frames = 10  # for testing

# initialize array for storing collated data
combined_data = np.zeros((dims[0], dims[1], total_frames), dtype=np.float32)

# collate images into single sequential stack
frame_list = [x+offset for x in list(range(total_frames))]
# frame_list = range(total_frames)
for i in frame_list:
    # if even, read from first set, if odd from second set
    if i%2==0:  
        buffer = s1[int(i/2)]
    else:
        buffer = s2[int((i-1)/2)]

    arr = buffer.as_masked_array()  # necessary conversion (see lavision pyio manual)
    tempdata = arr.data  # check if needed

    combined_data[:, :, i-offset] = tempdata


print(f'combined data dims: {combined_data.shape}')
# QC plots of a few frames
for frame in [0, 1, 1000, 1001, 5000, 5001]:
    plt.imshow(combined_data[:, :, frame], cmap='jet')
    plt.colorbar()
    plt.show()  

# save stack of raw data (if desired)
save_file = save_dir + save_name
np.save(save_file, combined_data)