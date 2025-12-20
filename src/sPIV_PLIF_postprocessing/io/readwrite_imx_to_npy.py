# read in .imx file from DaVis and write to .npy file
import lvpyio as lv
import numpy as np
import matplotlib.pyplot as plt

save_dir = 'E:/sPIV_PLIF_processedData/'
save_name = 'PLIF_8.29_30cms_smTG15cm_55pctHe1.0_45pctair0.816_PIV0.02_Iso.npy'
data_dir = 'D:/PLIF_20Hz_data/data_8.29_30cms_smTG15cm_55pctHe1.0_45pctair0.816_PIV0.02_Iso_L4/'

# specify list of file paths and new file names
path1 = data_dir + 'L4.set'
path2 = data_dir + 'L3.set'
offset = 0
#changes

# # set up buffer objects (see lavision pyio manual)
# s1 = lv.read_set(path1)
# s2 = lv.read_set(path2)
# # check size and n frames
# buffer = s1[0]
# arr = buffer.as_masked_array()
# dims = np.shape(arr.data)
# total_frames = len(s1) + len(s2)  
# total_frames = total_frames - offset  # manually select subset in time if needed
# # total_frames = 10  # for testing

# # initialize array for storing collated data
# combined_data = np.zeros((dims[0], dims[1], total_frames), dtype=np.float32)

# # collate images into single sequential stack
# frame_list = [x+offset for x in list(range(total_frames))]
# # frame_list = range(total_frames)
# for i in frame_list:
#     # if even, read from first set, if odd from second set
#     if i%2==0:  
#         buffer = s1[int(i/2)]
#     else:
#         buffer = s2[int((i-1)/2)]

#     arr = buffer.as_masked_array()  # necessary conversion (see lavision pyio manual)
#     tempdata = arr.data  # check if needed

#     combined_data[:, :, i-offset] = tempdata


print(f'combined data dims: {combined_data.shape}')
# QC plots of a few frames
for frame in [0, 1, 1000, 1001, 5000, 5001]:
    plt.figure()
    plt.imshow(combined_data[:, :, frame], cmap='jet', vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig(save_dir + f'QC/PLIF_55He_regGrid_frame_{frame}.png')  

# save stack of raw data (if desired)
save_file = save_dir + save_name
np.save(save_file, combined_data)
