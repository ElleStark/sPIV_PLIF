# read in .imx file from DaVis and write to .npy file
import lvpyio as lv
import numpy as np
import matplotlib.pyplot as plt

save_dir = 'C:/Users/LaVision/Documents/Data_processing/DaVis_LIF data/data/ff_bg_cal_imgs/'
data_dir = 'D:/Elle/PLIF_MOX_2023-07-21/'

# specify list of file paths and new file names
path_fname_list  = [['bg_r64_L3/Avgbgr64_L3.set','Avg_bg_r64_L3'], ['bg_r64_L3/Avgbgr64_L4.set','Avg_bg_r64_L4'], 
                    ['ff_r68_L3/Avg_ffr68_L3.set', 'Avg_ff_r68_L3'], ['ff_r68_L3/Avg_ffr68_L4.set', 'Avg_ff_r68_L4'],
                    ['bg_r69_L3/Avg_ff73_L3.set', 'Avg_bg_r69_L3'], ['bg_r69_L3/Avg_ff73_L4.set', 'Avg_bg_r69_L4'],
                    ['ff_r73_L4/Avg_ff73_L3.set', 'Avg_ff_r73_L3'], ['ff_r73_L4/Avg_ff73_L4.set', 'Avg_ff_r73_L4'],
                    ['bg_r74_L3/Avg_bg74_L3.set', 'Avg_bg_r74_L3'], ['bg_r74_L3/Avg_bg74_L4.set', 'Avg_bg_r74_L4'], 
                    ['ff_r78_L3/Avg_ff78_L3.set', 'Avg_ff_r78_L3'], ['ff_r78_L3/Avg_ff78_L4.set', 'Avg_ff_r78_L4']
]

for path, name in path_fname_list:
    read_path = data_dir + path
    set = lv.read_set(read_path)
    buffer = set[0]
    arr = buffer.as_masked_array()
    dataset = arr.data

    save_path = save_dir + name + '.npy'
    np.save(name, dataset)

    plt.imshow(dataset, cmap='Greys_r', vmin=100, vmax=400)
    plt.savefig(save_dir+name+'.png')
    plt.show()
