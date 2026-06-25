# read in .h5 file and split into 2 files, with specified number of frames in training file and remainder in testing file 

import numpy as np
import h5py
from pathlib import Path

CASE_NAME = "Baseline"
BASE_PATH = Path("E:/sPIV_PLIF_ProcessedData/PLIF")
ORIG_FILE = BASE_PATH / f"{CASE_NAME}_PLIF.h5"
DATASET_NAME = f"DataSets/{CASE_NAME}_FinalData"
SPLIT_FOLDER = BASE_PATH / "Train_Test_split"
N_FRAMES_TRAINING = 4000
N_FRAMES_TOTAL = 6000


def copy_with_filtered_dataset(src_name, src_obj, dest_file, target_path, frames_to_delete):
      # 1. If it's the specific dataset we want to modify
    if src_name == target_path and isinstance(src_obj, h5py.Dataset):
        total_frames = src_obj.shape[0]
        
        # Create a mask of frames to keep
        mask = np.ones(total_frames, dtype=bool)
        mask[frames_to_delete] = False
        kept_count = np.sum(mask)
        
        # Define new shape
        new_shape = (kept_count,) + src_obj.shape[1:]
        
        # Create the modified dataset
        dest_dset = dest_file.create_dataset(
            src_name, 
            shape=new_shape, 
            dtype=src_obj.dtype, 
            chunks=src_obj.chunks,
            compression=src_obj.compression
        )
        
        # Stream the kept frames frame-by-frame
        write_idx = 0
        for i in range(total_frames):
            if mask[i]:
                dest_dset[write_idx] = src_obj[i]
                write_idx += 1
                
        # Copy dataset attributes
        for attr_name, attr_val in src_obj.attrs.items():
            dest_dset.attrs[attr_name] = attr_val

    # 2. If it's a Group, replicate it in the new file
    elif isinstance(src_obj, h5py.Group):
        dest_group = dest_file.require_group(src_name)
        for attr_name, attr_val in src_obj.attrs.items():
            dest_group.attrs[attr_name] = attr_val

    # 3. For all other datasets, copy them entirely
    elif isinstance(src_obj, h5py.Dataset):
        dest_file.copy(src_obj, src_name)


train_file = SPLIT_FOLDER / f"{CASE_NAME}_PLIF_train.h5"
test_file = SPLIT_FOLDER / f"{CASE_NAME}_PLIF_test.h5"

with h5py.File(ORIG_FILE, 'r') as src_f, h5py.File(train_file, 'w') as train_f, h5py.File(test_file, 'w') as test_f:
    for attr_name, attr_val in src_f.attrs.items():
        train_f.attrs[attr_name] = attr_val
        test_f.attrs[attr_name] = attr_val

    src_f.visititems(lambda name, obj: copy_with_filtered_dataset(
        name, obj, train_f, DATASET_NAME, list(range(N_FRAMES_TRAINING, N_FRAMES_TOTAL))
    ))

    src_f.visititems(lambda name, obj: copy_with_filtered_dataset(
        name, obj, test_f, DATASET_NAME, list(range(0, N_FRAMES_TRAINING))
    ))