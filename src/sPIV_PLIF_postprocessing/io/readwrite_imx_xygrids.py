import numpy as np
import lvpyio as lv  # LaVision's package for importing .imx/.vc7 files
import matplotlib.pyplot as plt

def extract_axes(buffer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get frame data (float32, masked -> NaN) and physical x/y axes from a buffer.
    """
    arr = buffer.as_masked_array()
    frame = np.array(arr.filled(np.nan), dtype=np.float32)
    frameData = buffer[0]
    x_scale = frameData.scales.x
    y_scale = frameData.scales.y
    x_axis = x_scale.offset + x_scale.slope * np.arange(frame.shape[1])
    y_axis = 0 - y_scale.slope * np.arange(frame.shape[0])
    return frame, x_axis, y_axis

def main():
    # Example usage
    input_path = "D:\\SimulData_20Hz\\PLIF_proc_9.26.25_sourceConfig\\data_9.26.2025_30cms_DiffusiveSource_smTG15cm_neuHe0.876_air0.941_PIV0.02_Iso_L3\\Subtract_r_bg_L3\\divide_ffmbg\\Divide_C0_instantaneous\\AddCameraAttributes\\calibrated\\Resize\\Median filter.set"
    output_path_x = "C:\\Users\\Lavision\\Documents\\sPIV_PLIF_processing\\sPIV_PLIF\\ignore\\data\\xgrid_diffusive.npy"
    output_path_y = "C:\\Users\\Lavision\\Documents\\sPIV_PLIF_processing\\sPIV_PLIF\\ignore\\data\\ygrid_diffusive.npy"

    set = lv.read_set(input_path)
    buffer = set[0]

    # Extract x and y axes
    frame, x_axis, y_axis = extract_axes(buffer)

    # Combine x and y axes into a single 2D array (if needed)
    xgrid, ygrid = np.meshgrid(x_axis, y_axis)

    plt.figure(figsize=(8, 6))
    plt.pcolor(xgrid, ygrid, frame, shading='auto')
    plt.show()

    # Save the grids as .npy files
    np.save(output_path_x, xgrid.astype(np.float32, copy=False))
    np.save(output_path_y, ygrid.astype(np.float32, copy=False))

if __name__ == "__main__":
    main()