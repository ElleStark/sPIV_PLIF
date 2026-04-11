# load single im7 or vc7 file and return with associated x and y coordinate grids

import numpy as np
import lvpyio as lv


def load_im7_frame_as_npy(read_path, f_no):
    # set up buffer objects (see lavision pyio manual)
    s1 = lv.read_set(read_path)
    # check size and n frames
    buffer = s1[f_no]
    arr = buffer.as_masked_array()
    frame = np.array(arr.filled(np.nan), dtype=np.float32)

    frameData = buffer[0]
    x_scale = frameData.scales.x
    y_scale = frameData.scales.y
    x_axis = x_scale.offset + x_scale.slope * np.arange(frame.shape[1])
    y_axis = y_scale.offset + y_scale.slope * np.arange(frame.shape[0])

    return frame, x_axis, y_axis


def load_vc7_frame_as_npy(read_path, f_no, vec_grid=3):
    # set up buffer objects (see lavision pyio manual)
    source_set = lv.read_set(read_path)
    vecbuffer = source_set[f_no]

    arr = vecbuffer[0].as_masked_array()
    u = np.array(np.ma.filled(arr["u"], np.nan), dtype=np.float32)
    v = np.array(np.ma.filled(arr["v"], np.nan), dtype=np.float32)
    # w = np.array(np.ma.filled(arr["w"], np.nan), dtype=np.float32)

    frame = vecbuffer[0]
    arr = frame.as_masked_array()
    h, w = frame.shape

    scales = getattr(frame, "scales", None)
    # Build from slope/offset; assume uniform spacing
    x_axis = scales.x.offset + scales.x.slope * vec_grid * np.arange(w)
    y_axis = scales.y.offset + scales.y.slope * vec_grid * np.arange(h)

    return u, v, x_axis, y_axis


