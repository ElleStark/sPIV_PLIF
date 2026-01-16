import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cmasher as cmr

case: str = "fractal"
base_path: Path = Path("E:/sPIV_PLIF_ProcessedData")

w_flx_path = base_path / "flow_properties" / "flx_u_v_w" / f"w_flx_{case}_FINAL_AllTimeSteps.npy"

w_flx = np.load(w_flx_path, mmap_mode="r")
print(f"Loaded w_flx with shape {w_flx.shape} and dtype {w_flx.dtype}")

plt.subplots(figsize=(8, 6))
plt.pcolormesh(w_flx[:, :, 200], shading="auto", cmap=cmr.viola_r, vmin=-0.2, vmax=0.2)
plt.colorbar(label="w' (m/s)")
plt.title(f"w' snapshot for case '{case}' at first time step")
plt.show()


