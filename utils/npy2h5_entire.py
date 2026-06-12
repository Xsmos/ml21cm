# npy2h5_entire.py
import sys
import numpy as np
import h5py
from pathlib import Path

src = Path(sys.argv[1])
dst = src.with_suffix(".h5")

arr = np.load(src, mmap_mode="r")
print("shape:", arr.shape, "dtype:", arr.dtype)

with h5py.File(dst, "w") as f:
    dset = f.create_dataset(
        "denoising_trajectory",
        shape=arr.shape,
        dtype=arr.dtype,
        chunks=(1, *arr.shape[1:]),   # 每个 denoising step 一个 chunk
        compression=None              # 最快；需要省空间再开 gzip/lzf
    )

    for i in range(arr.shape[0]):
        dset[i] = arr[i]
        if i % 50 == 0:
            print(f"{src.name}: wrote step {i}/{arr.shape[0]}")
