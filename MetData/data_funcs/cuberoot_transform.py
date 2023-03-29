import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import os
from tqdm import tqdm


def cuberoot_transform_folder(in_folder, out_folder):
    for filename in tqdm(os.listdir(in_folder)):
        f = os.path.join(in_folder, filename)
        # checking if it is a file
        if filename != ".DS_Store":
            if os.path.isfile(f):
                ds = xr.open_dataset(f)
                ds["lwe_precipitation_rate"] = ds["lwe_precipitation_rate"] ** (1 / 3)
                ds.to_netcdf(f"{out_folder}/{filename}")
