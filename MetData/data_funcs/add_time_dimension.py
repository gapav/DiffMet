import numpy as np
import xarray as xr
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import datetime

import xarray as xr


def add_time_dim(folder, out_folder):
    """Moves the time from a 1d array to a 2d array and adds it
    as a dimension to be able to work as a np array. 

    Args:
        in_folder (str): Path to the input.
        out_folder (str): Path to the output.
    """

    for filename in tqdm(os.listdir(folder)):
        f = os.path.join(folder, filename)
        # checking if it is a file
        if filename != ".DS_Store":
            if os.path.isfile(f):
                ds = xr.open_dataset(f)
                time = ds["time"].values

                new_variable = np.empty(shape=(time.shape[0], 256, 256))

                for step in range(time.shape[0]):
                    # datetime:
                    dt = time[step]

                    new_variable[step, 0, 0] = dt

                new_ds = ds.assign(new_variable=new_variable)
                new_ds = new_ds.rename({"new_variable": "time_var"})

                new_ds.to_netcdf(f"{out_folder}/{filename}", mode="w")

                # ds = ds.assign({'new_variable': arr})

