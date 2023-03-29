import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import os
from tqdm import tqdm


def remove_variable(in_folder: str, var_name: str, out_folder: str = None):
    """Creates a new dataset without the specified variable

    Args:
        in_folder (str): Path to the input.
        out_folder (str): Path to the output.
    """
    if out_folder is None:
        out_folder = in_folder

    for filename in tqdm(os.listdir(in_folder)):
        f = os.path.join(in_folder, filename)
        # checking if it is a file
        if filename != ".DS_Store":
            if os.path.isfile(f):
                ds = xr.open_dataset(f)
                ds = ds.drop_vars(f"{var_name}")

