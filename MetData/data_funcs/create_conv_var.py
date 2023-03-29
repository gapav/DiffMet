import numpy as np
import xarray as xr
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm


import xarray as xr


def create_conv_strat_var_from_classification(da_cls, da_conv):
    """Create a new DataArray from two input DataArrays.
 

    Args:
        da_cls (xarray.DataArray): Input DataArray with values indicating 3=snow, 2=sleet, 1=rain, 0=None
        da_conv (xarray.DataArray): Input DataArray with values indicating 1=convective, 0=stratiform
    Returns:
        xarray.DataArray: Output DataArray with values set according to conditions below. 
    """

    # set all values to zeros:
    conv_strat_da = xr.zeros_like(da_cls)
    # set all values in new variable to 1 where it is rain and it is convective:
    conv_strat_da.values[(da_conv.values == 1) & (da_cls.values == 1)] = 1

    # set all values in new variable to 2 where it is rain and it is stratiform:
    conv_strat_da.values[(da_conv.values == 0) & (da_cls.values == 1)] = 2

    return conv_strat_da


def create_conv_strat_var_from_LWE(da_lwe, da_conv):
    """Create a new DataArray from two input DataArrays.
 

    Args:
        da_cls (xarray.DataArray): Input DataArray with values indicating 3=snow, 2=sleet, 1=rain, 0=None
        da_conv (xarray.DataArray): Input DataArray with values indicating 1=convective, 0=stratiform
    Returns:
        xarray.DataArray: Output DataArray with values set according to conditions below. 
    """

    # set all values to zeros:
    conv_strat_da = xr.zeros_like(da_conv)
    # set all values in new variable to 1 where it is rain and it is convective:
    conv_strat_da.values[(da_conv.values == 1) & (da_lwe.values > 0)] = 1

    # set all values in new variable to 2 where it is rain and it is stratiform:
    conv_strat_da.values[(da_conv.values == 0) & (da_lwe.values > 0)] = 2

    return conv_strat_da


def add_var_to_all_files_from_classification(in_folder, out_folder):
    """Add a new variable with conv/strat values to all xarray files in a folder, based on 
    classification variable from thredds dataset. 

    Args:
        in_folder (str): Path to the input.
        out_folder (str): Path to the output.
    """

    for filename in tqdm(os.listdir(in_folder)):
        f = os.path.join(in_folder, filename)
        # checking if it is a file
        if filename != ".DS_Store":
            if os.path.isfile(f):
                ds = xr.open_dataset(f)
                da_cls = ds["classification"]
                da_conv = ds["is_convective"]

                new_variable = create_conv_strat_var(da_cls=da_cls, da_conv=da_conv)

                new_ds = ds.assign(new_variable=new_variable)
                new_ds["new_variable"].attrs[
                    "units"
                ] = "2=stratiform, 1=convective, 0=None"
                new_ds["new_variable"].attrs[
                    "long_name"
                ] = "Convective/stratiform/No precipitation"

                new_ds = new_ds.rename({"new_variable": "conv_strat_none"})

                new_ds.to_netcdf(f"{out_folder}/{filename}")


def add_var_to_all_files_from_LWE(in_folder, out_folder=None):
    """Add a new variable with conv/strat values to all xarray files in a folder, based on 
    LWE variable from thredds dataset. 

    Args:
        in_folder (str): Path to the input.
        out_folder (str): Path to the output.
    """
    if out_folder == None:
        out_folder = in_folder

    for filename in tqdm(os.listdir(in_folder)):
        f = os.path.join(in_folder, filename)
        # checking if it is a file
        if filename != ".DS_Store":
            if os.path.isfile(f):
                ds = xr.open_dataset(f)
                da_lwe = ds["lwe_precipitation_rate"]
                da_conv = ds["is_convective"]

                new_variable = create_conv_strat_var_from_LWE(
                    da_lwe=da_lwe, da_conv=da_conv
                )

                new_ds = ds.assign(new_variable=new_variable)
                new_ds["new_variable"].attrs[
                    "units"
                ] = "2=stratiform, 1=convective, 0=None"
                new_ds["new_variable"].attrs[
                    "long_name"
                ] = "Convective/stratiform/No precipitation"
                new_ds["new_variable"] = new_ds["new_variable"].astype("int8")

                new_ds = new_ds.rename({"new_variable": "conv_strat_none"})

                new_ds = new_ds.drop('is_convective')

                new_ds.to_netcdf(f"{out_folder}/{filename}")
