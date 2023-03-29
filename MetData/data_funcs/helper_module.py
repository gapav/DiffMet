import numpy as np
import xarray as xr
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from numpy import unravel_index


def get_lws(month: int, year: int) -> xr.Dataset:

    URL = "https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/2020/08/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.laea-yrwms-1000.20200831.nc?"
    dataset2 = xr.open_dataset(URL2)
    dataset2

    return xr.open_dataset(url)


def get_open_DAP_precipitation(
    year: str, month: str, day: str, time_start: int, time_end: int
) -> xr.Dataset:

    # if year == "2021":
    #     URL =f"https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/2021/{month}/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.{year}{month}{day}.nc?time[0:1:286],lwe_precipitation_rate[0:1:286][0:1:2100][0:1:1600]"
    if year == "2021" or year == "2022":
        x_s = 1400
        x_e = x_s + 255
        y_s = 530
        y_e = y_s + 255
        # URL = f"https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/{year}/{month}/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.{year}{month}{day}.nc?time[{time_start}:{time_end}],lwe_precipitation_rate[{time_start}:{time_end}][{x_s}:{x_e}][{y_s}:{y_e}]"
        URL = f"https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/{year}/{month}/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.{year}{month}{day}.nc?time[{time_start}:{time_end}],lwe_precipitation_rate[{time_start}:{time_end}][{x_s}:{x_e}][{y_s}:{y_e}],is_convective[{time_start}:{time_end}][{x_s}:{x_e}][{y_s}:{y_e}]"
    else:
        print(f"year has to be a string and 2021 or 2022, not {year, type(year)}")

    dataset = xr.open_dataset(URL)
    return dataset


def get_open_DAP_lonlat(
    year: str, month: str, day: str, time_start: int, time_end: int
) -> xr.Dataset:

    # if year == "2021":
    #     URL =f"https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/2021/{month}/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.{year}{month}{day}.nc?time[0:1:286],lwe_precipitation_rate[0:1:286][0:1:2100][0:1:1600]"
    if year == "2021" or year == "2022":
        x_s = 1400
        x_e = x_s + 255
        y_s = 530
        y_e = y_s + 255
        # URL = f"https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/{year}/{month}/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.{year}{month}{day}.nc?time[{time_start}:{time_end}],lwe_precipitation_rate[{time_start}:{time_end}][{x_s}:{x_e}][{y_s}:{y_e}]"
        URL = f"https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/{year}/{month}/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.{year}{month}{day}.nc?time[{time_start}:{time_end}],lwe_precipitation_rate[{time_start}:{time_end}][{x_s}:{x_e}][{y_s}:{y_e}],lon[{x_s}:{x_e}][{y_s}:{y_e}],lat[{x_s}:{x_e}][{y_s}:{y_e}]"
    else:
        print(f"year has to be a string and 2021 or 2022, not {year, type(year)}")

    dataset = xr.open_dataset(URL)
    return dataset


def find_plot_max_prec_frame(sample: np.ndarray):

    max_prec_frame = np.zeros((256, 256))
    max_prec_rate = 0
    max_frame_idx = 0
    for frame in tqdm(range(sample.shape[0])):

        max_frame_prec = np.nanmax(sample[frame, :, :])
        if max_frame_prec > max_prec_rate:
            max_prec_rate = max_frame_prec
            max_prec_frame = sample[frame, :, :]
            max_frame_idx = frame

    max_x, max_y = unravel_index(max_prec_frame.argmax(), max_prec_frame.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    fig.suptitle("Radar Precipitation Rates")
    fig.patch.set_facecolor("white")
    ax1.scatter(max_y, max_x, s=300, facecolors="none", edgecolors="fuchsia")
    ax1.set_title(
        f"Max precipitation rate in frame[{max_frame_idx}]: {np.max(max_prec_frame)} mm/hr, at idx {max_x,max_y}"
    )
    im1 = ax1.imshow(max_prec_frame, cmap="coolwarm")
    cbar = fig.colorbar(im1, orientation="vertical")
    cbar.set_label("mm/hr", rotation=270)

    sub_frame = max_prec_frame[max_x - 5 : max_x + 5, max_y - 5 : max_y + 5]
    ax2.imshow(sub_frame, cmap="coolwarm")
    ax2.set_title("Sliced out sub frame with extreme values")
    for ax in (ax1, ax2):
        ax.set(xlabel="Km", ylabel="Km")

    plt.show()


def plot_histogram(dataset: np.ndarray, decimals: int, bins="auto"):
    flat_sample_data = np.reshape(dataset, newshape=(dataset.shape[0] * 256 * 256))
    flat_sample_data = flat_sample_data[~np.isnan(flat_sample_data)]
    flat_sample_data = flat_sample_data[flat_sample_data != 0.0]
    flat_sample_data = np.around(flat_sample_data, decimals=decimals)

    plt.hist(flat_sample_data, bins=bins)
    plt.show()


def log_transform_sqrt(frame: np.ndarray):
    return np.log(1 + np.sqrt(frame))


def log_transform_4_power(frame: np.ndarray):
    return np.log(1 + frame ** (0.25))


if __name__ == "__main__":
    dataset = xr.open_dataset(
        "https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/2020/04/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.laea-yrwms-1000.20200430.nc?time[0:1:287],Xc[0:1:2066],Yc[0:1:2242],lwe_precipitation_rate[0:1:0][0:1:0][0:1:0]"
    )
    print(dataset)
    # get_lws(month=4, year=2020)


def get_precipitation_histo(
    directory, output_seq, low_sum_tresh, max_val_tresh, clutter_prob
):
    seq_sum = []
    seq_sum_below_tresh = 0
    seq_sum_over_Tresh = 0
    cluttered_seq = 0
    for filename in tqdm(os.listdir(directory)):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename != ".DS_Store":
            ds = xr.open_dataset(f)

            arr = ds["lwe_precipitation_rate"].values
            clutter_arr = ds["clutter_probability"].values

            len_of_seq_in_file = arr.shape[0]
            for seq in range(0, len_of_seq_in_file - output_seq):
                sequence = arr[seq : seq + output_seq, :, :]
                sequence_sum = np.sum(sequence)
                sequence_max_value = np.max(sequence)

                if sequence_sum < low_sum_tresh:
                    seq_sum_below_tresh += 1
                    continue

                if sequence_max_value > max_val_tresh:
                    seq_sum_over_Tresh += 1
                    continue

                if clutter_prob:
                    if np.max(clutter_arr) > 10:
                        cluttered_seq += 1
                        continue

                seq_sum.append((sequence_sum))

    seq_arr = np.array(seq_sum)
    print(f"{low_sum_tresh=}, {max_val_tresh=}, {clutter_prob=}")
    print(
        f"{len(seq_arr)=},{seq_sum_below_tresh=},{seq_sum_over_Tresh=}, {cluttered_seq=}"
    )

    seq_arr = seq_arr[~np.isnan(seq_arr)]
    plt.hist(seq_arr)
