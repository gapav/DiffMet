import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime
import os
from tqdm import tqdm


def create_sequences(
    out_seq_len: int,
    in_folder: str,
    out_folder: str,
    low_sum_tresh: int,
    prec_tresh_val: int,
):
    max_prec_val_global = 0
    sequence_saved_count = 0
    max_prec_val_over_prec_tresh_val = 0
    seq_sum_smaller_low_sum_tresh = 0
    seq_with_nans = 0
    for filename in tqdm(os.listdir(in_folder)):
        if filename == ".DS_Store":
            continue

        f = os.path.join(in_folder, filename)

        # checking if it is a file
        if not os.path.isfile(f):
            continue

        ds = xr.open_dataset(f)

        lwe = ds["lwe_precipitation_rate"].values
        conv_strat_none = ds["conv_strat_none"].values
        time = ds["time"].values

        data_arr = np.zeros(shape=(3, lwe.shape[0], lwe.shape[1], lwe.shape[2]))
        data_arr[0, :, :, :] = lwe
        data_arr[1, :, :, :] = conv_strat_none
        seq_len = lwe.shape[0]
        dataset_max_val = 0

        for seq in range(0, seq_len - out_seq_len):

            sequence = data_arr[:, seq : seq + out_seq_len, :, :]

            nan_count = np.sum(np.isnan(sequence))
            if nan_count > 0:
                seq_with_nans += 1
                continue

            lwe_sequence = sequence[0, :, :, :]
            seq_LWE_sum = np.sum(lwe_sequence)
            max_prec_val = np.max(lwe_sequence)

            # if max_prec_val > max_prec_val_global:
            #     max_prec_val_global = max_prec_val

            if max_prec_val > prec_tresh_val:
                max_prec_val_over_prec_tresh_val += 1
                continue

            if seq_LWE_sum < low_sum_tresh:
                seq_sum_smaller_low_sum_tresh += 1
                continue
            if max_prec_val > dataset_max_val:
                dataset_max_val = max_prec_val
            # get date time for sequence
            time_seq = time[seq : seq + out_seq_len]

            # use time delta for first frame as fname
            fname = int(time_seq[0]) / 1000000000
            fname = int(fname)

            # np.save(f"{out_folder}/lwe/{str(fname)}", lwe_sequence)
            # np.save(f"{out_folder}/conv_strat/{str(fname)}", conv_sequence)
            np.save(f"{out_folder}/{str(fname)}", sequence)

            sequence_saved_count += 1
        os.remove(f)

    print(
        f"{sequence_saved_count=}, {dataset_max_val=},  {seq_sum_smaller_low_sum_tresh=},{max_prec_val_over_prec_tresh_val=}, {seq_with_nans=}"
    )
