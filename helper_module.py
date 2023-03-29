import numpy as np


def get_random_test_seq(dataset, num_of_sequences):
    seq_list = []
    for seq in range(num_of_sequences):
        max_idx = dataset.__len__()
        seq_list.append(dataset.__getitem__(np.random.randint(low=0, high=max_idx)))

    return seq_list


def get_CRPS_sequence(dataset, idx_list):
    seq_list = []
    for idx in idx_list:
        seq_list.append(dataset.__getitem__(idx))

    return seq_list


def get_20min_forecast_sequence(dataset, idx_list):
    seq_list = []
    for idx in idx_list:
        seq_list.append(dataset.__getitem__(idx))

    return seq_list


def draw_conv_sample(dataset):
    max_idx = dataset.__len__()
    fetched_sequence = dataset.__getitem__(np.random.randint(low=0, high=max_idx))
    strat_conv_arr = fetched_sequence[1]
    # Flatten the 3D array into a 1D array
    flat_arr = strat_conv_arr.flatten()
    # Count the number of occurrences of the integer 1
    num_occurrences = np.count_nonzero(flat_arr == 1)
    if num_occurrences < 1000:
        draw_conv_sample(dataset=dataset)
    return fetched_sequence


def get_convective_sequence(dataset, num_of_sequences):
    seq_list = []
    for seq in range(num_of_sequences):
        sequence = draw_conv_sample(dataset=dataset)
        seq_list.append(sequence)

    return seq_list


def get_stratiform_sequence(dataset, num_of_sequences):
    seq_list = []
    for seq in range(num_of_sequences):
        max_idx = dataset.__len__()
        seq_list.append(dataset.__getitem__(np.random.randint(low=0, high=max_idx)))

    return seq_list
