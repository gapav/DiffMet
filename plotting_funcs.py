import torch
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import datetime


def plot_convective_binary_map():
    plt.imshow()


def show_sequence(sample, seq_len, pred_ahead, reverse_transformed=None):
    fig, axs = plt.subplots(1, 4, figsize=(30, 5))

    cond_frames = sample[1]
    model_input = sample[0]

    cond_max = torch.max(cond_frames)
    input_max = torch.max(model_input)

    max_plot_val = torch.max(cond_max, input_max)
    print(
        f"min/max value in cond frames: {torch.min(cond_frames)}, {torch.max(cond_frames)}"
    )
    print(
        f"min/max value in model_input: {torch.min(model_input)}, {torch.max(model_input)}"
    )

    stamp_string = sample[3]
    time_stamp = stamp_string.split(".")[0]
    time_stamp = int(time_stamp)
    stamp_arr = [time_stamp + (i * 60 * 5) for i in range(4)]

    timestamp_arr = [
        datetime.datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M:%S")
        for stamp in stamp_arr
    ]

    ax2 = plt.subplot(141)
    ax2.imshow((cond_frames[0, :, :]), cmap="jet", vmin=-1.0, vmax=max_plot_val)
    ax2.set_title(f"{timestamp_arr[0]}")

    ax7 = plt.subplot(142)
    ax7.imshow((cond_frames[1, :, :]), cmap="jet", vmin=-1.0, vmax=max_plot_val)
    ax7.set_title(f"{timestamp_arr[1]}")

    ax3 = plt.subplot(143)
    ax3.imshow((cond_frames[2, :, :]), cmap="jet", vmin=-1.0, vmax=max_plot_val)
    ax3.set_title(f"{timestamp_arr[2]}")

    ax3 = plt.subplot(144)
    ax3.imshow((model_input[0, :, :]), cmap="jet", vmin=-1.0, vmax=max_plot_val)
    ax3.set_title(f"{timestamp_arr[3]}")

    plt.show()
