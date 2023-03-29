import torch
from fdp import extract_index
import matplotlib.pyplot as plt
from radar_transforms import reverse_transform
from tqdm import tqdm
import datetime
import numpy as np


@torch.no_grad()
def get_sample_at_t(

    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    x,
    t,
    condition,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
    model,
    cond_emb_model,
):
    betas_t = extract_index(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract_index(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract_index(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
        x
        - betas_t
        * model(x, t, condition, cond_emb_model)
        / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = extract_index(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def plot_samples_during_train(
    test_list,
    rgb_grayscale,
    img_out_size,
    sequence_length,
    device,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
    model,
    T,
    pred_ahead,
    numb_cond,
    betas,
    max_prec_val,
    epoch,
    out_folder,
    cond_emb_model,
):
    img_count = 0
    for test_sample in test_list:
        img_size = img_out_size
        # print(f"test_seq.shape = {test_seq.shape}")

        lwe_data = test_sample[0]

        stamp_string = test_sample[2]
        time_stamp = stamp_string.split(".")[0]
        time_stamp = int(time_stamp)
        stamp_arr = [time_stamp + (i * 60 * 5) for i in range(8)]

        timestamp_arr = [
            datetime.datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M:%S")
            for stamp in stamp_arr
        ]

        pred_area_idx = pred_ahead * 3  # max anticipated movement in wind

        test_seq = test_sample[1]
        # Sample noise
        img = torch.randn((1, rgb_grayscale, img_size, img_size), device=device)

        cond_frames = test_seq[1]
        # print(f"cond_frames.shape = {cond_frames.shape}")
        cond_frames = cond_frames.to(device)

        fig, axs = plt.subplots(1, 6, figsize=(30, 5))

        test_seq += 1
        test_seq /= 2
        test_seq *= max_prec_val
        test_seq **= 3
        max_test_seq_val = torch.max(test_seq)
        # img_copy = torch.squeeze(img.detach().cpu())
        # ax4 = plt.subplot(175)
        # ax4.imshow((img_copy), cmap="jet")
        # ax4.set_title("Model Input\nstandard normal\n distributed Noise")

        for i in range(0, T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = get_sample_at_t(
                x=img,
                t=t,
                condition=cond_frames,
                betas=betas,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                sqrt_recip_alphas=sqrt_recip_alphas,
                posterior_variance=posterior_variance,
                model=model,
                cond_emb_model=cond_emb_model,
            )
            if i == 0:
                # img.shape = torch.Size([1, 1, 64, 64])
                img = torch.squeeze(img)

                ax1 = plt.subplot(165)
                img = img.detach().c5u()
                img += 1
                img /= 2
                img *= max_prec_val
                img **= 3
                max_img_val = torch.max(img)
                max_plot_val = torch.max(max_img_val, max_test_seq_val)
                im = ax1.imshow(
                    (img[pred_area_idx:-pred_area_idx, pred_area_idx:-pred_area_idx]),
                    cmap="jet",
                    vmin=0,
                    vmax=max_plot_val,
                )
                ax1.set_title(f"Model Prediction for \n{timestamp_arr[4]}")
                fig.colorbar(im)

        ax2 = plt.subplot(151)
        ax2.imshow((test_seq[0, :, :]), cmap="jet", vmin=0, vmax=max_plot_val)
        ax2.set_title(f"Model Input\n{timestamp_arr[0]}")

        ax7 = plt.subplot(152)
        ax7.imshow((test_seq[1, :, :]), cmap="jet", vmin=0, vmax=max_plot_val)
        ax7.set_title(f"Model Input\n{timestamp_arr[1]}")

        ax3 = plt.subplot(153)
        ax3.imshow((test_seq[2, :, :]), cmap="jet", vmin=0, vmax=max_plot_val)
        ax3.set_title(f"Model Input\n{timestamp_arr[2]}")

        ax5 = plt.subplot(154)
        ground_truth = lwe_data[
            pred_area_idx:-pred_area_idx, pred_area_idx:-pred_area_idx
        ]

        im2 = ax5.imshow((ground_truth), cmap="jet", vmin=0, vmax=max_plot_val)
        ax5.set_title(f"Observed at\n{timestamp_arr[4]}")
        fig.colorbar(im2)

        fig.savefig(f"{out_folder}/seq_{img_count}_epoch_{epoch}.png")
        img_count += 1

        plt.show()


def sample_preds(
    cond_frames,
    rgb_grayscale,
    img_out_size,
    sequence_length,
    device,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
    model,
    T,
    betas,
    cond_emb_model,
    numb_of_samples=5,
):

    predictions = torch.empty(
        (numb_of_samples, img_out_size, img_out_size), device=device
    )
    for sample in range(numb_of_samples):
        img_size = img_out_size

        # print(f"test_seq.shape = {test_seq.shape}")
        # print(f"test_seq.shape = {test_seq.shape}")

        # Sample noise
        img = torch.randn((1, rgb_grayscale, img_size, img_size), device=device)
        # test_seq.shape=torch.Size([5, 64, 64, 1])
        # test_seq = torch.permute(test_seq, (3,0,1,2))
        # test_seq.shape=torch.Size([1, 5, 64, 64])
        # batch_size,channels,W,H

        for i in range(0, T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = get_sample_at_t(
                x=img,
                t=t,
                condition=cond_frames,
                betas=betas,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                sqrt_recip_alphas=sqrt_recip_alphas,
                posterior_variance=posterior_variance,
                model=model,
                cond_emb_model=cond_emb_model,
            )

            if i == 0:
                # img.shape = torch.Size([1, 1, 64, 64])
                img = torch.squeeze(img)
                predictions[sample, :, :] = img

    return predictions


def sample_next_step_pred(
    cond_frames,
    rgb_grayscale,
    img_out_size,
    device,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
    model,
    T,
    betas,
    cond_emb_model,
):

    img_size = img_out_size

    # print(f"test_seq.shape = {test_seq.shape}")
    # print(f"test_seq.shape = {test_seq.shape}")

    # Sample noise
    img = torch.randn((1, rgb_grayscale, img_size, img_size), device=device)
    # test_seq.shape=torch.Size([5, 64, 64, 1])
    # test_seq = torch.permute(test_seq, (3,0,1,2))
    # test_seq.shape=torch.Size([1, 5, 64, 64])
    # batch_size,channels,W,H

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = get_sample_at_t(
            x=img,
            t=t,
            condition=cond_frames,
            betas=betas,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas=sqrt_recip_alphas,
            posterior_variance=posterior_variance,
            model=model,
            cond_emb_model=cond_emb_model,
        )

        if i == 0:
            # img.shape = torch.Size([1, 1, 64, 64])
            img = torch.squeeze(img)

    return img


def sample_next_step_pred_CSI(
    cond_frames,
    rgb_grayscale,
    img_out_size,
    device,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
    model,
    T,
    betas,
    cond_emb_model,
    sample_scheme=False,
):

    img_size = img_out_size

    # print(f"test_seq.shape = {test_seq.shape}")
    # print(f"test_seq.shape = {test_seq.shape}")

    # Sample noise
    img = torch.randn((1, rgb_grayscale, img_size, img_size), device=device)
    # test_seq.shape=torch.Size([5, 64, 64, 1])
    # test_seq = torch.permute(test_seq, (3,0,1,2))
    # test_seq.shape=torch.Size([1, 5, 64, 64])
    # batch_size,channels,W,H
    if sample_scheme:

        scheme = np.linspace(T, 0, 30)

        for i in scheme:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = get_sample_at_t(
                x=img,
                t=t,
                condition=cond_frames,
                betas=betas,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                sqrt_recip_alphas=sqrt_recip_alphas,
                posterior_variance=posterior_variance,
                model=model,
                cond_emb_model=cond_emb_model,
            )

    else:
        for i in range(0, T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = get_sample_at_t(
                x=img,
                t=t,
                condition=cond_frames,
                betas=betas,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                sqrt_recip_alphas=sqrt_recip_alphas,
                posterior_variance=posterior_variance,
                model=model,
                cond_emb_model=cond_emb_model,
            )

    return img
