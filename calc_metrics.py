import torch
from metrics import calc_CRPS, pooling_func
import matplotlib.pyplot as plt
from sampler import sample_preds
import numpy as np


def get_CRPS(
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
    numb_cond,
    betas,
    max_prec_val,
    cond_emb_model,
    nowcast_mode=False,
):
    test_list_avg4_crps = []
    test_list_avg16_crps = []
    test_list_max16_crps = []
    test_list_max4_crps = []

    for test_sample in test_list:
        # only get LWE data, not time and convective

        cond_frames = test_sample[1]
        cond_frames = cond_frames.to(device)
        observation = test_sample[0]

        preds = sample_preds(
            cond_frames=cond_frames,
            rgb_grayscale=rgb_grayscale,
            img_out_size=img_out_size,
            sequence_length=sequence_length,
            device=device,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas=sqrt_recip_alphas,
            posterior_variance=posterior_variance,
            model=model,
            T=T,
            betas=betas,
            cond_emb_model=cond_emb_model,
        )
        # Reverste transforms to get back to orignal LWE values:
        pred_copy = preds
        pred_copy += 1
        pred_copy /= 2
        pred_copy *= max_prec_val
        pred_copy **= 3

        obs_copy = observation
        obs_copy += 1
        obs_copy /= 2
        obs_copy *= max_prec_val
        obs_copy **= 3

        avg_pooled_frames_4_4 = pooling_func(
            pred=pred_copy,
            observed=obs_copy,
            kernel_size=4,
            stride=2,
            device=device,
            pool_type="average",
        )
        avg_pooled_frames_16_16 = pooling_func(
            pred=pred_copy,
            observed=obs_copy,
            kernel_size=16,
            stride=16,
            device=device,
            pool_type="average",
        )
        max_pooled_frames_4_4 = pooling_func(
            pred=pred_copy,
            observed=obs_copy,
            kernel_size=4,
            stride=2,
            device=device,
            pool_type="max",
        )
        max_pooled_frames_16_16 = pooling_func(
            pred=pred_copy,
            observed=obs_copy,
            kernel_size=16,
            stride=16,
            device=device,
            pool_type="max",
        )

        avg_4_crps = calc_CRPS(
            preds=avg_pooled_frames_4_4[1:, :, :],
            observation=avg_pooled_frames_4_4[0, :, :],
            img_out_size=avg_pooled_frames_4_4.shape[-1],
        )
        avg_16_crps = calc_CRPS(
            preds=avg_pooled_frames_16_16[1:, :, :],
            observation=avg_pooled_frames_16_16[0, :, :],
            img_out_size=avg_pooled_frames_16_16.shape[-1],
        )
        max_4_crps = calc_CRPS(
            preds=max_pooled_frames_4_4[1:, :, :],
            observation=max_pooled_frames_4_4[0, :, :],
            img_out_size=max_pooled_frames_4_4.shape[-1],
        )
        max_16_crps = calc_CRPS(
            preds=max_pooled_frames_16_16[1:, :, :],
            observation=max_pooled_frames_16_16[0, :, :],
            img_out_size=max_pooled_frames_16_16.shape[-1],
        )

        test_list_avg4_crps.append(avg_4_crps)
        test_list_avg16_crps.append(avg_16_crps)
        test_list_max4_crps.append(max_4_crps)
        test_list_max16_crps.append(max_16_crps)

    avg4_array = np.array(test_list_avg4_crps)
    avg16_array = np.array(test_list_avg16_crps)
    max4_array = np.array(test_list_max4_crps)
    max16_array = np.array(test_list_max16_crps)

    avg4_std = np.std(avg4_array)
    avg16_std = np.std(avg16_array)
    max4_std = np.std(max4_array)
    max16_std = np.std(max16_array)

    avg4_mean = np.mean(avg4_array)
    avg16_mean = np.mean(avg16_array)
    max4_mean = np.mean(max4_array)
    max16_mean = np.mean(max16_array)

    return (
        avg4_mean,
        avg16_mean,
        max4_mean,
        max16_mean,
        avg4_std,
        avg16_std,
        max4_std,
        max16_std,
    )


def get_nowcast_20_CRPS(
    cond_frames,
    observation,
    rgb_grayscale,
    img_out_size,
    sequence_length,
    device,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
    model,
    betas,
    max_prec_val,
    numb_of_samples,
    T,
    cond_emb_model,
):

    preds = sample_preds(
        cond_frames=cond_frames,
        rgb_grayscale=rgb_grayscale,
        img_out_size=img_out_size,
        sequence_length=sequence_length,
        device=device,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas=sqrt_recip_alphas,
        posterior_variance=posterior_variance,
        model=model,
        T=T,
        betas=betas,
        numb_of_samples=numb_of_samples,
        cond_emb_model=cond_emb_model,
    )
    # Reverste transforms to get back to orignal LWE values:

    pred_copy = preds
    pred_copy += 1
    pred_copy /= 2
    pred_copy *= max_prec_val
    pred_copy **= 3

    obs_copy = observation
    obs_copy += 1
    obs_copy /= 2
    obs_copy *= max_prec_val
    obs_copy **= 3

    avg_pooled_frames_4_4 = pooling_func(
        pred=pred_copy,
        observed=obs_copy,
        kernel_size=4,
        stride=2,
        device=device,
        pool_type="average",
    )
    avg_pooled_frames_16_16 = pooling_func(
        pred=pred_copy,
        observed=obs_copy,
        kernel_size=16,
        stride=16,
        device=device,
        pool_type="average",
    )
    max_pooled_frames_4_4 = pooling_func(
        pred=pred_copy,
        observed=obs_copy,
        kernel_size=4,
        stride=2,
        device=device,
        pool_type="max",
    )
    max_pooled_frames_16_16 = pooling_func(
        pred=pred_copy,
        observed=obs_copy,
        kernel_size=16,
        stride=16,
        device=device,
        pool_type="max",
    )

    avg_4_crps = calc_CRPS(
        preds=avg_pooled_frames_4_4[1:, :, :],
        observation=avg_pooled_frames_4_4[0, :, :],
        img_out_size=avg_pooled_frames_4_4.shape[-1],
    )
    avg_16_crps = calc_CRPS(
        preds=avg_pooled_frames_16_16[1:, :, :],
        observation=avg_pooled_frames_16_16[0, :, :],
        img_out_size=avg_pooled_frames_16_16.shape[-1],
    )
    max_4_crps = calc_CRPS(
        preds=max_pooled_frames_4_4[1:, :, :],
        observation=max_pooled_frames_4_4[0, :, :],
        img_out_size=max_pooled_frames_4_4.shape[-1],
    )
    max_16_crps = calc_CRPS(
        preds=max_pooled_frames_16_16[1:, :, :],
        observation=max_pooled_frames_16_16[0, :, :],
        img_out_size=max_pooled_frames_16_16.shape[-1],
    )

    return avg_4_crps, avg_16_crps, max_4_crps, max_16_crps

