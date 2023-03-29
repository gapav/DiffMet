from fdp import fdp_sample
import torch.nn.functional as F


def loss_fn(
    model,
    x,
    t,
    sqrt_one_minus_alphas_cumprod,
    sqrt_alphas_cumprod,
    device,
    cond_emb_model,
    pred_ahead=False,
    condition=None,
    loss="mse",
):
    """
    Inspired by https://huggingface.co/blog/annotated-diffusion, with several
    extensions and modifications. 
    
    Method for calculating loss between the actual loss and the 
    model predicted loss. Can either be L1 or MSE loss.Added functionality
    for not calculating on the image borders. 
    """
    noised_img, epsilon = fdp_sample(
        x, t, sqrt_one_minus_alphas_cumprod, sqrt_alphas_cumprod
    )

    noise_pred = model(
        x=noised_img, timestep=t, condition=condition, cond_emb_model=cond_emb_model
    )
    if pred_ahead:
        pred_area_idx = pred_ahead * 3
        epsilon = epsilon[
            :, :, pred_area_idx:-pred_area_idx, pred_area_idx:-pred_area_idx
        ]
        noise_pred = noise_pred[
            :, :, pred_area_idx:-pred_area_idx, pred_area_idx:-pred_area_idx
        ]
    epsilon = epsilon.to(device)
    noise_pred = noise_pred.to(device)
    if loss == "f1_loss":
        return F.l1_loss(epsilon, noise_pred)
    if loss == "mse":
        return F.mse_loss(input=noise_pred, target=epsilon)
