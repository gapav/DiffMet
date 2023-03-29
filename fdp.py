import numpy as np
import math
import torch


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    
    https://github.com/openai/improved-diffusion
    
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":

        return torch.linspace(start=0.0001, end=0.02, steps=num_diffusion_timesteps)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    
    https://github.com/openai/improved-diffusion
    
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.Tensor(betas)


def extract_index(vals, t, x_shape):
    """ 
    based on https://huggingface.co/blog/annotated-diffusion

    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def fdp_sample(
    x_0, t, sqrt_one_minus_alphas_cumprod, sqrt_alphas_cumprod, device="cpu"
):
    """ 
    based on https://huggingface.co/blog/annotated-diffusion

    """

    eps = torch.randn_like(x_0)

    # get cumulative product of alphas:
    batch_size = t.shape[0]
    out_sqrt_a_cmprod = sqrt_alphas_cumprod.gather(-1, t.cpu())
    out_sqrt_one_min_a_cmprod = sqrt_one_minus_alphas_cumprod.gather(-1, t.cpu())
    sqrt_alphas_cumprod_t = out_sqrt_a_cmprod.reshape(
        batch_size, *((1,) * (len(x_0.shape) - 1))
    ).to(t.device)
    sqrt_one_minus_alphas_cumprod_t = out_sqrt_one_min_a_cmprod.reshape(
        batch_size, *((1,) * (len(x_0.shape) - 1))
    ).to(t.device)

    # return equation 4.4 from diffusion chapter in thesis:
    # q(x_t | x_{t-1})
    return (
        sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
        + sqrt_one_minus_alphas_cumprod_t.to(device) * eps.to(device),
        eps.to(device),
    )

