import numpy as np
import CRPS.CRPS as pscore
import torch


def calc_CRPS(preds, observation, img_out_size):

    crps_matrix = np.zeros((img_out_size, img_out_size))

    preds_arr = preds.cpu().numpy()

    observation_arr = observation.cpu().numpy()

    for row in range(img_out_size):
        for col in range(img_out_size):
            crps, _, _ = pscore(
                ensemble_members=preds_arr[:, row, col],
                observation=observation_arr[row, col],
            ).compute()
            crps_matrix[row, col] = crps

    return np.mean(crps_matrix)


def pooling_func(
    pred: torch.Tensor,
    observed: torch.Tensor,
    kernel_size: int,
    stride: int,
    device,
    pool_type: str,
) -> tuple:

    if pool_type == "average":
        pooling = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    elif pool_type == "max":
        pooling = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    else:
        raise SystemError("Wrong input to pool_type")

    pooled_observed = pooling(observed)
    # size = (preds + observed, img_size,img_size)
    pooled_output_tensor = torch.empty(
        (pred.shape[0] + 1, pooled_observed.shape[-1], pooled_observed.shape[-1])
    ).to(device)

    for i in range(pred.shape[0]):
        pred_frame = pred[i, :, :]
        pred_frame = pred_frame[None, None, :, :]
        pooled_pred_frame = pooling(pred_frame)

        pooled_output_tensor[i + 1, :, :] = pooled_pred_frame

    pooled_output_tensor[0, :, :] = pooled_observed

    return pooled_output_tensor


def crps_ensemble(predictions, observed):
    "C++ function from JB"
    n, m = predictions.shape[0]
    crps1, crps2 = 0
    crps = np.zeros(n)

    for i in range(n):
        crps1 = 0
        for j in range(m):
            crps1 += abs(predictions[i, j] - observed[i])
        crps2 = 0
        for j in range(m):
            for k in range(m):
                crps2 += abs(predictions[i, j] - predictions[i, k])
        crps[i] = crps1 / m - 0.5 * crps2 / (m * m)

    return crps


def csi(predictions,observed, t):
    
    flat_preds = torch.flatten(predictions)
    flat_obs = torch.flatten(observed)
    
    #get : greater or equal
    F_geq_t = torch.where(flat_preds>=t,1,0)
    O_geq_t = torch.where(flat_obs >= t,1,0)
    O_les_t = torch.where(flat_obs < t, 1,0)
    F_les_t = torch.where(flat_preds < t, 1,0)
    
    tp = torch.sum(torch.logical_and(F_geq_t == 1, O_geq_t == 1))
    fp = torch.sum(torch.logical_and(F_geq_t == 1, O_les_t == 1))
    fn = torch.sum(torch.logical_and(F_les_t == 1, O_geq_t == 1))

    csi = tp / (tp+fp+fn)
    return csi
    
