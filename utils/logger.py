import os
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import dataset


def log_predictions(tensor, epoch, batch_i, output_dir, sample_prob=0.001, col_names=None):
    """
    Logs predictions from the model (after flattening)

    Args:
        tensor (torch.Tensor): Shape (bs, na, ny, nx, no)
        epoch (int): Current epoch
        batch_i (int): Batch index
        output_dir (str): Directory for CSV files
        sample_prob (float): Sampling probability (e.g., 0.01 = 1%)
        col_names (list[str]): Optional, e.g., ["x", "y", "w", "h", "obj", "class_logits", "distance", "heading"]
    """

    os.makedirs(output_dir, exist_ok=True)

    if epoch % 10 == 0 and batch_i % 4 == 0:
        # bs = batch size, na = num
        bs, na, ny, nx, no = tensor.shape
        flat = tensor.view(bs * na * ny * nx, no)

        # random sample (1 %)
        mask = torch.rand(flat.shape[0]) < sample_prob
        sampled = flat[mask]

        if sampled.numel() == 0:
            return

        np_data = sampled.detach().cpu().numpy()

        # name CSV-file
        #now = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"pred_epoch{epoch}_batch{batch_i}.csv"
        fpath = os.path.join(output_dir, fname)

        # save
        header = ",".join(col_names) if col_names else None
        np.savetxt(fpath, np_data, delimiter=",", header=header if header else "", comments="")
        print(f"saved predictions to {fpath}")