import os
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import dataset


def log_predictions(tensor, epoch, batch_i, output_dir, sample_prob=0.001, col_names=None):
    """
    Loggt Vorhersagen aus dem Model (nach flatten), z.B. distance und heading.

    Args:
        tensor (torch.Tensor): Shape (bs, na, ny, nx, no)
        epoch (int): aktuelle Epoche
        batch_i (int): Batch-Index
        output_dir (str): Verzeichnis für CSV-Dateien
        sample_prob (float): Wahrscheinlichkeit für Sampling (z.B. 0.01 = 1 %)
        col_names (list[str]): Optional, z.B. ["x", "y", "w", "h", "obj", "class_logits", "distance", "heading"]
    """
    os.makedirs(output_dir, exist_ok=True)

    if epoch % 10 == 0 and batch_i % 4 == 0:
        bs, na, ny, nx, no = tensor.shape
        flat = tensor.view(bs * na * ny * nx, no)

        # Zufällig sampeln (1 %)
        mask = torch.rand(flat.shape[0]) < sample_prob
        sampled = flat[mask]

        if sampled.numel() == 0:
            return

        np_data = sampled.detach().cpu().numpy()

        # CSV-Dateiname
        #now = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"pred_epoch{epoch}_batch{batch_i}.csv"
        fpath = os.path.join(output_dir, fname)

        # Speichern
        header = ",".join(col_names) if col_names else None
        np.savetxt(fpath, np_data, delimiter=",", header=header if header else "", comments="")
        print(f"saved predictions to {fpath}")