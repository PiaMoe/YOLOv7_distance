import os
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import dataset
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re


def log_predictions(tensor, epoch, batch_i, output_dir, sample_prob=0.001, col_names=None):
    """
    Logs predictions from the model (after flattening), appending values to one file per (epoch, batch),
    adding missing columns as needed and aligning by row index.

    Args:
        tensor (torch.Tensor): Shape (bs, na, ny, nx, no)
        epoch (int): Current epoch
        batch_i (int): Batch index
        output_dir (str): Directory for CSV files
        sample_prob (float): Sampling probability (e.g., 0.01 = 1%)
        col_names (list[str]): Names of the tensor's last-dim columns
    """
    os.makedirs(output_dir, exist_ok=True)

    if epoch % 10 != 0 or batch_i % 10 != 0:
        return

    bs, na, ny, nx, no = tensor.shape
    flat = tensor.view(bs * na * ny * nx, no)

    mask = torch.rand(flat.shape[0]) < sample_prob
    sampled = flat[mask]

    if sampled.numel() == 0:
        return

    np_data = sampled.detach().cpu().numpy()
    df_new = pd.DataFrame(np_data, columns=col_names[:np_data.shape[1]])

    fname = f"pred_epoch{epoch}_batch{batch_i}.csv"
    fpath = os.path.join(output_dir, fname)

    if os.path.exists(fpath):
        df_existing = pd.read_csv(fpath)

        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing[col] = np.nan

        for col in df_existing.columns:
            if col not in df_new.columns:
                df_new[col] = np.nan

        df_new = df_new[df_existing.columns]

        # if different number of rows, pad the shorter one with NaNs
        if len(df_new) < len(df_existing):
            padding = pd.DataFrame(np.nan, index=range(len(df_existing) - len(df_new)), columns=df_existing.columns)
            df_new = pd.concat([df_new, padding], ignore_index=True)
        elif len(df_new) > len(df_existing):
            padding = pd.DataFrame(np.nan, index=range(len(df_new) - len(df_existing)), columns=df_existing.columns)
            df_existing = pd.concat([df_existing, padding], ignore_index=True)

        df_combined = df_existing.combine_first(df_new)
        df_combined.update(df_new)
        df_combined.to_csv(fpath, index=False)
    else:
        df_new.to_csv(fpath, index=False)

    print(f"[log_predictions] Updated predictions in {fpath}")


def safe_read_csv(file):
    rows = []
    with open(file, "r") as f:
        for line in f:
            sep = "\t" if "\t" in line else ","
            parts = line.strip().split(sep)
            if len(parts) == 9:
                try:
                    if float(parts[0]) < 200:
                        rows.append([float(x) for x in parts])
                except ValueError:
                    continue
    return pd.DataFrame(rows, columns=["x", "y", "w", "h", "obj", "class_0", "distance", "cosH", "sinH"])


def evaluate_logs(csv_dir):
    # take all csv files in the directory
    csv_files = glob.glob(os.path.join(csv_dir, "pred_epoch*_batch*.csv"))

    # Dictionary: epoch -> DataFrame
    epoch_data = {}

    # group files by epoch
    for file in csv_files:
        filename = os.path.basename(file)
        match = re.match(r"pred_epoch(\d+)_batch\d+\.csv", filename)
        if match:
            epoch = int(match.group(1))
            df = safe_read_csv(file)
            if not df.empty:
                if epoch not in epoch_data:
                    epoch_data[epoch] = []
                epoch_data[epoch].append(df)

    for epoch in epoch_data:
        epoch_data[epoch] = pd.concat(epoch_data[epoch], ignore_index=True)

    output_dir = csv_dir.replace("/preds", "/logs")
    os.makedirs(output_dir, exist_ok=True)

    for epoch, df in epoch_data.items():
        print(f"Epoche {epoch}: {len(df)} gültige Einträge")

        fig, axes = plt.subplots(3, 3, figsize=(16, 8))
        fig.suptitle(f"Verteilung der Vorhersagen – Epoche {epoch}", fontsize=16)

        columns = ["x", "y", "w", "h", "obj", "class_0", "distance", "cosH", "sinH"]
        for i, column in enumerate(columns):
            ax = axes[i // 3, i % 3]
            df[column].hist(bins=50, ax=ax)
            ax.set_title(column)
            ax.set_xlabel(column)
            ax.set_ylabel("Anzahl")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Platz für Titel lassen
        plt.savefig(os.path.join(output_dir, f"epoch{epoch}_all_columns.png"))
        plt.close()


if __name__ == "__main__":

    csv_dir = "../../runs/train/BOArDING_sc_aug/preds"
    evaluate_logs(csv_dir)

