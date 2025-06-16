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
    Logs predictions from the model (after flattening)

    Args:
        tensor (torch.Tensor): Shape (bs, na, ny, nx, no)
        epoch (int): Current epoch
        batch_i (int): Batch index
        output_dir (str): Directory for CSV files
        sample_prob (float): Sampling probability (e.g., 0.01 = 1%)
        col_names (list[str]): Optional, e.g., ["x", "y", "w", "h", "obj", "class_logits"]
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

def safe_read_csv(file):
    rows = []
    with open(file, "r") as f:
        for line in f:
            # Spalten durch Trennzeichen erkennen
            sep = "\t" if "\t" in line else ","
            parts = line.strip().split(sep)
            if len(parts) == 6:
                try:
                    if float(parts[0]) < 200:
                        rows.append([float(x) for x in parts])
                except ValueError:
                    continue  # überspringt Zeile mit nicht-konvertierbarem Wert
    return pd.DataFrame(rows, columns=["x", "y", "w", "h", "obj", "class_0"])


def evaluate_logs(csv_dir):
    # Alle Dateien holen
    csv_files = glob.glob(os.path.join(csv_dir, "pred_epoch*_batch*.csv"))

    # Dictionary: epoche -> DataFrame
    epoch_data = {}

    # Dateien einlesen und nach Epoche gruppieren
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

    # Alle DataFrames pro Epoche zusammenführen
    for epoch in epoch_data:
        epoch_data[epoch] = pd.concat(epoch_data[epoch], ignore_index=True)

    # Histogramme plotten
    output_dir = csv_dir.replace("/preds", "/logs")
    os.makedirs(output_dir, exist_ok=True)

    # Plot mit Subplots pro Epoche
    for epoch, df in epoch_data.items():
        print(f"Epoche {epoch}: {len(df)} gültige Einträge")

        fig, axes = plt.subplots(2, 3, figsize=(12, 10))
        fig.suptitle(f"Verteilung der Vorhersagen – Epoche {epoch}", fontsize=16)

        columns = ["x", "y", "w", "h", "obj", "class_0"]
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

    csv_dir = "../../runs/train/BOArDING_Det/preds"
    evaluate_logs(csv_dir)
