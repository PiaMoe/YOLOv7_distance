import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from secondStageModel.crop_dataloader import ObjectCropDataset
from secondStageModel.crop_regressor import CropRegressor
import torchvision.utils as vutils
import os
import torchvision.transforms.functional as TF
import wandb
import numpy as np

def custom_loss(outputs, targets):
    # outputs: [batch, 3] -> [distance_norm, cos, sin]
    # targets: [batch, 3] -> [distance_norm, cos, sin]
    distance_pred = outputs[:, 0]  # Normalized distance
    distance_true = targets[:, 0]  # Normalized distance
    distance_loss = nn.functional.mse_loss(distance_pred, distance_true)
    heading_pred = outputs[:, 1:3]  # [cos, sin]
    heading_true = targets[:, 1:3]  # [cos, sin]
    cosine_sim = nn.functional.cosine_similarity(heading_pred, heading_true, dim=1)
    heading_loss = 1 - cosine_sim.mean()
    total_loss = distance_loss + heading_loss
    return total_loss, distance_loss, heading_loss

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_distance_loss = 0.0
    total_heading_loss = 0.0
    angle_errors = []
    abs_distance_errors = []
    abs_distance_errors_bins = [[] for _ in range(5)]
    bin_edges = [0, 200, 400, 600, 800, 1000]
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss, distance_loss, heading_loss = custom_loss(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_distance_loss += distance_loss.item() * inputs.size(0)
            total_heading_loss += heading_loss.item() * inputs.size(0)

            # Winkel aus cos/sin berechnen (in Grad)
            pred_cos = outputs[:, 1].cpu().numpy()
            pred_sin = outputs[:, 2].cpu().numpy()
            target_cos = targets[:, 1].cpu().numpy()
            target_sin = targets[:, 2].cpu().numpy()
            pred_angle = np.arctan2(pred_sin, pred_cos) * 180 / np.pi
            target_angle = np.arctan2(target_sin, target_cos) * 180 / np.pi
            angle_diff = np.abs(pred_angle - target_angle)
            angle_diff = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)
            angle_errors.extend(angle_diff.tolist())

            # Distance denormalisieren für Fehlerberechnung
            pred_dist = outputs[:, 0].cpu().numpy() * 1000.0
            target_dist = targets[:, 0].cpu().numpy() * 1000.0
            abs_dist_err = np.abs(pred_dist - target_dist)
            abs_distance_errors.extend(abs_dist_err.tolist())
            # Bin Zuordnung
            for d, err in zip(target_dist, abs_dist_err):
                for i in range(5):
                    if bin_edges[i] <= d < bin_edges[i+1]:
                        abs_distance_errors_bins[i].append(err)
                        break

    n = len(val_loader.dataset)
    mean_angle_error = np.mean(angle_errors)
    mean_abs_distance_error = np.mean(abs_distance_errors)
    mean_abs_distance_error_bins = [
        np.mean(bin) if len(bin) > 0 else float('nan') for bin in abs_distance_errors_bins
    ]
    bin_counts = [len(bin) for bin in abs_distance_errors_bins]
    print("Number of examples per bin (0-200, 200-400, 400-600, 600-800, 800-1000):", bin_counts)
    return (
        total_loss / n,
        total_distance_loss / n,
        total_heading_loss / n,
        mean_angle_error,
        mean_abs_distance_error,
        mean_abs_distance_error_bins
    )

def train(train_dataset, val_dataset, epochs=20):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    model = CropRegressor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    best_model_path = "experiment_1/best.pth"

    crops_saved = 0
    max_crops_to_save = 0

    wandb.init(project="crop_regression", name="experiment_1")
    wandb.watch(model)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_distance_loss = 0.0
        running_heading_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Speichere Crops mit Bildnamen und Objektindex im Dateinamen
            if crops_saved < max_crops_to_save and epoch == 0:
                os.makedirs("debug_crops", exist_ok=True)
                batch_start = batch_idx * train_loader.batch_size
                for i in range(inputs.size(0)):
                    if crops_saved >= max_crops_to_save:
                        break
                    dataset_idx = batch_start + i
                    if dataset_idx >= len(train_dataset):
                        break
                    img_name = train_dataset.data_list[dataset_idx]["image_name"]
                    crop = inputs[i].cpu()
                    # Unnormalize: x = x * std + mean
                    crop_unnorm = crop * 0.5 + 0.5
                    vutils.save_image(
                        crop_unnorm,
                        f"debug_crops/crop_{crops_saved}_{os.path.splitext(img_name)[0]}_{dataset_idx}.png"
                    )
                    crops_saved += 1

            optimizer.zero_grad()
            outputs = model(inputs)
            loss, distance_loss, heading_loss = custom_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_distance_loss += distance_loss.item() * inputs.size(0)
            running_heading_loss += heading_loss.item() * inputs.size(0)

            # Log batch losses
            #wandb.log({
            #    "batch_loss": loss.item(),
            #    "batch_distance_loss": distance_loss.item(),
            #    "batch_heading_loss": heading_loss.item()
            #})

        n = len(train_loader.dataset)
        epoch_train_loss = running_loss / n
        epoch_train_distance_loss = running_distance_loss / n
        epoch_train_heading_loss = running_heading_loss / n

        (epoch_val_loss, epoch_val_distance_loss, epoch_val_heading_loss,
         epoch_val_angle_error, epoch_val_abs_distance_error, epoch_val_abs_distance_error_bins) = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, "
            f"\nVal Loss: {epoch_val_loss:.4f}, "
            f"\nTrain Distance Loss: {epoch_train_distance_loss:.4f}, "
            f"\nVal Distance Loss: {epoch_val_distance_loss:.4f}, "
            f"\nTrain Heading Loss: {epoch_train_heading_loss:.4f}, "
            f"\nVal Heading Loss: {epoch_val_heading_loss:.4f}, "
            f"\nVal Mean Angle Error: {epoch_val_angle_error:.2f} deg, "
            f"\nVal Mean Abs Distance Error: {epoch_val_abs_distance_error:.2f} m"
        )
        print("Val Mean Abs Distance Error Bins (0-200, 200-400, 400-600, 600-800, 800-1000):", epoch_val_abs_distance_error_bins)

        # Log train losses
        wandb.log({
            "train/train_loss": epoch_train_loss,
            "train/train_distance_loss": epoch_train_distance_loss,
            "train/train_heading_loss": epoch_train_heading_loss,
        })

        # Log val losses
        wandb.log({
            "val/val_loss": epoch_val_loss,
            "val/val_distance_loss": epoch_val_distance_loss,
            "val/val_heading_loss": epoch_val_heading_loss,
        })

        # Log val errors
        wandb.log({
            "val/val_mean_angle_error_deg": epoch_val_angle_error,
            "val/val_mean_abs_distance_error": epoch_val_abs_distance_error,
            "val/val_mean_abs_distance_error_bin_0_200": epoch_val_abs_distance_error_bins[0],
            "val/val_mean_abs_distance_error_bin_200_400": epoch_val_abs_distance_error_bins[1],
            "val/val_mean_abs_distance_error_bin_400_600": epoch_val_abs_distance_error_bins[2],
            "val/val_mean_abs_distance_error_bin_600_800": epoch_val_abs_distance_error_bins[3],
            "val/val_mean_abs_distance_error_bin_800_1000": epoch_val_abs_distance_error_bins[4],
        })

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")

    wandb.finish()

if __name__ == '__main__':

    dataset_path = "/home/pmoessner/data/BOArDING_Dataset/BOArDING_cos_sin/"

    dataset_train = ObjectCropDataset(
        label_dir= dataset_path + "train/labels/",
        image_dir= dataset_path + "train/images/",
        image_size=None,
        require_heading=True
    )

    dataset_val = ObjectCropDataset(
        label_dir= dataset_path + "val/labels/",
        image_dir= dataset_path + "val/images/",
        image_size=None,
        require_heading=True
    )

    crop, target = dataset_train[0]
    print(crop.shape)  # (3, 64, 64)
    print(target)  # Tensor with [distance, cos, sin]

    train(dataset_train, dataset_val, 100)