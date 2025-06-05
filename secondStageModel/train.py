import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from secondStageModel.crop_dataloader import ObjectCropDataset
from secondStageModel.crop_regressor import CropRegressor


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

    return total_loss / len(val_loader.dataset)


def train(train_dataset, val_dataset):

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    model = CropRegressor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    best_model_path = "best_model.pth"

    for epoch in range(20):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")


if __name__ == '__main__':

    # Label Datei muss to aussehen:
    #image_0001.jpg
    # 0 0.12 0.25 0.05 0.05 300.0 0.87 0.5
    # 0 0.45 0.33 0.07 0.07 400.0 0.0 0.0
    #image_0002.jpg
    #...

    dataset_train = ObjectCropDataset(
        label_dir="train/labels/",
        image_dir="train/images/",
        image_size=(640, 480),
        require_heading=True
    )

    dataset_val = ObjectCropDataset(
        label_dir="val/labels/",
        image_dir="val/images/",
        image_size=(640, 480),
        require_heading=True
    )

    crop, target = dataset_train[0]
    print(crop.shape)  # (3, 64, 64)
    print(target)  # Tensor mit [distance, cos, sin]

    train(dataset_train, dataset_val)