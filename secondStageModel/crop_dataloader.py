import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import os

class ObjectCropDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_size=None, transform=None, require_heading=False):
        """
        image_dir: directory of images
        label_dir: directory of label files
        image_size: original image size (width, height) or None to read from image
        require_heading: if True, only use entries with valid heading (cos, sin not both zero)
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform or T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        self.data_list = []

        for img_name in os.listdir(image_dir):
            if not img_name.lower().endswith((".jpg", ".png")):
                continue

            label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                items = line.strip().split()
                if len(items) != 8:
                    continue

                cls, x, y, w, h, dist, cos, sin = map(float, items)
                if dist == -1:
                    continue
                if require_heading and (cos == 0.0 and sin == 0.0):
                    continue

                self.data_list.append({
                    "image_name": img_name,
                    "bbox_norm": [x, y, w, h],
                    "distance": dist,
                    "cos": cos,
                    "sin": sin
                })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        entry = self.data_list[idx]
        img_path = os.path.join(self.image_dir, entry["image_name"])
        image = Image.open(img_path).convert("RGB")
        image_name = entry["image_name"]

        if self.image_size is None:
            img_w, img_h = image.size
        else:
            img_w, img_h = self.image_size

        x, y, w, h = entry["bbox_norm"]
        cx, cy = x * img_w, y * img_h
        bw, bh = w * img_w, h * img_h
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

        crop = image.crop((x1, y1, x2, y2))
        crop = self.transform(crop)

        # Output target: distance, heading (optional)
        distance = entry["distance"]
        cos = entry["cos"]
        sin = entry["sin"]

        distance_norm = min(distance, 1000.0) / 1000.0

        target = torch.tensor([distance_norm, cos, sin], dtype=torch.float32)

        return crop, target, image_name