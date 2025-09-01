import torch.nn as nn
import torch
from torchvision.models import resnet18, mobilenet_v2

class CropRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # -> [16, 32, 32]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> [32, 16, 16]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # -> [32, 1, 1]
        )
        self.fc = nn.Sequential(
            nn.Flatten(),          # -> [32]
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 3)       # Output: [distance, cos(heading), sin(heading)]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        # Distance sigmoid, heading is unchanged
        x[:, 0] = self.sigmoid(x[:, 0])
        return x

class ResNetCustomOutput(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # ohne fc
        self.fc = nn.Linear(base_model.fc.in_features, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x[:, 0] = self.sigmoid(x[:, 0])
        return x

class MobileNetV2CustomOutput(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = mobilenet_v2(pretrained=True)
        self.backbone = base_model.features  # feature extractor
        self.pool = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.fc = nn.Linear(base_model.last_channel, 3)  # output 3 values
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x[:, 0] = self.sigmoid(x[:, 0])  # apply sigmoid to the first output (e.g. distance)
        return x
