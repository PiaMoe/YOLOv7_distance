import torch.nn as nn
import torch

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
        # Distance sigmoid, heading bleibt unverändert
        x[:, 0] = self.sigmoid(x[:, 0])
        return x
