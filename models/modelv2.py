import torch
from torch import nn

class ModelV2(nn.Module):

    def __init__(self, in_shape, hidden, n_classes):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_shape, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden * 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(hidden * 8, hidden * 4, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 4 * 4 * 4, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, n_classes)
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=1)