import torch
from torch import nn

class ModelV1(nn.Module):
    def __init__(self, in_shape, hidden, n_classes):
        super().__init__()
        # VGG like architecture
        block_1 = nn.Sequential(
            nn.Conv2d(in_shape, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        block_2 = nn.Sequential(
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        block_3 = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        block_4 = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden * 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.blocks = nn.Sequential(
            block_1,
            block_2,
            block_3,
            block_4
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 8 * 4 * 4, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, n_classes)
        )
    def forward(self, x):
        x = self.blocks(x)
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=1)
