import torch.nn as nn

from config import NUM_CLASSES


#model architecture
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.fc1   = nn.Linear(64 * 7 * 7, 512)
        self.fc2   = nn.Linear(512, num_classes)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x, _ = self.pool1(x)               #discard indices, keep grad support
        x = self.relu(self.conv2(x))
        x, _ = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
