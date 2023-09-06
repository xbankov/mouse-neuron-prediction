import torch.nn as nn
import torch.nn.functional as F


class PointCloudNet(nn.Module):
    def __init__(self, num_classes, num_neurons):
        super(PointCloudNet, self).__init__()
        # Convolutional layers

        self.num_layers = 4
        self.num_neurons = num_neurons
        self.conv1 = nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * (self.num_neurons // (2 ** self.num_layers)), 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.squeeze().permute(0, 2, 1)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Softmax activation returning probabilities for the class
        logits = self.softmax(x)
        return logits
