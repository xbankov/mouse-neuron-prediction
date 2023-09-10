import torch.nn as nn
import torch.nn.functional as F


class PointCloudNet(nn.Module):
    def __init__(self, num_classes, num_neurons, channels=8, num_layers=4):
        super(PointCloudNet, self).__init__()
        # Convolutional layers

        self.num_layers = num_layers
        self.channels = channels
        self.num_neurons = num_neurons

        self.conv1 = nn.Conv1d(4, 8 * channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(8 * channels, 16 * channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(16 * channels, 32 * channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(32 * channels, 64 * channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * channels * (self.num_neurons // (2 ** self.num_layers)), 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)  # Adjust the dropout rate as needed

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
        x = self.dropout(x)
        x = self.fc2(x)

        # Softmax activation returning probabilities for the class
        logits = self.softmax(x)
        return logits


class LinearPointCloudNet(nn.Module):
    def __init__(self, num_classes, num_neurons, num_layers=4):
        super(LinearPointCloudNet, self).__init__()
        # Convolutional layers

        self.num_layers = num_layers
        self.num_neurons = num_neurons

        self.fc1 = nn.Linear(num_neurons * 4, 2048)
        self.fc2 = nn.Linear(2048, 1028)
        self.fc3 = nn.Linear(1028, 512)
        self.fc4 = nn.Linear(512, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)  # Adjust the dropout rate as needed

    def forward(self, x):
        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        # Softmax activation returning probabilities for the class
        logits = self.softmax(x)
        return logits
