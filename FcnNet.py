from torch import nn

from ConvNet import Flatten


class FullyConnectedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            Flatten(),
            nn.Linear(10000, 5000),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Linear(5000, 64),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Linear(64, 13),

        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight = nn.init.constant_(m.weight, 1)
                m.bias = nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)
