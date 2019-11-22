import torch.nn.functional as F
from torch import nn


class Shortcut(nn.Module):
    def __init__(self, downsample=False):
        super().__init__()
        self.downsample = downsample

    def forward(self, x):
        if self.downsample:
            return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, x.shape[1] // 2, x.shape[1] // 2), "constant", 0)

        else:
            return x


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.shortcut = Shortcut(downsample)
        stride = 1 if not downsample else 2

        self.stacks = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.stacks(x) + self.shortcut(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_units, downsample=False):
        super().__init__()
        self.units = nn.Sequential(
            ResidualUnit(in_channels, out_channels, downsample),
            *[ResidualUnit(out_channels, out_channels) for _ in range(num_units - 1)],
        )

    def forward(self, x):
        return self.units(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class ResidualNetwork(nn.Module):
    def __init__(self, num_units=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),

            ResidualBlock(16, 16, num_units),
            ResidualBlock(16, 32, num_units, downsample=True),
            ResidualBlock(32, 64, num_units, downsample=True),

            nn.AdaptiveAvgPool2d(1),
            Flatten(),
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
