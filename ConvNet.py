from torch import nn


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.stacks = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.stacks(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class ConvolutionalNetwork(nn.Module):
    def __init__(self, num_units=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),

            ConvUnit(16, 32, downsample=True),
            ConvUnit(32, 32),
            ConvUnit(32, 64, downsample=True),

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
