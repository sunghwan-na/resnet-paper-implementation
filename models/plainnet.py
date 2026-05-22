import torch
import torch.nn as nn

class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PlainBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
    
class PlainNetCIFAR(nn.Module):
    """
    CIFAR-10용 PlainNet 구조

    ResNet과 같은 깊이와 채널 구조를 사용하지만,
    shortcut connection은 사용하지 않는다.
    """

    def __init__(self, depth=20, num_classes=10):
        super(PlainNetCIFAR, self).__init__()

        assert (depth - 2) % 6 == 0, "Depth should be 6n + 2."
        n = (depth - 2) // 6

        self.in_channels = 16

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(out_channels=16, num_blocks=n, stride=1)
        self.layer2 = self._make_layer(out_channels=32, num_blocks=n, stride=2)
        self.layer3 = self._make_layer(out_channels=64, num_blocks=n, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []

        strides = [stride] + [1] * (num_blocks - 1)

        for stride in strides:
            layers.append(
                PlainBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride
                )
            )
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
    
def plainnet20(num_classes=10):
    return PlainNetCIFAR(depth=20, num_classes=num_classes)

def plainnet32(num_classes=10):
    return PlainNetCIFAR(depth=32, num_classes=num_classes)

def plainnet44(num_classes=10):
    return PlainNetCIFAR(depth=44, num_classes=num_classes)

def plainnet56(num_classes=10):
    return PlainNetCIFAR(depth=56, num_classes=num_classes)