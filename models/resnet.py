import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

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

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNetCIFAR(nn.Module):
    """
    CIFAR-10용 ResNet 구조

    ResNet 논문의 CIFAR 구조는 depth = 6n + 2 형태를 따른다.
    예:
    ResNet-20 -> n = 3
    ResNet-32 -> n = 5
    ResNet-56 -> n = 9
    """

    def __init__(self, depth=20, num_classes=10):
        super(ResNetCIFAR, self).__init__()

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
                BasicBlock(
                    in_channels = self.in_channels,
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

def resnet20(num_classes=10):
    return ResNetCIFAR(depth=20, num_classes=num_classes)

def resnet32(num_classes=10):
    return ResNetCIFAR(depth=32, num_classes=num_classes)

def resnet44(num_classes=10):
    return ResNetCIFAR(depth=44, num_classes=num_classes)

def resnet56(num_classes=10):
    return ResNetCIFAR(depth=56, num_classes=num_classes)