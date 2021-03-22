#!/usr/bin/env python3
"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Courtesy of https://github.com/kuangliu/pytorch-cifar
"""
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.uniform_(self.weight, 0, 1)
        #nn.init.ones_(self.weight)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = MyBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = MyBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                MyBatchNorm2d(self.expansion*planes)
            )
        for name, mod in self.named_modules():
            if "batchnorm" in name.lower():
                nn.init.uniform_(mod.weight)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = MyBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = MyBatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = MyBatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                MyBatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def update_shape(shape, out_channels, stride):
    return [
        out_channels,
        int(ceil(shape[1] / stride)),
        int(ceil(shape[2] / stride)),
    ]


class ResNet(nn.Module):
    def __init__(
        self,
        input_shape,
        block,
        num_blocks,
        channels=None,
        strides=None,
        num_classes=None
    ):
        super(ResNet, self).__init__()
        channels = channels or [64, 128, 256, 512]
        strides = strides or [1, 2, 2, 2]
        self.in_planes = channels[0]
        shape = list(input_shape)
        self.conv1 = nn.Conv2d(shape[0], channels[0], kernel_size=3,
                               stride=strides[0], padding=1, bias=False)
        shape = update_shape(shape, channels[0], strides[0])
        self.bn1 = MyBatchNorm2d(channels[0])
        for i in range(4):
            layer = self._make_layer(
                block,
                channels[i],
                num_blocks[i],
                stride=strides[i]
            )
            setattr(self, f"layer{i+1}", layer)
            tot_stride = strides[i]
            tot_channels = channels[i]*block.expansion
            shape = update_shape(shape, tot_channels, tot_stride)
        # Avg pool
        shape[1] //= 4
        shape[2] //= 4
        self.hidden_size = shape[0] * shape[1] * shape[2]
        if num_classes is None:
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(
                channels[3]*block.expansion,
                num_classes
            )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNetS(input_shape, num_classes=None):
    return ResNet(
        input_shape,
        BasicBlock,
        [2, 2, 2, 2],
        [20, 40, 80, 160],
        [1, 2, 2, 2],
        num_classes=num_classes
    )


def ResNet18(input_shape, num_classes=None):
    return ResNet(
        input_shape,
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
    )


def ResNet34(input_shape, num_classes=None):
    return ResNet(
        input_shape,
        BasicBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
    )


def ResNet50(input_shape, num_classes=None):
    return ResNet(
        input_shape,
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
    )


def ResNet101(input_shape, num_classes=None):
    return ResNet(
        input_shape,
        Bottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
    )


def ResNet152(input_shape, num_classes=None):
    return ResNet(
        input_shape,
        Bottleneck,
        [3, 8, 36, 3],
        num_classes=num_classes,
    )


def test():
    for arch in [ResNetS, ResNet152, ResNet50]:
        net = arch((3, 32, 32))
        y = net(torch.randn(1, 3, 32, 32))
        print(net.hidden_size, y.size())


if __name__ == "__main__":
    test()
