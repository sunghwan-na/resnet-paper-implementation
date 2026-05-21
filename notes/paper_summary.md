# ResNet Paper Summary

## Paper
Deep Residual Learning for Image Recognition

## Core Problem
As neural networks become deeper, training becomes harder and performance can degrade.

## Main Idea
ResNet uses shortcut connections to learn residual functions.

Instead of learning:
H(x)

ResNet learns:
F(x) = H(x) - x

So the output becomes:
F(x) + x

## Why It Matters
Shortcut connections make it easier to train deeper networks.

## Implementation Goal
- Implement PlainNet and ResNet for CIFAR-10
- Compare PlainNet-20 / PlainNet-56 with ResNet-20 / ResNet-56
- Analyze loss and accuracy curves