# ResNet Paper Implementation

PyTorch implementation and reproduction project of the ResNet paper on CIFAR-10.

This repository is based on the paper:

**Deep Residual Learning for Image Recognition**  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
Paper: https://arxiv.org/abs/1512.03385

## Project Goal

The goal of this project is to understand and reproduce the core idea of ResNet through CIFAR-10 experiments.

This project focuses on comparing:

- PlainNet: CNN without shortcut connections
- ResNet: CNN with residual shortcut connections

The main idea is to observe how residual connections help deeper neural networks train more effectively.

## Implemented Models

Currently implemented:

- PlainNet-20
- PlainNet-56
- ResNet-20
- ResNet-56

## Core Concept

A normal CNN block learns a direct mapping:

```text
H(x)
```

ResNet instead learns a residual function:

```text
F(x) = H(x) - x
```

So the final block output becomes:

```text
F(x) + x
```

This shortcut connection helps deeper networks train more stably.

## Project Structure

```text
resnet-paper-implementation/
│
├── models/
│   ├── resnet.py
│   └── plainnet.py
│
├── notes/
│   └── paper_summary.md
│
├── results/
│   ├── resnet20_history.csv
│   ├── resnet20_loss_curve.png
│   └── resnet20_accuracy_curve.png
│
├── train.py
├── plot_results.py
├── test.py
├── utils.py
├── README.md
└── .gitignore
```

## Training Pipeline

The training pipeline includes:

- CIFAR-10 dataset loading
- Data augmentation
  - RandomCrop
  - RandomHorizontalFlip
- Normalization
- Model selection
- CrossEntropyLoss
- SGD optimizer
- Training and evaluation loop
- Best checkpoint saving
- CSV history saving
- Loss and accuracy curve plotting

## Current Test Result

The current result is from a **1 epoch test run** to check whether the training pipeline works correctly.

This is **not the final reproduction result**.

| Model | Epochs | Train Loss | Train Acc | Test Loss | Test Acc |
|---|---:|---:|---:|---:|---:|
| ResNet-20 | 1 | 1.6511 | 38.36% | 1.5524 | 44.00% |

## Result Curves

### Loss Curve

![Loss Curve](results/resnet20_loss_curve.png)

### Accuracy Curve

![Accuracy Curve](results/resnet20_accuracy_curve.png)

## How to Run

Train ResNet-20:

```bash
python train.py
```

Plot training results:

```bash
python plot_results.py
```

## Current Status

Completed:

- CIFAR-10 ResNet model implementation
- CIFAR-10 PlainNet model implementation
- Training pipeline
- CSV result saving
- Loss and accuracy plot generation
- Initial 1 epoch test run

Next steps:

- Train ResNet-20 for more epochs
- Train PlainNet-20 under the same setting
- Compare PlainNet-20 vs ResNet-20
- Extend experiments to PlainNet-56 and ResNet-56
- Analyze whether residual connections improve deep network training

## Notes

Model checkpoints and datasets are not uploaded to this repository.

Ignored files include:

- `data/`
- `checkpoints/`
- `*.pth`
- `*.pt`
- Python cache files