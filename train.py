import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.resnet import resnet20, resnet56
from models.plainnet import plainnet20, plainnet56

# ==========================
# 1. Device Setting
# ==========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# ==========================
# 2. Hyperparameters
# ==========================

batch_size = 128
learning_rate = 0.1
epochs = 10
num_classes = 10

# ==========================
# 3. Data Transform
# ==========================

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

# ==========================
# 4. Dataset and DataLoader
# ==========================

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# ==========================
# 5. Model Selection
# ==========================

model_name = "resnet20"

if model_name == "resnet20":
    model = resnet20(num_classes=num_classes)
elif model_name == "resnet56":
    model = resnet56(num_classes=num_classes)
elif model_name == "plainnet20":
    model = plainnet20(num_classes=num_classes)
elif model_name == "plainnet56":
    model = plainnet56(num_classes=num_classes)
else:
    raise ValueError("Invalid model name")

model = model.to(device)

# ==========================
# 6. Loss and Optimizer
# ==========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=0.9,
    weight_decay=5e-4
)

# ==========================
# 7. Train Function
# ==========================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

# ==========================
# 8. Test Function
# ==========================

def evaluate(model, test_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

# ==========================
# 9. Main Training Loop
# ==========================

best_acc = 0.0
history = []

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_acc:.2f}%"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc
    })

    if test_acc > best_acc:
        best_acc = test_acc

        torch.save(
            model.state_dict(),
            f"checkpoints/{model_name}_best.pth"
        )

        print(f"Best model saved! Test Acc: {best_acc:.2f}%")

result_path = f"results/{model_name}_history.csv"

with open(result_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc"]
    )

    writer.writeheader()
    writer.writerows(history)

print(f"Training history saved to {result_path}")