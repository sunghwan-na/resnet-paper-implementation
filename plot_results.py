import csv
import os
import matplotlib.pyplot as plt

# ========================
# 1. File Path Setting
# ========================

model_name = "resnet20"

csv_path = f"results/{model_name}_history.csv"
loss_plot_path = f"results/{model_name}_loss_curve.png"
acc_plot_path = f"results/{model_name}_accuracy_curve.png"

# ========================
# 2. Load CSV History
# ========================

epochs = []
train_losses = []
train_accs = []
test_losses = []
test_accs = []

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        epochs.append(int(row["epoch"]))
        train_losses.append(float(row["train_loss"]))
        train_accs.append(float(row["train_acc"]))
        test_losses.append(float(row["test_loss"]))
        test_accs.append(float(row["test_acc"]))

# ========================
# 3. Plot Loss Curve
# ========================

plt.figure()
plt.plot(epochs, train_losses, label="Train loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{model_name} Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig(loss_plot_path)
plt.close()

# ========================
# 4. Plot Accuracy Curve
# ========================

plt.figure()
plt.plot(epochs, train_accs, label="Train Accuracy")
plt.plot(epochs, test_accs, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title(f"{model_name} Accuracy Curve")
plt.legend()
plt.grid(True)
plt.savefig(acc_plot_path)
plt.close()

print(f"Loss curve saved to {loss_plot_path}")
print(f"Accuracy curve saved to {acc_plot_path}")