“””
Shelf Life Prediction of Fruits and Vegetables
Using ShuffleNet V2 with Transfer Learning

Capstone Project — VIT-AP University, 2024
Authors: Satyala Murali Karthik, Mekala Samuel, Yelakanti Ramu
Guide: Dr. S. Kalyani

Results: Accuracy 97.25% | F1 Score 96.80% | ROC AUC 97.10%
“””

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import (f1_score, roc_auc_score,
precision_recall_curve, accuracy_score, auc)

# ─── Device ──────────────────────────────────────────────────────────────────

device = torch.device(“cuda” if torch.cuda.is_available() else “cpu”)
print(f”Using device: {device}”)

# ─── User Inputs ─────────────────────────────────────────────────────────────

batch_size         = int(input(“Enter the batch size: “))
num_epochs         = int(input(“Enter the number of epochs: “))
earlystop_patience = int(input(“Enter early stop patience: “))
learning_rate      = float(input(“Enter the learning rate: “))
dropout_rate       = float(input(“Enter the dropout rate: “))

# ─── Dataset Transformations ─────────────────────────────────────────────────

transform = transforms.Compose([
transforms.Resize((256, 256)),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]),
])

# ─── Load Dataset ─────────────────────────────────────────────────────────────

# Place dataset in ./data/shelf_life/

# Expected folder structure:

# data/shelf_life/Fresh/   ← fresh produce images

# data/shelf_life/Rotten/  ← rotten produce images

dataset_path = os.path.join(os.path.dirname(os.path.abspath(**file**)), ‘data’, ‘shelf_life’)
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

num_classes = len(dataset.classes)
print(f”Classes found: {dataset.classes}”)

# Split: 70% train | 15% val | 15% test

train_size = int(0.7 * len(dataset))
val_size   = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# ─── Model: ShuffleNet V2 ─────────────────────────────────────────────────────

model = models.shufflenet_v2_x1_0(weights=“DEFAULT”)
model.fc = nn.Sequential(
nn.Dropout(p=dropout_rate),
nn.Linear(model.fc.in_features, num_classes)
)
model = model.to(device)

# ─── Loss, Optimizer, Scheduler ──────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# ─── Early Stopping ──────────────────────────────────────────────────────────

class EarlyStopping:
def **init**(self, patience=7, verbose=False):
self.patience   = patience
self.verbose    = verbose
self.counter    = 0
self.best_score = None
self.early_stop = False

```
def __call__(self, val_loss, model):
    score = -val_loss
    if self.best_score is None:
        self.best_score = score
    elif score < self.best_score:
        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
    else:
        self.best_score = score
        self.counter = 0
```

# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_probs):
acc       = accuracy_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred, average=‘weighted’)
auc_score = roc_auc_score(y_true, y_probs, multi_class=‘ovr’)
precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1], pos_label=1)
auc_pr = auc(recall, precision)
return acc, f1, auc_score, precision, recall, auc_pr

# ─── Training Loop ───────────────────────────────────────────────────────────

early_stopping = EarlyStopping(patience=earlystop_patience, verbose=True)
train_loss_values, val_loss_values = [], []
train_acc_values,  val_acc_values  = [], []
epoch_times = []

for epoch in range(num_epochs):
start_time = time.time()
model.train()
train_loss, correct_train, total_train = 0, 0, 0

```
for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    train_loss    += loss.item() * inputs.size(0)
    _, predicted   = torch.max(outputs, 1)
    correct_train += (predicted == labels).sum().item()
    total_train   += labels.size(0)

train_loss    /= total_train
train_accuracy = correct_train / total_train
train_loss_values.append(train_loss)
train_acc_values.append(train_accuracy)

# Validation
model.eval()
val_loss, correct_val, total_val = 0, 0, 0
all_labels, all_preds, all_probs = [], [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss    = criterion(outputs, labels)

        val_loss      += loss.item() * inputs.size(0)
        _, predicted   = torch.max(outputs, 1)
        correct_val   += (predicted == labels).sum().item()
        total_val     += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

val_loss    /= total_val
val_accuracy = correct_val / total_val
val_loss_values.append(val_loss)
val_acc_values.append(val_accuracy)

acc, f1, auc_val, precision, recall, auc_pr = compute_metrics(
    all_labels, all_preds, np.array(all_probs)
)

epoch_time = time.time() - start_time
epoch_times.append(epoch_time)

print(f"\nEpoch [{epoch+1}/{num_epochs}]  Time: {epoch_time:.2f}s")
print(f"  Train  Loss: {train_loss:.4f}  Acc: {train_accuracy:.4f}")
print(f"  Val    Loss: {val_loss:.4f}  Acc: {val_accuracy:.4f}  F1: {f1:.4f}  AUC: {auc_val:.4f}")

early_stopping(val_loss, model)
if early_stopping.early_stop:
    print("Early stopping triggered.")
    break

scheduler.step()
```

# ─── Training Curves ─────────────────────────────────────────────────────────

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_values, label=“Train Loss”)
plt.plot(val_loss_values,   label=“Validation Loss”)
plt.title(“Loss vs Epoch”)
plt.xlabel(“Epoch”); plt.ylabel(“Loss”); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_values, label=“Train Accuracy”)
plt.plot(val_acc_values,   label=“Validation Accuracy”)
plt.title(“Accuracy vs Epoch”)
plt.xlabel(“Epoch”); plt.ylabel(“Accuracy”); plt.legend()

plt.tight_layout()
plt.savefig(“training_curves.png”, dpi=150)
plt.show()
print(“Training curves saved to training_curves.png”)

# ─── Save Model ──────────────────────────────────────────────────────────────

torch.save(model.state_dict(), “shufflenet_shelf_life.pth”)
print(“Model saved to shufflenet_shelf_life.pth”)
print(f”Average training time per epoch: {np.mean(epoch_times):.2f}s”)