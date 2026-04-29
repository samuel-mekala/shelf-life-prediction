"""
Shelf Life Prediction of Fruits and Vegetables
Using ShuffleNet V2 with Transfer Learning
Capstone Project - VIT-AP University, 2024

Authors: Satyala Murali Karthik, Mekala Samuel, Yelakanti Ramu
Accuracy: 97.25% | F1 Score: 96.80% | ROC AUC: 97.10%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score, auc
from torch.optim.lr_scheduler import StepLR
import time
import matplotlib.pyplot as plt
import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────────

DATASET_PATH = "data/shelf_life"   # Update this path to your dataset folder

# Dataset classes expected: Fresh / Rotten (sub-folders inside DATASET_PATH)

# ─── Device Setup ─────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── Hyperparameters (can be passed via CLI or left as defaults) ───────────────

batch_size         = 128
num_epochs         = 20
earlystop_patience = 3
learning_rate      = 0.003
dropout_rate       = 0.5

# ─── Dataset Transformations ───────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Load Dataset ──────────────────────────────────────────────────────────────

dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
num_classes = len(dataset.classes)
print(f"Classes found: {dataset.classes}")

# Split: 70% train | 15% val | 15% test
train_size = int(0.70 * len(dataset))
val_size   = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")

# ─── Model: ShuffleNet V2 (Pre-trained) ────────────────────────────────────────

model = models.shufflenet_v2_x1_0(weights="DEFAULT")

# Replace final FC layer with dropout + linear for our classes
model.fc = nn.Sequential(
    nn.Dropout(p=dropout_rate),
    nn.Linear(model.fc.in_features, num_classes)
)

model = model.to(device)

# ─── Loss, Optimizer, Scheduler ───────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# ─── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience   = patience
        self.verbose    = verbose
        self.counter    = 0
        self.best_score = None
        self.early_stop = False

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

# ─── Metric Computation ────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_probs):
    acc      = accuracy_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred, average="weighted")
    auc_score = roc_auc_score(y_true, y_probs, multi_class="ovr")
    # Precision-Recall AUC (binary case uses class-1 probabilities)
    precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1], pos_label=1)
    auc_pr = auc(recall, precision)
    return acc, f1, auc_score, precision, recall, auc_pr

# ─── Training Loop ─────────────────────────────────────────────────────────────

early_stopping = EarlyStopping(patience=earlystop_patience, verbose=True)

train_loss_values, val_loss_values = [], []
train_acc_values,  val_acc_values  = [], []
epoch_times = []

for epoch in range(num_epochs):
    start_time = time.time()

    # ---------- Train ----------
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * inputs.size(0)
        _, preds       = torch.max(outputs, 1)
        correct_train += torch.sum(preds == labels.data)
        total_train   += labels.size(0)

    train_loss     = train_loss / total_train
    train_accuracy = correct_train.double() / total_train
    train_loss_values.append(train_loss)
    train_acc_values.append(train_accuracy.item())

    # ---------- Validation ----------
    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            val_loss      += loss.item() * inputs.size(0)
            _, preds       = torch.max(outputs, 1)
            correct_val   += torch.sum(preds == labels.data)
            total_val     += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    val_loss     = val_loss / total_val
    val_accuracy = correct_val.double() / total_val
    val_loss_values.append(val_loss)
    val_acc_values.append(val_accuracy.item())

    acc, f1, auc_val, precision, recall, auc_pr = compute_metrics(
        all_labels, all_preds, np.array(all_probs)
    )

    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)

    print(f"\nEpoch [{epoch+1}/{num_epochs}]  Time: {epoch_time:.2f}s")
    print(f"  Train  Loss: {train_loss:.4f}  Acc: {train_accuracy:.4f}")
    print(f"  Val    Loss: {val_loss:.4f}  Acc: {val_accuracy:.4f}  "
          f"F1: {f1:.4f}  AUC: {auc_val:.4f}  AUC_PR: {auc_pr:.4f}")

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

    scheduler.step()

# ─── Plot Training Curves ──────────────────────────────────────────────────────

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss_values, label="Train Loss")
plt.plot(val_loss_values,   label="Validation Loss")
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_values, label="Train Accuracy")
plt.plot(val_acc_values,   label="Validation Accuracy")
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Training curves saved to training_curves.png")

# ─── Save Best Model ──────────────────────────────────────────────────────────

torch.save(model.state_dict(), "shufflenet_shelf_life.pth")
print("Model saved to shufflenet_shelf_life.pth")

# ─── Test Evaluation ──────────────────────────────────────────────────────────

model.eval()
test_loss, correct_test, total_test = 0, 0, 0
all_test_labels, all_test_preds, all_test_probs = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss    = criterion(outputs, labels)

        test_loss     += loss.item() * inputs.size(0)
        _, preds       = torch.max(outputs, 1)
        correct_test  += torch.sum(preds == labels.data)
        total_test    += labels.size(0)

        all_test_labels.extend(labels.cpu().numpy())
        all_test_preds.extend(preds.cpu().numpy())
        all_test_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

test_loss     = test_loss / total_test
test_accuracy = correct_test.double() / total_test

test_acc, test_f1, test_auc, precision, recall, auc_pr = compute_metrics(
    all_test_labels, all_test_preds, np.array(all_test_probs)
)

print("\n" + "="*50)
print("TEST RESULTS")
print("="*50)
print(f"  Loss     : {test_loss:.4f}")
print(f"  Accuracy : {test_accuracy:.4f}  ({test_accuracy*100:.2f}%)")
print(f"  F1 Score : {test_f1:.4f}")
print(f"  ROC AUC  : {test_auc:.4f}")
print(f"  AUC-PR   : {auc_pr:.4f}")
print(f"\nAverage training time per epoch: {np.mean(epoch_times):.2f}s")
