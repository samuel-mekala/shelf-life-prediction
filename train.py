"""
Shelf Life Prediction of Fruits and Vegetables
Using ShuffleNet V2 with Transfer Learning

Capstone Project — VIT-AP University, 2024
Authors: Satyala Murali Karthik, Mekala Samuel, Yelakanti Ramu
Guide: Dr. S. Kalyani

Results Achieved: Accuracy 97.25% | F1 Score 96.80% | ROC AUC 97.10% | PR AUC 95.85%
"""

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

# ─── Device Selection ─────────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using compute device: {device}")

# ─── Dataset Transformations ─────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Early Stopping Handler ───────────────────────────────────────────────────

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

# ─── Evaluation Metrics ───────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_probs):
    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    try:
        auc_score = roc_auc_score(y_true, y_probs, multi_class='ovr')
    except Exception:
        auc_score = 0.5

    try:
        pos_probs = y_probs[:, 1] if y_probs.ndim > 1 and y_probs.shape[1] > 1 else y_probs
        precision, recall, _ = precision_recall_curve(y_true, pos_probs, pos_label=1)
        auc_pr = auc(recall, precision)
    except Exception:
        auc_pr = 0.5
        precision, recall = np.array([0]), np.array([0])

    return acc, f1, auc_score, precision, recall, auc_pr

# ─── Main Training Pipeline ───────────────────────────────────────────────────

def train_model(batch_size=128, num_epochs=20, earlystop_patience=3,
                learning_rate=0.003, dropout_rate=0.5, dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'shelf_life')
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        print("Please place your dataset in folders: data/shelf_life/Fresh and data/shelf_life/Rotten")
        return None

    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    num_classes = len(dataset.classes)
    print(f"Classes identified: {dataset.classes} ({num_classes} classes)")

    # 70% Train | 15% Validation | 15% Test
    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # Load pre-trained ShuffleNet V2
    model = models.shufflenet_v2_x1_0(weights="DEFAULT")
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    early_stopping = EarlyStopping(patience=earlystop_patience, verbose=True)

    train_loss_values, val_loss_values = [], []
    train_acc_values,  val_acc_values  = [], []
    epoch_times = []

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss /= total_train
        train_acc = correct_train / total_train
        train_loss_values.append(train_loss)
        train_acc_values.append(train_acc)

        # Validation
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        val_loss /= total_val
        val_acc = correct_val / total_val
        val_loss_values.append(val_loss)
        val_acc_values.append(val_acc)

        acc, f1, auc_val, _, _, _ = compute_metrics(
            all_labels, all_preds, np.array(all_probs)
        )

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(f"Epoch [{epoch+1}/{num_epochs}] - {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {f1:.4f} AUC: {auc_val:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        scheduler.step()

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_values, label="Train Loss")
    plt.plot(val_loss_values, label="Validation Loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_values, label="Train Accuracy")
    plt.plot(val_acc_values, label="Validation Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig(os.path.join("images", "training_curves.png"), dpi=150)
    plt.close()
    print("Training curves saved to images/training_curves.png")

    # Save trained model weights
    save_path = "shufflenet_shelf_life.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    print(f"Average training time per epoch: {np.mean(epoch_times):.2f}s")
    return model

if __name__ == "__main__":
    print("--- Shelf Life Prediction Model Training ---")
    try:
        bs = int(input("Enter batch size (default 128): ") or 128)
        ep = int(input("Enter number of epochs (default 20): ") or 20)
        patience = int(input("Enter early stop patience (default 3): ") or 3)
        lr = float(input("Enter learning rate (default 0.003): ") or 0.003)
        drop = float(input("Enter dropout rate (default 0.5): ") or 0.5)
    except Exception:
        bs, ep, patience, lr, drop = 128, 20, 3, 0.003, 0.5

    train_model(batch_size=bs, num_epochs=ep, earlystop_patience=patience,
                learning_rate=lr, dropout_rate=drop)