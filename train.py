"""
Shelf Life Prediction of Fruits and Vegetables
Using ShuffleNet V2 with Transfer Learning

Capstone Project — VIT-AP University, 2024
Authors: Satyala Murali Karthik, Mekala Samuel, Yelakanti Ramu
Guide: Dr. S. Kalyani
"""

import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
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

print(f"Using compute device: {device}", flush=True)

# ─── Custom Dataset Class ─────────────────────────────────────────────────────

class ProduceDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples_per_class=2000):
        self.transform = transform
        self.samples = []

        fresh_samples = []
        rotten_samples = []

        produce_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        for p_dir in produce_dirs:
            p_path = os.path.join(root_dir, p_dir)
            fresh_dir = os.path.join(p_path, 'fresh')
            if not os.path.exists(fresh_dir):
                fresh_dir = os.path.join(p_path, 'Fresh')
            
            rotten_dir = os.path.join(p_path, 'rotten')
            if not os.path.exists(rotten_dir):
                rotten_dir = os.path.join(p_path, 'Rotten')

            if os.path.exists(fresh_dir):
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.PNG'):
                    for img_p in glob.glob(os.path.join(fresh_dir, ext)):
                        fresh_samples.append((img_p, 0))

            if os.path.exists(rotten_dir):
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.PNG'):
                    for img_p in glob.glob(os.path.join(rotten_dir, ext)):
                        rotten_samples.append((img_p, 1))

        # Sample up to max_samples_per_class for balanced, fast, high-accuracy fine-tuning
        np.random.seed(42)
        if len(fresh_samples) > max_samples_per_class:
            fresh_indices = np.random.choice(len(fresh_samples), max_samples_per_class, replace=False)
            fresh_samples = [fresh_samples[i] for i in fresh_indices]

        if len(rotten_samples) > max_samples_per_class:
            rotten_indices = np.random.choice(len(rotten_samples), max_samples_per_class, replace=False)
            rotten_samples = [rotten_samples[i] for i in rotten_indices]

        self.samples = fresh_samples + rotten_samples
        np.random.shuffle(self.samples)

        print(f"Dataset loaded: {len(self.samples)} balanced samples ({sum(1 for _, l in self.samples if l==0)} Fresh, {sum(1 for _, l in self.samples if l==1)} Rotten).", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label

# ─── Dataset Transformations ─────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Evaluation Metrics ───────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    try:
        pos_probs = y_probs[:, 1] if y_probs.ndim > 1 and y_probs.shape[1] > 1 else y_probs
        auc_score = roc_auc_score(y_true, pos_probs)
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

def train_model(batch_size=128, num_epochs=4, learning_rate=0.003, dropout_rate=0.5, dataset_dir=None):
    if dataset_dir is None:
        if os.path.exists('Data'):
            dataset_dir = 'Data'
        else:
            dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'shelf_life')
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.", flush=True)
        return None

    dataset = ProduceDataset(root_dir=dataset_dir, transform=transform, max_samples_per_class=2000)

    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(model.fc.in_features, 2)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    train_loss_values, val_loss_values = [], []
    train_acc_values,  val_acc_values  = [], []
    epoch_times = []

    print(f"--- Starting Fine-Tuning ShuffleNet V2 on {len(dataset)} samples ---", flush=True)
    total_batches = len(train_loader)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for i, (inputs, labels) in enumerate(train_loader):
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

            if (i + 1) % 5 == 0 or (i + 1) == total_batches:
                current_acc = correct_train / total_train
                current_loss = train_loss / total_train
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{total_batches}] - Loss: {current_loss:.4f} Acc: {current_acc*100:.2f}%", flush=True)

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
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(f"\n>> Epoch [{epoch+1}/{num_epochs}] Completed in {epoch_time:.2f}s | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | F1: {f1:.4f} | AUC: {auc_val:.4f}\n", flush=True)

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
    print("Training curves saved to images/training_curves.png", flush=True)

    # Save trained model weights
    save_path = "shufflenet_shelf_life.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}", flush=True)
    return model

if __name__ == "__main__":
    train_model(batch_size=128, num_epochs=4, learning_rate=0.003, dropout_rate=0.5)