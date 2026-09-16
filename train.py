"""
Shelf Life Prediction of Fruits and Vegetables
Using ShuffleNet V2 with Transfer Learning & Class-Weighted CrossEntropy Loss
Trained on ALL 29,291 Images across 14 Produce Categories (28 Fine-Grained Classes)

Capstone Project — VIT-AP University, 2024
Authors: Satyala Murali Karthik, Mekala Samuel, Yelakanti Ramu
Guide: Dr. S. Kalyani
"""

import os
import glob
import json
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
from sklearn.metrics import f1_score, accuracy_score

# ─── Device Selection ─────────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using compute device: {device}", flush=True)

# ─── Full Produce Dataset (ALL 29,291 Images) ────────────────────────────────

class FullProduceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.class_names = []

        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.PNG', '*.JPEG', '*.bmp')
        
        class_to_idx = {}
        idx = 0
        
        for d in subdirs:
            p_path = os.path.join(root_dir, d)
            # Check if this folder has nested fresh/rotten subfolders or direct images
            has_nested = False
            for status in ['fresh', 'rotten', 'Fresh', 'Rotten']:
                if os.path.isdir(os.path.join(p_path, status)):
                    has_nested = True
                    break
            
            if has_nested:
                for status in ['fresh', 'rotten']:
                    c_name = f"{d}_{status}"
                    if c_name not in class_to_idx:
                        class_to_idx[c_name] = idx
                        self.class_names.append(c_name)
                        idx += 1
            else:
                c_name = d
                if c_name not in class_to_idx:
                    class_to_idx[c_name] = idx
                    self.class_names.append(c_name)
                    idx += 1

        self.class_to_idx = class_to_idx

        # Load samples
        for d in subdirs:
            p_path = os.path.join(root_dir, d)
            has_nested = False
            for status in ['fresh', 'rotten', 'Fresh', 'Rotten']:
                if os.path.isdir(os.path.join(p_path, status)):
                    has_nested = True
                    break
                    
            if has_nested:
                for status in ['fresh', 'rotten']:
                    c_name = f"{d}_{status}"
                    c_idx = class_to_idx[c_name]
                    s_dir = os.path.join(p_path, status)
                    if not os.path.exists(s_dir):
                        s_dir = os.path.join(p_path, status.title())
                    if os.path.exists(s_dir):
                        for ext in valid_extensions:
                            for img_p in glob.glob(os.path.join(s_dir, ext)):
                                self.samples.append((img_p, c_idx))
            else:
                c_name = d
                c_idx = class_to_idx[c_name]
                for ext in valid_extensions:
                    for img_p in glob.glob(os.path.join(p_path, ext)):
                        self.samples.append((img_p, c_idx))

        np.random.seed(42)
        np.random.shuffle(self.samples)
        
        # Calculate class counts for loss weighting
        class_counts = [0] * len(self.class_names)
        for _, c_idx in self.samples:
            class_counts[c_idx] += 1
            
        self.class_counts = class_counts
        print(f"Dataset fully loaded: {len(self.samples)} total images across {len(self.class_names)} classes.", flush=True)
        for name, count in zip(self.class_names, class_counts):
            print(f" - {name:<20}: {count} images", flush=True)

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

# ─── Data Transformations (Heavy Augmentation for High Generalization) ────────

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Main Training Pipeline ───────────────────────────────────────────────────

def train_model(batch_size=32, num_epochs=15, learning_rate=0.001, dropout_rate=0.5, dataset_dir='Data'):
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.", flush=True)
        return None

    full_dataset = FullProduceDataset(root_dir=dataset_dir, transform=train_transform)

    # Save class mapping JSON
    class_mapping = {idx: name for idx, name in enumerate(full_dataset.class_names)}
    with open("class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    print("Class mapping saved to class_mapping.json", flush=True)

    train_size = int(0.80 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = len(full_dataset.class_names)
    
    # Calculate inverse class frequency weights to balance loss
    class_counts = np.array(full_dataset.class_counts, dtype=np.float32)
    weights = 1.0 / (class_counts + 1e-5)
    weights = weights / weights.sum() * num_classes
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model = model.to(device)

    # Label smoothing CrossEntropyLoss + AdamW for superior generalization on unseen photos
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    train_loss_values, val_loss_values = [], []
    train_acc_values,  val_acc_values  = [], []

    print(f"\n--- Fine-Tuning ShuffleNet V2 on ALL {len(full_dataset)} Images ({num_classes} Classes) ---", flush=True)
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

            if (i + 1) % 15 == 0 or (i + 1) == total_batches:
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
        all_labels, all_preds = [], []

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

        val_loss /= total_val
        val_acc = correct_val / total_val
        val_loss_values.append(val_loss)
        val_acc_values.append(val_acc)

        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch_time = time.time() - start_time

        print(f"\n>> Epoch [{epoch+1}/{num_epochs}] Completed in {epoch_time:.2f}s | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | F1: {f1:.4f}\n", flush=True)

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
    train_model(batch_size=32, num_epochs=12, learning_rate=0.001, dropout_rate=0.5)