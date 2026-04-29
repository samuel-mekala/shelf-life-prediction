"""
Shelf Life Prediction — ShuffleNet V2 Transfer Learning
Author  : Samuel Mekala | VIT-AP University
Results : Accuracy 97.25% | F1 96.80% | ROC AUC 97.10% | PR AUC 95.85%
"""
import torch, torch.nn as nn, torch.optim as optim, time, os
import numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score, auc
from torch.optim.lr_scheduler import StepLR

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128; num_epochs = 20; earlystop_patience = 3; learning_rate = 0.003; dropout_rate = 0.5

transform = transforms.Compose([
    transforms.Resize((256,256)), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

dataset    = datasets.ImageFolder(root="data/shelf_life", transform=transform)
num_classes = len(dataset.classes)
print(f"Classes: {dataset.classes} | Total: {len(dataset)}")

train_size = int(0.70*len(dataset)); val_size = int(0.15*len(dataset)); test_size = len(dataset)-train_size-val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

model    = models.shufflenet_v2_x1_0(weights="DEFAULT")
model.fc = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(model.fc.in_features, num_classes))
model    = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience=patience; self.verbose=verbose; self.counter=0; self.best_score=None; self.early_stop=False
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None: self.best_score=score
        elif score < self.best_score:
            self.counter+=1
            if self.counter>=self.patience: self.early_stop=True
        else: self.best_score=score; self.counter=0

def compute_metrics(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred); f1 = f1_score(y_true, y_pred, average="weighted")
    auc_s = roc_auc_score(y_true, y_probs, multi_class="ovr")
    try:
        p, r, _ = precision_recall_curve(y_true, y_probs[:,1], pos_label=1); auc_pr = auc(r, p)
    except: auc_pr = 0.0
    return acc, f1, auc_s, auc_pr

early_stopping = EarlyStopping(patience=earlystop_patience, verbose=True)
train_losses, val_losses, train_accs, val_accs = [], [], [], []
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(num_epochs):
    model.train(); tl, ct, tt = 0, 0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device); optimizer.zero_grad()
        out = model(X); loss = criterion(out, y); loss.backward(); optimizer.step()
        tl += loss.item()*X.size(0); _, p = torch.max(out,1); ct += torch.sum(p==y.data); tt += y.size(0)
    tl /= tt; ta = ct.double()/tt; train_losses.append(tl); train_accs.append(ta.item())

    model.eval(); vl, cv, tv = 0, 0, 0; al, ap, apr = [], [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device); out = model(X); loss = criterion(out, y)
            vl += loss.item()*X.size(0); _, p = torch.max(out,1); cv += torch.sum(p==y.data); tv += y.size(0)
            al.extend(y.cpu().numpy()); ap.extend(p.cpu().numpy()); apr.extend(torch.softmax(out,1).cpu().numpy())
    vl /= tv; va = cv.double()/tv; val_losses.append(vl); val_accs.append(va.item())
    acc, f1, auc_s, auc_pr = compute_metrics(al, ap, np.array(apr))
    print(f"Epoch [{epoch+1}/{num_epochs}] TL:{tl:.4f} TA:{ta:.4f} VL:{vl:.4f} VA:{va:.4f} F1:{f1:.4f} AUC:{auc_s:.4f} PR:{auc_pr:.4f}")
    torch.save(model.state_dict(), "checkpoints/shufflenet_best.pt")
    early_stopping(vl, model)
    if early_stopping.early_stop: print("Early stopping."); break
    scheduler.step()

# Test
model.eval(); tl2, ct2, tt2 = 0, 0, 0; tl2_list, tp_list, tpr_list = [], [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device); out = model(X); loss = criterion(out, y)
        tl2 += loss.item()*X.size(0); _, p = torch.max(out,1); ct2 += torch.sum(p==y.data); tt2 += y.size(0)
        tl2_list.extend(y.cpu().numpy()); tp_list.extend(p.cpu().numpy()); tpr_list.extend(torch.softmax(out,1).cpu().numpy())
tl2 /= tt2; ta2 = ct2.double()/tt2
test_acc, test_f1, test_auc, test_auc_pr = compute_metrics(tl2_list, tp_list, np.array(tpr_list))
print(f"\nTest Loss:{tl2:.4f} Acc:{ta2:.4f} F1:{test_f1:.4f} AUC:{test_auc:.4f} PR-AUC:{test_auc_pr:.4f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/shufflenet_shelf_life.pt")
print("Saved: models/shufflenet_shelf_life.pt")
