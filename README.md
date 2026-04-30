# 🍎 Predicting Shelf Life of Fruits & Vegetables Using Deep Learning

> **Capstone Project** · VIT-AP University · Jul–Dec 2024  
> **Team:** Satyala Murali Karthik · **Mekala Samuel** · Yelakanti Ramu  
> **Guide:** Dr. S. Kalyani · School of Electronics Engineering

---

## 📌 Overview

Food wastage is a massive global challenge — particularly for perishable produce. This project develops a **deep learning-based image classification system** that predicts the shelf life of fruits and vegetables (Fresh / Rotten), helping grocery stores, warehouses, and supply chains **reduce waste and improve quality control**.

The solution uses **ShuffleNet V2** with transfer learning, outperforming ResNet, MobileNet, and EfficientNet — all while being lightweight enough for **edge device deployment**.

---

## 🏆 Results

### Our Model vs State-of-the-Art

| Model | Accuracy | F1 Score | Remarks |
|---|---|---|---|
| **ShuffleNet V2 (Ours)** | **97.25%** | **96.80%** | Superior accuracy + lowest computation cost ✅ |
| MobileNetV2 | 96.40% | 95.85% | Lightweight, close runner-up |
| EfficientNet-B0 | 95.75% | 95.25% | Good accuracy/speed balance |
| DenseNet-121 | 95.10% | 94.60% | Slower than ShuffleNet |
| ResNet-18 | 94.85% | 94.10% | Reliable but computationally heavier |

### Final Metrics on Test Set
| Metric | Score |
|---|---|
| **Accuracy** | **97.25%** |
| **F1 Score** | **96.80%** |
| **ROC AUC** | **97.10%** |
| **Precision-Recall AUC** | **95.85%** |

---

## 🧠 Why ShuffleNet V2?

ShuffleNet V2 uses **channel split + shuffle operations** to achieve a remarkable balance between computational efficiency and classification accuracy — ideal for:
- ✅ **Resource-constrained devices** (Raspberry Pi, mobile)
- ✅ **Real-time inference** in grocery/warehouse environments
- ✅ **High accuracy** without expensive compute

---

## ⚙️ Methodology

### Dataset
- Labeled images of fruits and vegetables categorized by **freshness levels**
- Organized into class folders for PyTorch `ImageFolder` compatibility

### Preprocessing Pipeline
```
Raw Images
    → Resize to 256×256
    → Center Crop to 224×224
    → Normalize (ImageNet: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    → Optional Augmentation (Flip · Rotation · Color Jitter)
    → Tensor Conversion
    → Split (70% Train / 15% Val / 15% Test)
```

### Training Configuration
```python
model = models.shufflenet_v2_x1_0(weights="DEFAULT")  # Pre-trained ImageNet weights
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

optimizer = optim.Adam(model.parameters(), lr=0.003)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()
```

**Hyperparameters:**
- Batch Size: 128 · Epochs: 20 · LR: 0.003 · Dropout: 0.5 · Early Stop Patience: 3

---

## 🖥️ Tkinter GUI

A **Python Tkinter GUI** enables non-technical users to:
1. Upload an image of a fruit/vegetable via file dialog
2. View the uploaded image in the app canvas
3. Receive the freshness prediction + confidence score + estimated days remaining (for fresh items)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Pillow](https://img.shields.io/badge/Pillow-blue?style=flat-square)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-blue?style=flat-square)

---

## 📁 Project Structure

```
shelf-life-prediction/
├── train.py            # Full training pipeline with early stopping & evaluation
├── app.py              # Tkinter GUI for real-time freshness prediction
├── requirements.txt
└── README.md
```

> After training, `shufflenet_shelf_life.pth` (model weights) and `training_curves.png` are saved to the project root.

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/samuel-mekala/shelf-life-prediction.git
cd shelf-life-prediction

# Install dependencies
pip install -r requirements.txt

# Place your dataset in ./data/shelf_life/
# Folder structure: data/shelf_life/Fresh/ and data/shelf_life/Rotten/

# Train the model
python train.py

# Launch GUI (load the saved .pth weights)
python app.py
```

---

## 🔮 Future Work

- [ ] Ensemble learning (ShuffleNet + MobileNetV2)
- [ ] Vision Transformers (ViT) for further gains
- [ ] Deploy to mobile via TensorFlow Lite
- [ ] Expand dataset with diverse lighting and environments
- [ ] Multi-task learning: detect disease + predict severity

---

*VIT-AP University · Capstone Project · Dec 2024*
