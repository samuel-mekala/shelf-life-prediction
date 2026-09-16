# 🍎 Shelf Life Prediction of Fruits & Vegetables Using Deep Learning

> **Capstone Project** · VIT-AP University · Dec 2024  
> **Team:** Satyala Murali Karthik · **Mekala Samuel** · Yelakanti Ramu  
> **Guide:** Dr. S. Kalyani · School of Electronics Engineering  
> **Repository:** [github.com/samuel-mekala/shelf-life-prediction](https://github.com/samuel-mekala/shelf-life-prediction)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=flat-square&logo=github-actions)](/.github/workflows/ci.yml)

---

## 📌 Overview

Food wastage is a critical global challenge, particularly for perishable produce. This project presents a **deep learning-based image classification and shelf-life estimation system** using a fine-tuned **ResNet-50** deep neural network combined with a **Dual-Stage General Produce Vision Recognizer**. The system categorizes fruits and vegetables into fine-grained shelf life stages, estimates the remaining shelf life in days for fresh items, and handles both trained produce classes (`Apple`, `Banana`, `Carrot`, `Tomato`) and general produce items (`Guava`, `Orange`, `Mango`, `Strawberry`, `Pineapple`, `Pomegranate`, `Cucumber`, `Potato`, etc.) with high-precision confidence scores.

---

## 🏗️ System Architecture & Dual-Stage Workflow

```
Produce Image Input (Upload / Camera)
               │
               ▼
┌──────────────────────────────────────────────┐
│            Image Preprocessing               │
│  • Resize → 256×256 | CenterCrop → 224×224    │
│  • Normalize (ImageNet Mean & Std)           │
│  • Split: 70% Train / 15% Val / 15% Test     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│     Stage 1: Fine-Tuned ResNet-50 Model      │
│  • 25.6M Deep Residual Parameters            │
│  • Balanced Class Oversampling & Dropout     │
│  • 14 Direct Shelf-Life Stage Categories     │
└──────────────────────┬───────────────────────┘
                       │
             Conf >= 70% ?
            ┌──────────┴──────────┐
         YES│                     │NO
            ▼                     ▼
  Trained Class Output  ┌─────────────────────────────────────┐
  (Apple, Banana,       │ Stage 2: General Produce Vision     │
   Carrot, Tomato)      │ Recognizer (Pre-trained ImageNet)   │
                        │ Maps Guava, Orange, Mango, etc.     │
                        └──────────────────┬──────────────────┘
                                           │
                                           ▼
                                 General Produce Output
                               (Guava: 4-7 Days Shelf Life)
```

---

## 🏆 Model Evaluation & Performance Metrics

### Final Metrics on 14 Direct Stage Categories (1,932 Images)
| Metric | Score | Performance Highlights |
|---|---|---|
| **Training Accuracy** | **95.40%** | Deep residual feature extraction |
| **Validation Accuracy** | **91.73%** | High generalization across unseen produce |
| **Validation F1-Score** | **0.9187** | Balanced precision & recall across all categories |
| **Dataset Size** | **1,932 Images** | 14 Fine-Grained Shelf-Life Stage Classes |
| **Model Architecture** | **ResNet-50 + General Vision** | Dual-Stage Inference Pipeline |

---

## 💻 Interfaces

### 🌐 1. Streamlit Web Interface (`web_app.py`)
An interactive web portal for live browser-based shelf-life prediction, produce category selection, and dynamic metric visualizations.

- **Drag-and-Drop Image Upload**
- **Dual-Stage Produce Recognition (Auto-Detect Guava, Apple, Banana, etc.)**
- **Interactive Freshness Meter & Shelf Life Days Badge**
- **SOTA Comparison & Metric Analytics Tab**

```bash
# Launch Streamlit Web App locally
streamlit run web_app.py
```

### 🖥️ 2. Desktop Tkinter Interface (`app.py`)
A desktop GUI matching the capstone project report specifications:
```bash
# Launch Tkinter Desktop App
python app.py
```

Output format as specified in report:
`Predicted: Apple (10-14) days of shelf life left (81.62% confidence)`

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/samuel-mekala/shelf-life-prediction.git
cd shelf-life-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Web App
streamlit run web_app.py

# 4. Or launch the Desktop App
python app.py
```

---

## 🌐 Deploying to the Web (Live Hosting)

### ⚡ Option 1: Deploy to Streamlit Community Cloud (Free & 1-Click)
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account (`samuel-mekala`).
2. Click **Create app** / **New app**.
3. Fill in the repository details:
   - **Repository:** `samuel-mekala/shelf-life-prediction`
   - **Branch:** `main`
   - **Main file path:** `web_app.py`
4. Click **Deploy!**
5. Your app will automatically build and publish to a public URL (e.g., `https://shelf-life-prediction.streamlit.app`)!

### 📦 Option 2: Deploy to Render or Hugging Face Spaces
- **Render**: Create a Web Service connected to your repo, set Build Command to `pip install -r requirements.txt`, and Start Command to `streamlit run web_app.py --server.port $PORT --server.address 0.0.0.0`.
- **Hugging Face Spaces**: Create a Streamlit space and point to `web_app.py`.

---

## 🛠️ Tech Stack

- **Deep Learning Framework:** PyTorch & torchvision
- **Model Backbone:** ResNet-50 (25.6M parameters)
- **General Produce Classifier:** Pre-trained Vision Transformer & ResNet Feature Extractor
- **Evaluation & Metrics:** scikit-learn & matplotlib
- **Web Interface:** Streamlit
- **Desktop Interface:** Tkinter
- **Version Control:** Git & GitHub

---

*VIT-AP University · Capstone Project Report · School of Electronics Engineering · Dec 2024*
