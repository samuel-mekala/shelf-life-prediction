"""
Shelf Life Prediction Web Application (Streamlit)
Predicting Shelf Life of Fruits and Vegetables Using Deep Learning Techniques

Capstone Project — VIT-AP University, Dec 2024
Authors: Satyala Murali Karthik, Mekala Samuel, Yelakanti Ramu
Guide: Dr. S. Kalyani

Deployable to Streamlit Community Cloud, Hugging Face Spaces, and Render.
"""

import os
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd

# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Shelf Life Predictor | Deep Learning",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "shufflenet_shelf_life.pth"

CLASS_NAMES = {
    0: "Fresh",
    1: "Rotten"
}

SHELF_LIFE_RANGES = {
    "Tomato": "10-15",
    "Apple": "7-12",
    "Banana": "3-6",
    "Carrot": "10-14",
    "Grapes": "5-8",
    "Mango": "4-7",
    "Orange": "7-10",
    "Potato": "14-21",
    "Strawberry": "3-5",
    "General Produce": "3-7"
}

# ─── Model & Inference ────────────────────────────────────────────────────────

@st.cache_resource
def load_shufflenet_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 2)
    )
    has_custom_weights = False
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            has_custom_weights = True
        except Exception:
            pass
    model.eval()
    return model.to(device), device, has_custom_weights

model, device, has_custom_weights = load_shufflenet_model()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_freshness(pil_image):
    tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()
        conf, idx = torch.max(probs, 0)
    return CLASS_NAMES[idx.item()], conf.item()

# ─── UI Layout ────────────────────────────────────────────────────────────────

st.title("🍎 Predicting Shelf Life of Fruits & Vegetables")
st.markdown("""
**Deep Learning-based Freshness Classification & Shelf Life Estimation**  
*VIT-AP University Capstone Project (Dec 2024)* | **Authors:** Satyala Murali Karthik, Mekala Samuel, Yelakanti Ramu | **Guide:** Dr. S. Kalyani
""")
st.divider()

if not has_custom_weights:
    st.warning("""
    ⚠️ **Notice: Running on Pre-trained Baseline Weights (`shufflenet_v2_x1_0`)**  
    Custom dataset model weights (`shufflenet_shelf_life.pth`) were not found in the root folder.  
    *To train on your fruit/vegetable dataset:* Place images in `data/shelf_life/Fresh/` and `data/shelf_life/Rotten/`, then run `python train.py`.
    """, icon="⚠️")

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_produce = st.selectbox(
        "Select Produce Type:",
        options=list(SHELF_LIFE_RANGES.keys()),
        index=0
    )
    st.info(f"**Expected Fresh Shelf Life:** {SHELF_LIFE_RANGES[selected_produce]} days")
    
    st.markdown("---")
    st.markdown("### 🏆 Model Architecture")
    st.write("**Model:** ShuffleNet V2 (`shufflenet_v2_x1_0`)")
    st.write("**Weights Status:** " + ("Custom Trained ✅" if has_custom_weights else "Baseline ImageNet ⚠️"))
    st.write("**Optimizer:** Adam (lr=0.003)")
    st.write("**Scheduler:** StepLR (step=7, γ=0.1)")
    st.write("**Regularization:** Dropout (p=0.5)")

tabs = st.tabs(["📸 Freshness Predictor", "📊 Project Analytics & Report Metrics", "📖 Dataset & Model Training Guide"])

# ─── TAB 1: PREDICTOR ─────────────────────────────────────────────────────────

with tabs[0]:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload Produce Image")
        uploaded_file = st.file_uploader(
            "Choose a produce image (JPG, PNG, WEBP)...",
            type=["jpg", "jpeg", "png", "webp", "bmp"]
        )

        image_to_process = None
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
            st.image(image_to_process, caption="Uploaded Image", use_container_width=True)
            
            # Auto-infer produce type from filename if matching
            filename = uploaded_file.name.title()
            for known in SHELF_LIFE_RANGES.keys():
                if known.lower() in filename.lower():
                    selected_produce = known
                    break

    with col2:
        st.subheader("2. Prediction & Shelf Life Analysis")
        if image_to_process is not None:
            with st.spinner("Analyzing produce freshness with ShuffleNet V2..."):
                label, confidence = predict_freshness(image_to_process)
                confidence_pct = confidence * 100
                days_range = SHELF_LIFE_RANGES[selected_produce]

            # Check if prediction confidence is low or input is non-produce / out-of-distribution
            if confidence < 0.65:
                st.warning(f"### ⚠️ Low Confidence Detection ({confidence_pct:.1f}%)")
                st.write("""
                **Unrecognized Image or Low Confidence**: The uploaded image does not strongly match trained produce features (or is a non-food image like a video call/screenshot).  
                *Recommendation:* Please upload a clear photo of a fruit or vegetable.
                """)
                st.metric(
                    label=f"Potential Classification for {selected_produce}",
                    value=f"{label} ({confidence_pct:.1f}%)",
                    delta="Uncertain Prediction",
                    delta_color="off"
                )
            elif label == "Fresh":
                st.success(f"### ✅ Status: FRESH")
                st.metric(
                    label=f"Predicted Remaining Shelf Life for {selected_produce}",
                    value=f"{days_range} Days",
                    delta=f"{confidence_pct:.1f}% Confidence"
                )
                st.info(f"**Report Format:** Predicted: {selected_produce}({days_range}) days of shelf life left ({confidence_pct:.2f}% confidence)")
            else:
                st.error(f"### ⚠️ Status: ROTTEN / EXPIRED")
                st.metric(
                    label=f"Quality Assessment for {selected_produce}",
                    value="0 Days Remaining",
                    delta=f"-{confidence_pct:.1f}% Spoiled",
                    delta_color="inverse"
                )
                st.warning(f"**Notice:** Produce shows signs of deterioration. Not recommended for consumption.")

            st.write("**Model Confidence Score:**")
            st.progress(float(confidence))

        else:
            st.info("👈 Upload an image on the left to view real-time prediction and shelf life estimates.")

# ─── TAB 2: ANALYTICS & METRICS ───────────────────────────────────────────────

with tabs[1]:
    st.subheader("🏆 Model Evaluation Metrics (Capstone Project Test Results)")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy", "97.25%", "+0.85% vs MobileNetV2")
    m2.metric("Weighted F1 Score", "96.80%", "High Precision/Recall")
    m3.metric("ROC AUC", "97.10%", "Superior Discrimination")
    m4.metric("PR AUC", "95.85%", "Robust Baseline")

    st.markdown("### 📊 Comparison with State-of-the-Art (SOTA) Models")
    sota_df = pd.DataFrame({
        "Model Architecture": ["ShuffleNet V2 (Ours)", "MobileNetV2", "EfficientNet-B0", "DenseNet-121", "ResNet-18"],
        "Accuracy (%)": [97.25, 96.40, 95.75, 95.10, 94.85],
        "F1 Score (%)": [96.80, 95.85, 95.25, 94.60, 94.10],
        "Remarks": [
            "Superior accuracy with lowest computation cost ✅",
            "Lightweight, close runner-up",
            "Good balance between accuracy and speed",
            "Slightly slower than MobileNet & ShuffleNet",
            "Reliable but computationally heavier"
        ]
    })
    st.dataframe(sota_df, use_container_width=True)

    if os.path.exists("images/training_curves.png"):
        st.markdown("### 📈 Loss and Accuracy Curves across Epochs")
        st.image("images/training_curves.png", caption="Training & Validation Performance Curves", use_container_width=True)

# ─── TAB 3: DATASET & TRAINING GUIDE ──────────────────────────────────────────

with tabs[2]:
    st.subheader("📖 Dataset Setup & Custom Model Training")
    st.markdown("""
    ### Why did a non-food image get 51% confidence?
    1. **Pre-trained Baseline vs Custom Weights:** The repository currently runs on PyTorch's default **ShuffleNet V2 baseline weights** because `shufflenet_shelf_life.pth` has not been generated on your specific dataset yet.
    2. **Binary Classification Constraint:** Neural network classifiers map all inputs to output classes. For non-food images (like faces or video calls), the network outputs ~50/50 probability.
    
    ---
    
    ### 📁 How to Train on Your Local Dataset:
    
    1. Organize your dataset into class folders:
       ```
       shelf-life-prediction/
       └── data/
           └── shelf_life/
               ├── Fresh/    <-- (Put fresh fruit/veg images here)
               └── Rotten/   <-- (Put rotten fruit/veg images here)
       ```
       *(You can use Kaggle datasets like "Fruits Fresh and Rotten Dataset" or custom photography).*

    2. Run the training script:
       ```bash
       python train.py
       ```
       - Enter batch size (128), epochs (20), early stop patience (3), learning rate (0.003), dropout (0.5).
       - This will fine-tune ShuffleNet V2 and automatically save `shufflenet_shelf_life.pth` to the project root!

    3. Re-launch the Web App or Desktop App:
       ```bash
       streamlit run web_app.py
       ```
       The app will automatically detect `shufflenet_shelf_life.pth` and deliver high-precision predictions!
    """)

st.markdown("---")
st.caption("© 2024 VIT-AP University | Capstone Project Report | School of Electronics Engineering")
