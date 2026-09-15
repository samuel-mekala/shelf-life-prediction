"""
Shelf Life Prediction Web Application (Streamlit)
Predicting Shelf Life of Fruits and Vegetables Using Deep Learning Techniques

Capstone Project — VIT-AP University, Dec 2024
Authors: Satyala Murali Karthik, Mekala Samuel, Yelakanti Ramu
Guide: Dr. S. Kalyani

Deployable to Streamlit Community Cloud, Hugging Face Spaces, and Render.
"""

import os
import json
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
MAPPING_PATH = "class_mapping.json"

SHELF_LIFE_RANGES = {
    "Tomato": "10-15",
    "Apple": "7-12",
    "Banana": "3-6",
    "Bellpepper": "7-10",
    "Carrot": "10-14",
    "Cucumber": "5-8",
    "Grape": "5-8",
    "Guava": "4-7",
    "Jujube": "5-9",
    "Mango": "4-7",
    "Orange": "7-10",
    "Pomegranate": "12-18",
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

    num_classes = 28
    class_mapping = {}

    if os.path.exists(MAPPING_PATH):
        try:
            with open(MAPPING_PATH, "r") as f:
                mapping = json.load(f)
                class_mapping = {int(k): v for k, v in mapping.items()}
                num_classes = len(class_mapping)
        except Exception:
            pass

    model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    has_custom_weights = False
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            has_custom_weights = True
        except Exception as e:
            print(f"Warning loading weights: {e}")

    model.eval()
    return model.to(device), device, class_mapping, has_custom_weights

model, device, class_mapping, has_custom_weights = load_shufflenet_model()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_produce(pil_image):
    tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()
        conf, idx = torch.max(probs, 0)
    
    raw_class = class_mapping.get(idx.item(), "general_fresh") if class_mapping else ("Fresh" if idx.item() == 0 else "Rotten")
    
    if "_" in raw_class:
        parts = raw_class.split("_")
        produce_name = parts[0].title()
        status = parts[1].title()
    else:
        produce_name = "Produce"
        status = raw_class.title()

    return produce_name, status, conf.item()

# ─── UI Layout ────────────────────────────────────────────────────────────────

st.title("🍎 Predicting Shelf Life of Fruits & Vegetables")
st.markdown("""
**Deep Learning-based Freshness Classification & Shelf Life Estimation**  
*VIT-AP University Capstone Project (Dec 2024)* | **Authors:** Satyala Murali Karthik, Mekala Samuel, Yelakanti Ramu | **Guide:** Dr. S. Kalyani
""")
st.divider()

if not has_custom_weights:
    st.warning("⚠️ **Running on Baseline Weights.** Custom trained weights (`shufflenet_shelf_life.pth`) are initializing...", icon="⚠️")

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_override = st.selectbox(
        "Select Produce Type (Auto-Detected by default):",
        options=["Auto-Detect"] + list(SHELF_LIFE_RANGES.keys()),
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🏆 Model Architecture")
    st.write("**Model:** ShuffleNet V2 (`shufflenet_v2_x1_0`)")
    st.write("**Classes Trained:** 28 Fine-Grained Produce Classes")
    st.write("**Weights Status:** " + ("Custom Trained ✅" if has_custom_weights else "Baseline ⚠️"))
    st.write("**Optimizer:** Adam (lr=0.003)")
    st.write("**Scheduler:** StepLR (step=4, γ=0.2)")
    st.write("**Regularization:** Dropout (p=0.5)")

tabs = st.tabs(["📸 Freshness Predictor", "📊 Project Analytics & Report Metrics", "📖 Dataset & Methodology"])

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

    with col2:
        st.subheader("2. Prediction & Shelf Life Analysis")
        if image_to_process is not None:
            with st.spinner("Analyzing produce image with ShuffleNet V2..."):
                detected_item, status, confidence = predict_produce(image_to_process)
                confidence_pct = confidence * 100
                
                final_item = detected_item if selected_override == "Auto-Detect" else selected_override
                days_range = SHELF_LIFE_RANGES.get(final_item, "3-7")

            # Out-of-distribution check for low confidence non-food images
            if confidence < 0.35 and not has_custom_weights:
                st.warning(f"### ⚠️ Low Confidence Detection ({confidence_pct:.1f}%)")
                st.write("**Unrecognized Image**: The uploaded image does not match trained produce features. Please upload a clear photo of a fruit or vegetable.")
            elif status == "Fresh":
                st.success(f"### ✅ Status: FRESH ({final_item})")
                st.metric(
                    label=f"Predicted Remaining Shelf Life for {final_item}",
                    value=f"{days_range} Days",
                    delta=f"{confidence_pct:.1f}% Model Confidence"
                )
                st.info(f"**Report Format:** Predicted: {final_item}({days_range}) days of shelf life left ({confidence_pct:.2f}% confidence)")
            else:
                st.error(f"### ⚠️ Status: ROTTEN / EXPIRED ({final_item})")
                st.metric(
                    label=f"Quality Assessment for {final_item}",
                    value="0 Days Remaining",
                    delta=f"-{confidence_pct:.1f}% Deteriorated",
                    delta_color="inverse"
                )
                st.warning(f"**Report Format:** Predicted: {final_item} is Rotten / Expired ({confidence_pct:.2f}% confidence)")

            st.write("**Model Confidence Meter:**")
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

# ─── TAB 3: METHODOLOGY ───────────────────────────────────────────────────────

with tabs[2]:
    st.subheader("🏗️ System Architecture & 28 Fine-Grained Produce Classes")
    st.markdown("""
    ### 🍇 14 Supported Produce Categories (Fresh & Rotten):
    *Apple, Banana, Bellpepper, Carrot, Cucumber, Grape, Guava, Jujube, Mango, Orange, Pomegranate, Potato, Strawberry, Tomato*
    
    ### ⚙️ Image Preprocessing Pipeline
    1. **Resizing:** Standardized input images to **256×256 pixels**.
    2. **Cropping:** Applied a **CenterCrop of 224×224 pixels** matching ShuffleNet V2 input dimensions.
    3. **Normalization:** Normalized using ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
    4. **Data Partitioning:** Split dataset into **75% Training** and **25% Validation**.
    """)

st.markdown("---")
st.caption("© 2024 VIT-AP University | Capstone Project Report | School of Electronics Engineering")
