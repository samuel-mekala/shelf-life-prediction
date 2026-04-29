"""
Shelf Life Prediction GUI
Tkinter-based app to predict freshness of fruits & vegetables
using the trained ShuffleNet V2 model.

Usage:
    python app.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models
import os

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH  = "shufflenet_shelf_life.pth"

# Map class indices to labels (update if your dataset has different class names)
CLASS_NAMES = {
    0: "Fresh",
    1: "Rotten",
}

# Estimated shelf life in days for fresh items (rough guide)
SHELF_LIFE_DAYS = {
    "Apple":      7,
    "Banana":     3,
    "Carrot":    10,
    "Grapes":     5,
    "Mango":      4,
    "Orange":     7,
    "Potato":    14,
    "Strawberry": 3,
    "Tomato":     5,
}

# ─── Model Loading ────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(num_classes=2):
    model = models.shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: {MODEL_PATH} not found. Using untrained model.")
    model.eval()
    return model.to(device)

model = load_model(num_classes=len(CLASS_NAMES))

# ─── Image Preprocessing ──────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(image_path):
    """Run inference on a single image. Returns (class_name, confidence)."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1).squeeze()
        conf, idx = torch.max(probs, 0)
    return CLASS_NAMES[idx.item()], conf.item()

# ─── GUI ──────────────────────────────────────────────────────────────────────

class ShelfLifeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Shelf Life Predictor 🍎")
        self.geometry("600x550")
        self.configure(bg="#f5f5f5")
        self._build_ui()

    def _build_ui(self):
        # Title
        tk.Label(self, text="Shelf Life Prediction",
                 font=("Helvetica", 18, "bold"),
                 bg="#f5f5f5", fg="#2c3e50").pack(pady=12)
        tk.Label(self, text="Upload an image of a fruit or vegetable",
                 font=("Helvetica", 11), bg="#f5f5f5", fg="#7f8c8d").pack()

        # Canvas for image preview
        self.canvas = tk.Canvas(self, width=300, height=300,
                                bg="#ecf0f1", bd=2, relief="groove")
        self.canvas.pack(pady=12)

        # Buttons
        btn_frame = tk.Frame(self, bg="#f5f5f5")
        btn_frame.pack()
        tk.Button(btn_frame, text="📂  Upload Image",
                  font=("Helvetica", 11), bg="#3498db", fg="white",
                  padx=12, pady=6, command=self._upload).grid(row=0, column=0, padx=8)
        tk.Button(btn_frame, text="🔍  Predict",
                  font=("Helvetica", 11), bg="#27ae60", fg="white",
                  padx=12, pady=6, command=self._predict).grid(row=0, column=1, padx=8)

        # Result label
        self.result_var = tk.StringVar(value="No prediction yet.")
        tk.Label(self, textvariable=self.result_var,
                 font=("Helvetica", 13), bg="#f5f5f5",
                 fg="#2c3e50", wraplength=500, justify="center").pack(pady=16)

        self.image_path = None

    def _upload(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if path:
            self.image_path = path
            # Display image in canvas
            img = Image.open(path).resize((300, 300))
            self._tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
            self.result_var.set("Image loaded. Click Predict.")

    def _predict(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        label, confidence = predict(self.image_path)
        pct = confidence * 100
        if label == "Fresh":
            # Try to infer item name from filename for shelf-life hint
            name = os.path.splitext(os.path.basename(self.image_path))[0].title()
            days = SHELF_LIFE_DAYS.get(name, "~3–7")
            msg = (f"✅  Predicted: {label}  ({pct:.1f}% confidence)\n"
                   f"Estimated shelf life remaining: {days} days")
        else:
            msg = (f"⚠️  Predicted: {label}  ({pct:.1f}% confidence)\n"
                   f"This item may no longer be safe to consume.")
        self.result_var.set(msg)


if __name__ == "__main__":
    app = ShelfLifeApp()
    app.mainloop()
