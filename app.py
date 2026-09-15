"""
Shelf Life Prediction GUI (Tkinter Desktop Application)
Predicts freshness of fruits & vegetables using trained ShuffleNet V2 model.
Aligned with Capstone Project Report Requirements (VIT-AP, Dec 2024).

Usage:
    python app.py
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = "shufflenet_shelf_life.pth"

CLASS_NAMES = {
    0: "Fresh",
    1: "Rotten",
}

# Shelf life ranges in days for various fresh produce items
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

# ─── Model Loading ────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def load_model(num_classes=2):
    model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Could not load state dict ({e}). Using initialized ShuffleNet V2.")
    else:
        print(f"Notice: {MODEL_PATH} not found. Running with ShuffleNet V2 baseline weights.")
    model.eval()
    return model.to(device)

model = load_model(num_classes=len(CLASS_NAMES))

# ─── Preprocessing ────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()
        conf, idx = torch.max(probs, 0)
    return CLASS_NAMES[idx.item()], conf.item()

# ─── GUI Application ──────────────────────────────────────────────────────────

class ShelfLifeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Classifier - Shelf Life Prediction")
        self.geometry("620x650")
        self.configure(bg="#ffffff")
        self.image_path = None
        self._build_ui()

    def _build_ui(self):
        # Header / Title
        header_frame = tk.Frame(self, bg="#ffffff")
        header_frame.pack(fill="x", pady=15)
        
        tk.Label(
            header_frame,
            text="Select an Image",
            font=("Helvetica", 16, "bold"),
            bg="#ffffff",
            fg="#1a1a1a"
        ).pack()

        # Produce Item Selector Dropdown
        item_frame = tk.Frame(self, bg="#ffffff")
        item_frame.pack(pady=5)
        
        tk.Label(
            item_frame,
            text="Produce Type: ",
            font=("Helvetica", 11),
            bg="#ffffff",
            fg="#444444"
        ).pack(side="left")
        
        self.item_var = tk.StringVar(value="Tomato")
        item_choices = list(SHELF_LIFE_RANGES.keys())
        self.item_dropdown = ttk.Combobox(
            item_frame,
            textvariable=self.item_var,
            values=item_choices,
            state="readonly",
            width=18
        )
        self.item_dropdown.pack(side="left")

        # Choose Image Button
        btn_frame = tk.Frame(self, bg="#ffffff")
        btn_frame.pack(pady=10)
        
        tk.Button(
            btn_frame,
            text="Choose Image",
            font=("Helvetica", 11),
            bg="#e0e0e0",
            fg="#000000",
            relief="groove",
            padx=15,
            pady=5,
            command=self._choose_image
        ).pack()

        # Image Display Frame / Canvas
        self.image_frame = tk.Frame(self, width=280, height=280, bg="#f0f0f0", bd=1, relief="solid")
        self.image_frame.pack_propagate(False)
        self.image_frame.pack(pady=15)

        self.img_label = tk.Label(self.image_frame, bg="#f0f0f0")
        self.img_label.pack(expand=True, fill="both")

        # Result Display Box
        self.result_var = tk.StringVar(value="Predicted: Select an image to predict shelf life")
        self.result_label = tk.Label(
            self,
            textvariable=self.result_var,
            font=("Helvetica", 12, "bold"),
            bg="#ffffff",
            fg="#222222",
            wraplength=550,
            justify="center"
        )
        self.result_label.pack(pady=20)

    def _choose_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff")]
        )
        if not path:
            return

        self.image_path = path
        
        # Display image preview
        img = Image.open(path).convert("RGB")
        img.thumbnail((260, 260))
        self._tk_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self._tk_img)

        # Infer produce item from filename if known, otherwise use selected dropdown value
        filename_base = os.path.splitext(os.path.basename(path))[0].title()
        for known_item in SHELF_LIFE_RANGES.keys():
            if known_item.lower() in filename_base.lower():
                self.item_var.set(known_item)
                break

        # Run Prediction immediately upon image selection (matching Report behavior)
        self._run_prediction()

    def _run_prediction(self):
        if not self.image_path:
            return
            
        label, confidence = predict_image(self.image_path)
        confidence_pct = confidence * 100
        selected_item = self.item_var.get()
        days_range = SHELF_LIFE_RANGES.get(selected_item, "3-7")

        if label == "Fresh":
            # Exact format specified in Page 19 of Capstone Report
            output_msg = f"Predicted: {selected_item}({days_range}) days of shelf life left ({confidence_pct:.2f}% confidence)"
            self.result_label.configure(fg="#2e7d32")  # Dark Green
        else:
            output_msg = f"Predicted: {selected_item} is Rotten / Expired ({confidence_pct:.2f}% confidence)"
            self.result_label.configure(fg="#c62828")  # Dark Red

        self.result_var.set(output_msg)

if __name__ == "__main__":
    app = ShelfLifeApp()
    app.mainloop()
