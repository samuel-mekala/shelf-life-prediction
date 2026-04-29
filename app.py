"""
Tkinter GUI for Shelf Life Prediction
Author: Samuel Mekala | VIT-AP University
Usage : python gui/app.py
"""
import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas
from PIL import Image, ImageTk
import torch, torch.nn as nn, numpy as np
from torchvision import transforms, models

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/shufflenet_shelf_life.pt"
CLASS_NAMES = ["Fresh", "NearingExpiry", "Rotten"]   # update to match your dataset classes
SHELF_DAYS  = {"Fresh": "5-7 days remaining", "NearingExpiry": "1-2 days remaining", "Rotten": "Expired"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ─────────────────────────────────────────────────────────────
def load_model():
    m = models.shufflenet_v2_x1_0(weights=None)
    m.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(m.fc.in_features, len(CLASS_NAMES)))
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.eval()
    return m.to(device)

model = load_model()

transform = transforms.Compose([
    transforms.Resize((256,256)), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ── Prediction ─────────────────────────────────────────────────────────────
def predict(img_path):
    img   = Image.open(img_path).convert("RGB")
    inp   = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out  = model(inp)
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]
    idx        = int(np.argmax(prob))
    class_name = CLASS_NAMES[idx]
    confidence = prob[idx] * 100
    return img, class_name, confidence

# ── GUI ─────────────────────────────────────────────────────────────────────
class ShelfLifeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shelf Life Predictor")
        self.root.geometry("500x600")
        self.root.configure(bg="#f0f0f0")

        Label(root, text="Shelf Life Prediction", font=("Arial",18,"bold"), bg="#f0f0f0").pack(pady=10)
        self.canvas = Canvas(root, width=300, height=300, bg="white", relief="sunken", bd=2)
        self.canvas.pack(pady=10)
        Button(root, text="Upload Image", command=self.upload_image,
               bg="#4CAF50", fg="white", font=("Arial",12), padx=20, pady=8).pack(pady=10)
        self.result_label   = Label(root, text="", font=("Arial",14,"bold"), bg="#f0f0f0")
        self.result_label.pack(pady=5)
        self.shelf_label    = Label(root, text="", font=("Arial",12), bg="#f0f0f0", fg="#555")
        self.shelf_label.pack(pady=5)
        self.conf_label     = Label(root, text="", font=("Arial",11), bg="#f0f0f0", fg="#777")
        self.conf_label.pack(pady=5)

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        img, class_name, confidence = predict(path)
        img_tk = ImageTk.PhotoImage(img.resize((300,300)))
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk
        color = {"Fresh":"#2ecc71", "NearingExpiry":"#f39c12", "Rotten":"#e74c3c"}.get(class_name, "black")
        self.result_label.config(text=f"Prediction: {class_name}", fg=color)
        self.shelf_label.config(text=SHELF_DAYS.get(class_name, ""))
        self.conf_label.config(text=f"Confidence: {confidence:.1f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app  = ShelfLifeApp(root)
    root.mainloop()
