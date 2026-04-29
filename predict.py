"""
Single Image Prediction — loads any saved model and predicts plant disease
Author: Samuel Mekala | VIT-AP University
Usage : python predict.py --model models/GoogleNetModel.h5 --image test_leaf.jpg
"""
import argparse, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

CLASS_NAMES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def predict(model_path, image_path, target_size=(120,120)):
    model = load_model(model_path)
    img   = image.load_img(image_path, target_size=target_size)
    arr   = image.img_to_array(img) / 255.0
    arr   = np.expand_dims(arr, axis=0)
    pred  = model.predict(arr)
    if isinstance(pred, list):   # GoogLeNet returns list
        pred = pred[0]
    pred = pred.flatten()
    idx  = np.argmax(pred)
    print(f"Predicted Class : {CLASS_NAMES[idx]}")
    print(f"Confidence      : {pred[idx]*100:.2f}%")
    plt.figure(figsize=(4,4)); plt.imshow(img); plt.axis('off'); plt.title(CLASS_NAMES[idx]); plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved .h5 model")
    parser.add_argument("--image", required=True, help="Path to leaf image")
    args = parser.parse_args()
    predict(args.model, args.image)
