import tensorflow as tf
import numpy as np
import cv2
import os
import json

# Constants
IMG_SIZE = 128
MODEL_DIR = "model/"
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

# ✅ Auto-load the latest trained model
if not os.path.exists(MODEL_DIR):
    print("❌ Error: Model directory not found! Ensure `train.py` has been run.")
    exit(1)

# Find the latest saved model file
model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")], reverse=True)
if not model_files:
    print("❌ No trained model found in `model/`. Run `train.py` first.")
    exit(1)

MODEL_PATH = os.path.join(MODEL_DIR, model_files[0])
print(f"🔄 Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Load class names
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
    print(f"📜 Loaded class names: {len(CLASS_NAMES)} classes")
else:
    print("⚠ Warning: `class_names.json` not found! Using 'Unknown'.")
    CLASS_NAMES = ["Unknown"]

def predict_image(image_path):
    """Predicts the class of a given image using the trained model."""
    
    # ❌ Validate input file
    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' not found!")
        return

    # 📸 Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not read the image '{image_path}'.")
        return
    
    # 🔄 Preprocess the Image
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (optional but recommended)
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # 🧠 Make Prediction
    predictions = model.predict(image)
    class_index = np.argmax(predictions)  # Get class index
    confidence = np.max(predictions)  # Get confidence score

    # Ensure valid class index
    predicted_class = CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else "Unknown"

    print(f"✅ Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")

# 🔍 Test Image Path
test_image_path = "C:/Users/ASUS/Desktop/plant/AppleScab3.jpg"
predict_image(test_image_path)
