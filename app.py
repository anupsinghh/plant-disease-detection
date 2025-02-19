from flask import Flask, request, render_template
import cv2
import numpy as np
import os
import glob
import time
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ✅ Automatically get the latest model file from 'model' directory
MODEL_DIR = os.path.join(os.getcwd(), "model")
model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.h5")), key=os.path.getmtime, reverse=True)

if model_files:
    MODEL_PATH = model_files[0]  # Get the most recent model file
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
else:
    print("❌ Error: No model file found in the 'model' directory!")
    model = None

# ✅ Get class labels from dataset folder
DATASET_PATH = os.path.join(os.getcwd(), "dataset", "train")  # Update this path if needed
if os.path.exists(DATASET_PATH):
    class_labels = sorted(os.listdir(DATASET_PATH))  # Folder names as labels
else:
    print("⚠️ Warning: Dataset folder not found! Using default labels.")
    class_labels = ["Unknown"]

# ✅ Keep image size consistent with training
IMG_SIZE = 128  

def predict_image(image_path):
    """Predict the class of an uploaded image."""
    if model is None:
        return "Error: Model not loaded!"

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not read image {image_path}.")
        return "Error reading image"
    
    # ✅ Resize to the correct input size (128x128)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # ✅ Make prediction
    prediction = model.predict(image)
    print(f"🔍 Raw Predictions: {prediction}")  # Debugging output

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # ✅ Ensure valid class index
    predicted_class = class_labels[class_index] if class_index < len(class_labels) else "Unknown"

    return f"{predicted_class} (Confidence: {confidence:.2f})"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        
        # ✅ Generate a unique filename to prevent browser caching issues
        timestamp = int(time.time())
        file_path = f"static/uploaded_{timestamp}.jpg"
        file.save(file_path)
        
        result = predict_image(file_path)
        return render_template("index.html", result=result, image=file_path)

    return render_template("index.html", result=None, image=None)

if __name__ == "__main__":
    app.run(debug=True)
