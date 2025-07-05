from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# oad Model
MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define Class Labels (trained with Plant___Disease format)
class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
                'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy']

IMG_SIZE = (224, 224)

def predict_image(img_path):
    """Load and predict image class, returning only disease name"""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    if "___" in predicted_class:
        disease_name = predicted_class.split("___")[1].replace("_", " ")
    else:
        disease_name = predicted_class.replace("_", " ")

    return disease_name, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            file_path = os.path.join("static", image_file.filename)
            image_file.save(file_path)

            # Get Prediction (disease name only)
            predicted_label, confidence = predict_image(file_path)
            return render_template("index.html", image_path=file_path, label=predicted_label, confidence=confidence)

    return render_template("index.html", image_path=None)

if __name__ == "__main__":
    app.run(debug=True)
