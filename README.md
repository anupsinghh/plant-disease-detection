# 🌱 Plant Disease Detection using Deep Learning  

![Project Banner](https://your-image-link.com)  

## 📌 Overview  
This project detects plant diseases using **Deep Learning (ResNet50)**.  
We utilize **TensorFlow, Keras, and OpenCV** for training and classification.  

---

## 🔥 Features  
✅ **Pretrained ResNet50 Model** for feature extraction  
✅ **Image Augmentation** for improved generalization  
✅ **Categorical Classification** for multiple plant diseases  
✅ **Training with GPU Acceleration (NVIDIA)**  
✅ **Visualization (Training Graphs, Heatmaps, Predictions)**  

---

## 📂 Project Structure  
```bash
Plant-Disease-Detection/
│── dataset/                 # Training images (organized in subfolders)
│── model/                   # Saved models and class names
│── test.py                   # Testing the trained model
│── train.py                  # Model training script
│── app.py                    # Web API for prediction (Flask/FastAPI)
│── load_data.py              # Preprocessing and dataset handling
│── plot_history.py           # Training loss & accuracy visualization
│── requirements.txt          # Dependencies list
│── README.md                 # Project documentation
│── .gitignore                # Files to ignore in GitHub
