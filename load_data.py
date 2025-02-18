import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 📂 Define dataset paths
train_dir = "dataset/train"
valid_dir = "dataset/valid"
test_dir = "dataset/test"

# 🛠️ Data Augmentation for Training (helps prevent overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,     # Normalize pixel values
    rotation_range=20,  # Randomly rotate images
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ✅ Validation and Testing: Only rescaling (No augmentation)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 📥 Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Resize images
    batch_size=32,
    class_mode="categorical"
)

# 📥 Load validation data
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

# 📥 Load test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

# ✅ Check class names (to ensure correct mapping)
print("Class Indices:", train_generator.class_indices)
