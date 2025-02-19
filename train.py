import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from datetime import datetime

# ✅ Check GPU Availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Available: {gpus}")
else:
    print("⚠️ No GPU found, using CPU.")

# ✅ Ensure the model directory exists
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# ✅ Dataset path
dataset_path = "dataset/"

# ✅ Image Parameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10  # 🔹 Increased for better learning

# ✅ Improved Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,  
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,  
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  
    fill_mode='nearest'
)

# ✅ Load images with augmentation
train_data = datagen.flow_from_directory(
    dataset_path, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', subset="training")

val_data = datagen.flow_from_directory(
    dataset_path, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', subset="validation")

# ✅ Save class names for `test.py`
class_names = list(train_data.class_indices.keys())
class_names_path = os.path.join(model_dir, "class_names.json")
with open(class_names_path, "w") as f:
    json.dump(class_names, f)
print(f"✅ Class Labels Saved: {train_data.class_indices}")

# ✅ Load Pretrained ResNet50 (excluding top layers)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# ✅ Unfreeze last 50 layers for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# ✅ Custom Fully Connected Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)  
x = Dense(256, activation="relu")(x)
output = Dense(len(class_names), activation="softmax")(x)

# ✅ Define the Model
model = Model(inputs=base_model.input, outputs=output)

# ✅ Compile Model with an improved learning rate
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ Early stopping and Learning Rate Scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# ✅ Train Model on GPU if available
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[early_stopping, lr_scheduler])

# ✅ Save training history
history_path = os.path.join(model_dir, "training_history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f)
print(f"📊 Training history saved to {history_path}")

# ✅ Save model with timestamp
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = os.path.join(model_dir, f"plant_{timestamp}.h5")
model.save(model_path)
print(f"✅ Model saved successfully at {model_path}")

# ✅ Function to Plot Training History
def plot_training_history(history):
    """Plots the training and validation accuracy/loss over epochs."""
    plt.figure(figsize=(12, 5))

    # 🔹 Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Train Accuracy", marker='o')
    plt.plot(history["val_accuracy"], label="Validation Accuracy", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("📈 Model Accuracy")
    plt.legend()

    # 🔹 Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train Loss", marker='o')
    plt.plot(history["val_loss"], label="Validation Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("📉 Model Loss")
    plt.legend()

    plt.show()

# ✅ Call function to plot graphs
plot_training_history(history.history)
