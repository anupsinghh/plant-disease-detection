import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load the trained model
model = load_model("model.h5")

# Define image size (must match training size)
IMG_SIZE = (224, 224)  # Change if your model used a different size
BATCH_SIZE = 32

# Define dataset path (Use validation folder)
dataset_path = "PlantVillage/val"  # Update this path

# Load validation dataset using ImageDataGenerator
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Keep False to match labels correctly
)

# Get class labels
class_labels = list(val_generator.class_indices.keys())

# Predict probabilities for validation data
y_pred_probs = model.predict(val_generator)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred_probs, axis=1)  # Predicted class labels
y_true = val_generator.classes  # True labels

# Print accuracy, precision, recall, and F1-score
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average="weighted"))
print("Recall:", recall_score(y_true, y_pred, average="weighted"))
print("F1-score:", f1_score(y_true, y_pred, average="weighted"))

# Print detailed classification report
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

# Save results to a file
with open("classification_report.txt", "w") as f:
    f.write(report)

print("Classification report saved as classification_report.txt")
