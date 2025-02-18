import json
import matplotlib.pyplot as plt

# ✅ Load saved history
history_path = "model/training_history.json"
with open(history_path, "r") as f:
    history = json.load(f)

# ✅ Function to plot training history
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

# ✅ Plot the history
plot_training_history(history)
