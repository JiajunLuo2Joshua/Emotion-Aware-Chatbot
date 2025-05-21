import json
import matplotlib.pyplot as plt

# Load the saved training records
history_file = "emotion_model/checkpoints/training_history_full.json"

with open(history_file, "r") as f:
    history = json.load(f)

# Draw the Loss curve
plt.figure(figsize=(10, 5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("emotion_model/checkpoints/loss_curve.png")
plt.show()

# Draw the Accuracy curve
plt.figure(figsize=(10, 5))
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Val Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("emotion_model/checkpoints/accuracy_curve.png")
plt.show()
