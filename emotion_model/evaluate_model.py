import os
import torch
import torch.nn as nn
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import numpy as np

# Configuration
#DATA_DIR = os.path.join("..", "data", "emotion_dataset", "val")
DATA_DIR = "E:/COMPSYS731/data/data/emotion_dataset/val"
CHECKPOINT_DIR = "E:/ljj22/Documents/Github/cs731-2025-project-group-8/emotion_model/emotion_model/checkpoints"
#CHECKPOINT_DIR = os.path.join("..", "emotion_model", "checkpoints")
NUM_CLASSES = 7  # Ekman theory 
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on device: {DEVICE}")

emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# load the validation dataset
val_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# Ensure the checkpoint directory exists
models_to_test = [
    ("EfficientNet", 'efficientnet_b0', os.path.join(CHECKPOINT_DIR, 'best_model_efficientnet_b0.pt')),
    ("MobileNetV2", 'mobilenetv2_100', os.path.join(CHECKPOINT_DIR, 'best_model_mobilenetv2_100.pt')),
    ("ResNet50", 'resnet50', os.path.join(CHECKPOINT_DIR, 'best_model_resnet50.pt')),
]

for model_name, timm_name, weight_path in models_to_test:
    print(f"\n--- Evaluating {model_name} ---")
    # Load the model structure
    model = create_model(timm_name, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    total_time = 0
    total_images = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            start = time.time()
            outputs = model(inputs)
            end = time.time()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_time += (end - start)
            total_images += inputs.size(0)

    # Classification report (including the accuracy rates of various emotions)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=emotion_labels, digits=4))

    # Overall accuracy rate
    acc = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy for {model_name}: {acc:.4f}")

    # Model size in MB
    size_mb = os.path.getsize(weight_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")

    # Average inference time and FPS
    avg_time_per_image = (total_time / total_images) * 1000 if total_images > 0 else 0  # ms/张
    fps = total_images / total_time if total_time > 0 else 0
    print(f"Average inference time: {avg_time_per_image:.2f} ms/image")
    print(f"FPS: {fps:.2f} images/second")

    # Draw the confusion matrix
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_labels, yticklabels=emotion_labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{model_name} Confusion Matrix")
        plt.show()
    except ImportError:
        print("seaborn or matplotlib not installed; skipping confusion matrix.")

    print("-" * 40)