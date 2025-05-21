import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# configure
DATA_DIR = "/Users/owenshi/github warehouse/cs731-2025-project-group-8/data/emotion_dataset/val"
MODEL_PATH = "/Users/owenshi/github warehouse/cs731-2025-project-group-8/emotion_model/checkpoints/best_model_full.pt"
NUM_CLASSES = 7  # Ekman theory 
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on device: {DEVICE}")

# labels name
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

# load the model
model = create_model('efficientnet_b0', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# reasoning assessment
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# print the evaluation results
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=emotion_labels))

# display the confusion matrix
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
except ImportError:
    print("seaborn or matplotlib not installed; skipping confusion matrix.")
