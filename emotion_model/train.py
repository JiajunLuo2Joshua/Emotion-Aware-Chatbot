import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from sklearn.metrics import accuracy_score
import json

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc, model, path):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)
        if self.verbose:
            print(f" Validation acc improved, model saved to {path}")

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    preds = []
    targets = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds.extend(outputs.argmax(dim=1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    return running_loss / len(loader), acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    return running_loss / len(loader), acc

def create_emotion_model(model_name, num_classes):
    model = create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

if __name__ == '__main__':
    # configuration parameter
    DATA_DIR = "E:/COMPSYS731/dataset"
    SAVE_DIR = "checkpoints"
    NUM_CLASSES = 7
    BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f" Training on device: {DEVICE}")

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(10),      
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

    # Lodaing the full dataset
    full_train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    full_val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
    train_dataset = full_train_dataset
    val_dataset = full_val_dataset

    print(f" Using Full Dataset: Train samples {len(train_dataset)}, Val samples {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # train the 3 models
    model_list = ["efficientnet_b0", "mobilenetv2_100", "resnet50"]

    for model_name in model_list:
        print(f"\n=== Training model: {model_name} ===")
        model = create_emotion_model(model_name, NUM_CLASSES)
        model = model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.1)
        early_stopping = EarlyStopping(patience=7, verbose=True)

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        best_model_path = os.path.join(SAVE_DIR, f"best_model_{model_name}.pt")

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            scheduler.step(val_acc)
            early_stopping(val_acc, model, best_model_path)
            if early_stopping.early_stop:
                print(" Early stopping triggered. Training stopped.")
                break

        # save the training process
        with open(os.path.join(SAVE_DIR, f"training_history_{model_name}.json"), "w") as f:
            json.dump(history, f)
        # save the best model
        print(f" Training complete for {model_name}. History saved.")