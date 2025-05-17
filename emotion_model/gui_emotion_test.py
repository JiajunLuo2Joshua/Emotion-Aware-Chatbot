import cv2
import torch
import torchvision.transforms as transforms
from timm import create_model
from ultralytics import YOLO
import numpy as np
import os

# absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------ CONFIG ------------------------
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints', 'best_model_full.pt')
FACE_MODEL_PATH = os.path.join(BASE_DIR, '..', 'face_detection', 'yolov8n-face.pt')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
KEY_LABEL_MAP = {
    'a': 'Anger',
    'g': 'Disgust',
    'f': 'Fear',
    'h': 'Happy',
    'n': 'Neutral',
    'd': 'Sad',
    's': 'Surprise'
}

# ------------------------ LOAD MODELS ------------------------
# Emotion recognition model
emotion_model = create_model('efficientnet_b0', pretrained=False, num_classes=len(LABELS))
emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=DEVICE))
emotion_model.to(DEVICE)
emotion_model.eval()

# Face detection model (YOLOv8n-face)
face_model = YOLO(FACE_MODEL_PATH)

# ------------------------ TRANSFORM ------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------ METRICS ------------------------
total = 0
correct = 0

# ------------------------ CAMERA LOOP ------------------------
cap = cv2.VideoCapture(0)
print("Press:")
for k, v in KEY_LABEL_MAP.items():
    print(f"'{k}' → {v}")
print("Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = face_model(frame, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # Run emotion prediction
        try:
            face_tensor = transform(face).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = emotion_model(face_tensor)
                probs = torch.softmax(output, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                pred_label = LABELS[pred_idx]
                confidence = probs[pred_idx].item()

            # Draw face box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{pred_label} ({confidence * 100:.1f}%)"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            print(f"[Error] Emotion prediction failed: {e}")
            continue

    # Display accuracy
    if total > 0:
        acc_text = f"Accuracy: {correct}/{total} = {correct / total * 100:.1f}%"
        cv2.putText(frame, acc_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Face + Emotion Recognition", frame)

    # Keyboard input
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key != -1:
        try:
            key_char = chr(key).lower()
            if key_char in KEY_LABEL_MAP:
                true_label = KEY_LABEL_MAP[key_char]
                total += 1
                if pred_label == true_label:
                    correct += 1
                print(f"True: {true_label} | Predicted: {pred_label} | {'✔️' if pred_label == true_label else '❌'}")
        except:
            continue

cap.release()
cv2.destroyAllWindows()
