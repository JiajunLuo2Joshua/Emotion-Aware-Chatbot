import cv2
import torch
import torchvision.transforms as transforms
from timm import create_model
from ultralytics import YOLO
import numpy as np
import os
import time
from openai import OpenAI
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
from PIL import Image

os.environ["OPENAI_API_KEY"] = "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"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints', 'best_model_full.pt')
FACE_MODEL_PATH = os.path.join(BASE_DIR, '..', 'face_detection', 'yolov8n-face.pt')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

emotion_model = create_model('efficientnet_b0', pretrained=False, num_classes=len(LABELS))
emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=DEVICE))
emotion_model.to(DEVICE)
emotion_model.eval()

face_model = YOLO(FACE_MODEL_PATH)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

client = OpenAI()

def build_prompt(emotion):
    return (
        f"The user appears to be experiencing the following emotion: {emotion}.\n"
        f"Please generate a concise and gentle sentence based on this emotion, suitable to say to a lonely elderly person."
    )

def send_to_chatgpt(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a gentle companion chatbot for the elderly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print("[ERROR] ChatGPT request failed:", e)
        return "(Failed to generate the response)"

class EmotionCompanion(QtWidgets.QWidget):
    def __init__(self, log_interval=5):
        super().__init__()
        self.setWindowTitle("Emotion Companion")
        self.setFixedSize(700, 600)

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setGeometry(20, 20, 640, 480)

        self.response_label = QtWidgets.QLabel(self)
        self.response_label.setGeometry(20, 520, 660, 60)
        self.response_label.setWordWrap(True)
        self.response_label.setStyleSheet("font-size: 14px;")

        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.state = {
            "last_log_time": time.time(),
            "current_emotion": None,
            "log_interval": log_interval
        }

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results = face_model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            try:
                face_tensor = transform(face).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = emotion_model(face_tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    pred_idx = torch.argmax(probs).item()
                    pred_label = LABELS[pred_idx]
                    confidence = probs[pred_idx].item()

                self.state["current_emotion"] = pred_label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{pred_label} ({confidence * 100:.1f}%)"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"[Error] Emotion prediction failed: {e}")
                continue

        now = time.time()
        if self.state["current_emotion"] and now - self.state["last_log_time"] >= self.state["log_interval"]:
            self.state["last_log_time"] = now

            print("[LOGGED EMOTION]:", self.state["current_emotion"])
            prompt = build_prompt(self.state["current_emotion"])
            print("[PROMPT]:", prompt)

            response = send_to_chatgpt(prompt)
            print("[RESPONSE]:", response)

            self.response_label.setText(response)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480))
        qt_img = QtGui.QImage(img.tobytes(), img.width, img.height, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


def run_emotion_capture_gui(log_interval=5):
    app = QtWidgets.QApplication(sys.argv)
    window = EmotionCompanion(log_interval=log_interval)
    window.show()
    sys.exit(app.exec_())
