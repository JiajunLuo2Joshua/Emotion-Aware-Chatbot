import cv2
import torch
import torchvision.transforms as transforms
from timm import create_model
from ultralytics import YOLO
import numpy as np
import os
import time
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "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"  

# ------------------------ CONFIG ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMOTION_MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints', 'best_model_full.pt')
FACE_MODEL_PATH = os.path.join(BASE_DIR, '..', 'face_detection', 'yolov8n-face.pt')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ------------------------ LOAD MODELS (ONCE) ------------------------
emotion_model = create_model('efficientnet_b0', pretrained=False, num_classes=len(LABELS))
emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=DEVICE))
emotion_model.to(DEVICE)
emotion_model.eval()

face_model = YOLO(FACE_MODEL_PATH)

# ------------------------ TRANSFORM ------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------ FUNCTION MODULE ------------------------

def run_emotion_capture(log_interval=5):
    client = OpenAI()
    emotion_log = []
    last_log_time = time.time()
    current_emotion = None

    def build_prompt(emotions):
        if not emotions:
            return "Please say a caring word as a gentle companion of an elderly person."
        summary = ", ".join(emotions)
        return (
            f"The user's recent emotional records are as follows: {summary}.\n"
            f"Please generate a concise and gentle sentence based on these emotions, suitable to say to a lonely elderly person."
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

    # ------------------------ CAMERA LOOP ------------------------
    cap = cv2.VideoCapture(0)
    print("[INFO] Press ESC to exit.")

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

            try:
                face_tensor = transform(face).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = emotion_model(face_tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    pred_idx = torch.argmax(probs).item()
                    pred_label = LABELS[pred_idx]
                    confidence = probs[pred_idx].item()

                current_emotion = pred_label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{pred_label} ({confidence * 100:.1f}%)"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            except Exception as e:
                print(f"[Error] Emotion prediction failed: {e}")
                continue

        now = time.time()
        if current_emotion and now - last_log_time >= log_interval:
            emotion_log.append(current_emotion)
            last_log_time = now
            print(f"[LOG] Emotion recorded: {current_emotion}")
            if len(emotion_log) > 6:
                emotion_log.pop(0)

        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # ChatGPT Output
    print("\n[INFO] Final emotion list:")
    print(emotion_log)

    prompt = build_prompt(emotion_log)
    print("\n[INFO] ChatGPT Prompt:")
    print(prompt)

    response = send_to_chatgpt(prompt)
    print("\n[ChatGPT Response]:")
    print(response)
