import sys
import time
import os
import torch
import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtWidgets, QtGui, QtCore
from timm import create_model
from ultralytics import YOLO
from collections import Counter
import openai
from chat_memory import ChatMemory
import voice_choice
from voice_input_handler import VoiceInputHandler
from functools import partial
from chat_bubble import ChatBubble
from ui_layout import setup_ui
import threading

# Load models and constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints', 'best_model_full.pt')
FACE_MODEL_PATH = os.path.join(BASE_DIR, '..', 'face_detection', 'yolov8n-face.pt')
LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load emotion classification model
emotion_model = create_model('efficientnet_b0', pretrained=False, num_classes=len(LABELS))
emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=DEVICE))
emotion_model.to(DEVICE)
emotion_model.eval()

# Load face detection model
face_model = YOLO(FACE_MODEL_PATH)

# Placeholder transform
transform = torch.nn.Sequential(torch.nn.Identity())

# Load OpenAI API key and initialize client
api_key_path = os.path.join(BASE_DIR, "openai_key.txt")
def load_api_key(file_path=api_key_path):
    with open(file_path, "r") as f:
        return f.read().strip()

openai.api_key = load_api_key()
client = openai.OpenAI(api_key=openai.api_key)

# Initialize chat memory with system prompt
system_prompt = (
    "You are a warm, emotionally supportive, and conversational assistant. "
    "Your goal is to gently comfort and accompany elderly users who may be feeling anxious, lonely, or upset. "
    "Speak in a natural, human-like way—avoid listing items or giving advice in numbered or bullet-point format. "
    "Instead, focus on having a warm, flowing conversation, showing empathy and gentle curiosity. "
    "Offer encouragement with kind words and emotional presence. "
    "If the user seems very distressed, gently suggest they talk to someone they trust or a doctor. "
    "Use clear, simple language, and prioritize emotional connection over structured advice. "
    "Only provide the main advice, no more than 5 sentences."
)
memory = ChatMemory(system_prompt)

class EmotionChatApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion-Aware Companion Chatbot")
        self.resize(1280, 960)
        setup_ui(self)

        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.vote_buffer = []
        self.vote_start_time = time.time()
        self.vote_window_seconds = 2
        self.vote_sample_target = 100
        self.vote_threshold = 0.8
        self.final_emotion = "Unknown"
        self.pre_emotion = None
        self.recent_emotions = []

        self.voice_handler = VoiceInputHandler()
        self.voice_mode_active = False
        self.voice_loop_timer = QtCore.QTimer()
        self.voice_loop_timer.setSingleShot(True)
        self.voice_loop_timer.timeout.connect(self.voice_loop_step)

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
                face_tensor = transform(torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(DEVICE)
                with torch.no_grad():
                    output = emotion_model(face_tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    pred_idx = torch.argmax(probs).item()
                    pred_label = LABELS[pred_idx]
                self.vote_emotion(pred_label)
                label = f"{pred_label} ({probs[pred_idx].item()*100:.1f}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print("Emotion error:", e)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((960, 540))
        qt_img = QtGui.QImage(img.tobytes(), img.width, img.height, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_img))

    def vote_emotion(self, emotion):
        self.vote_buffer.append(emotion)
        now = time.time()
        duration = now - self.vote_start_time
        if duration >= self.vote_window_seconds or len(self.vote_buffer) >= self.vote_sample_target:
            counter = Counter(self.vote_buffer)
            most_common, count = counter.most_common(1)[0]
            ratio = count / len(self.vote_buffer)
            if self.final_emotion != self.pre_emotion:
                self.update_suggestion(self.final_emotion)
                self.update_emotion_history(self.final_emotion)
                self.pre_emotion = self.final_emotion
            if ratio >= self.vote_threshold:
                self.final_emotion = most_common
                self.emotion_label.setText(f"Emotion: {most_common}")
            self.vote_buffer = []
            self.vote_start_time = now

    def update_emotion_history(self, new_emotion):
        self.recent_emotions.append(new_emotion)
        if len(self.recent_emotions) > 6:
            self.recent_emotions.pop(0)
        emoji_map = {
            "Happy": "😊", "Sad": "😢", "Neutral": "😐",
            "Anger": "😠", "Disgust": "🤢", "Fear": "😨", "Surprise": "😲"
        }
        emoji_sequence = " → ".join(emoji_map.get(e, '') for e in self.recent_emotions)
        self.emotion_history_label.setText(f"Recent Emotions:\n {emoji_sequence}")

    def update_suggestion(self, emotion):
        suggestions = {
            "Happy": "💡 You seem joyful! What’s something fun or exciting that happened today?",
            "Sad": "💡 I’m here for you. Would you like to talk about something or someone that brings you comfort?",
            "Anger": "💡 It’s okay to feel frustrated. Want to tell me what happened or what’s been bothering you?",
            "Neutral": "💡 Just checking in — is there anything on your mind you'd like to share today?",
            "Surprise": "💡 That caught you off guard! Want to tell me what just happened?",
            "Fear": "💡 You’re not alone. Would it help to talk about what’s making you uneasy right now?",
            "Disgust": "💡 That didn’t sit right with you, huh? Want to switch topics or tell me what happened?"
        }
        self.suggestion_box.setText(suggestions.get(emotion, "💡 I'm here for you. Feel free to share anything on your mind."))

    def add_chat_bubble(self, role, message):
        bubble = ChatBubble(role, message)
        self.chat_layout.addWidget(bubble)
        QtCore.QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum()))

    def chatgpt_query(self, prompt):
        memory.add_user_input(prompt)
        try:
            response = client.chat.completions.create(model="gpt-4o", messages=memory.get_messages())
            reply = response.choices[0].message.content.strip()
            memory.add_assistant_response(reply)
            return reply
        except Exception as e:
            print("ChatGPT error:", e)
            return "(ChatGPT failed to respond.)"

    def handle_text_input(self):
        user_text = self.input_line.toPlainText().strip()
        if user_text:
            self.add_chat_bubble("user", user_text)
            self.input_line.clear()
            self.send_btn.setDisabled(True)
            self.input_line.setDisabled(True)
            self.toggle_btn.setDisabled(True)
            prompt = f"(The user seems to be feeling: {self.final_emotion}.)\n" + user_text
            reply = self.chatgpt_query(prompt)
            self.add_chat_bubble("bot", reply)
            voice_choice.voice_keyword = "United States"
            QtCore.QTimer.singleShot(100, lambda: self.speak_and_reenable(reply))

    def toggle_input_mode(self):
        if self.mode == "text":
            self.mode = "voice"
            self.toggle_btn.setText("⌨️ Switch to Text")
            self.voice_mode_active = True
            self.input_line.setDisabled(True)
            self.send_btn.setDisabled(True)
            voice_choice.speak("Switched to voice mode. Please speak.")
            self.voice_loop_timer.start(300)
        else:
            self.mode = "text"
            self.toggle_btn.setText("🎤 Switch to Voice")
            self.voice_mode_active = False
            self.voice_loop_timer.stop()
            self.input_line.setDisabled(False)
            self.send_btn.setDisabled(False)
            voice_choice.speak("Switched to text mode.")

    def voice_loop_step(self):
        if self.mode != "voice" or not self.voice_mode_active:
            return
        try:
            voice_reply = self.voice_handler.transcribe_and_respond(
                chat_fn=partial(EmotionChatApp.chatgpt_query, self),
                final_emotion=self.final_emotion,
                add_bubble_fn=self.add_chat_bubble
            )
            self.send_btn.setDisabled(True)
            self.input_line.setDisabled(True)
            self.toggle_btn.setDisabled(True)
            QtCore.QTimer.singleShot(100, lambda: self.speak_and_resume(voice_reply))
        except Exception as e:
            print("Voice input error:", e)
            voice_choice.speak("Something went wrong during voice input.")

    def speak_and_resume(self, reply):
        try:
            self.speak_and_reenable(reply)
        except Exception as e:
            print("Voice speak error:", e)
        if self.voice_mode_active:
            self.voice_loop_timer.start(300)

    def speak_and_reenable(self, reply):
        def speak_and_restore():
            try:
                voice_choice.speak(reply)
            except Exception as e:
                print("Voice speak error:", e)
            finally:
                QtCore.QTimer.singleShot(0, self.enable_input_buttons)
        threading.Thread(target=speak_and_restore).start()

    def enable_input_buttons(self):
        self.send_btn.setDisabled(False)
        self.input_line.setDisabled(False)
        self.toggle_btn.setDisabled(False)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = EmotionChatApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
