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
import threading


# Load Models and Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints', 'best_model_efficientnet_b0.pt')
FACE_MODEL_PATH = os.path.join(BASE_DIR, '..', 'face_detection', 'yolov8n-face.pt')
LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_model = create_model('efficientnet_b0', pretrained=False, num_classes=len(LABELS))
emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=DEVICE))
emotion_model.to(DEVICE)
emotion_model.eval()

face_model = YOLO(FACE_MODEL_PATH)

transform = torch.nn.Sequential(
    torch.nn.Identity(),  # placeholder for torchvision transforms in a sequential wrapper
)

api_key_path = os.path.join(BASE_DIR, "openai_key.txt")
def load_api_key(file_path=api_key_path):
    with open(file_path, "r") as f:
        return f.read().strip()
openai.api_key = load_api_key()
client = openai.OpenAI(api_key=openai.api_key)


system_prompt = (
    "You are a warm, emotionally supportive, and conversational assistant. "
    "Your goal is to gently comfort and accompany elderly users who may be feeling anxious, lonely, or upset. "
    "Speak in a natural, human-like way—avoid listing items or giving advice in numbered or bullet-point format. "
    "Instead, focus on having a warm, flowing conversation, showing empathy and gentle curiosity. "
    "Offer encouragement with kind words and emotional presence. "
    "If the user seems very distressed, gently suggest they talk to someone they trust or a doctor. "
    "Use clear, simple language, and prioritize emotional connection over structured advice."
    "Only provid the main advice, no more than 5 sentences."
)
memory = ChatMemory(system_prompt)





class EmotionChatApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion-Aware Companion Chatbot")
        self.resize(1280, 960)
        self.init_ui()

        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Emotion Voting
        self.vote_buffer = []
        self.vote_start_time = time.time()
        self.vote_window_seconds = 2
        self.vote_sample_target = 100
        self.vote_threshold = 0.8
        self.final_emotion = "Unknown"
        self.pre_emotion = None
        self.voice_handler = VoiceInputHandler()

        self.voice_mode_active = False
        self.voice_loop_timer = QtCore.QTimer()
        self.voice_loop_timer.setSingleShot(True)
        self.voice_loop_timer.timeout.connect(self.voice_loop_step)

        self.recent_emotions = []

    def init_ui(self):        
        
        layout = QtWidgets.QVBoxLayout(self)
        top_layout = QtWidgets.QHBoxLayout()
        
        self.video_label = QtWidgets.QLabel()  
        self.video_label.setFixedSize(960, 540)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed
        )
        top_layout.addWidget(self.video_label)

        ## top right
        top_right_layout = QtWidgets.QVBoxLayout()

        # 1. Emotion Label
        self.emotion_label = QtWidgets.QLabel("Emotion: Unknown")
        self.emotion_label.setStyleSheet("""
            background-color: #e6ffe6;
            font-size: 36px;
            font-weight: bold;
            border: 2px solid #b2fab4;
            border-radius: 12px;
            padding: 12px;
        """)
        self.emotion_label.setAlignment(QtCore.Qt.AlignCenter)
        top_right_layout.addWidget(self.emotion_label, stretch=1)

        # 2. Emotion History Panel
        self.emotion_history_label = QtWidgets.QLabel("Recent Emotions:\n(none yet)")
        self.emotion_history_label.setStyleSheet("""
            background-color: #f0f8ff;
            border: 2px solid #a0c4ff;
            border-radius: 12px;
            padding: 10px;
            font-size: 32px;
            font-weight: bold;                                     
                                                 
        """)
        self.emotion_history_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        top_right_layout.addWidget(self.emotion_history_label, stretch=1)

        # 3. Suggestion Box
        self.suggestion_box = QtWidgets.QLabel("💡 Tip: Try talking about a happy memory!")
        self.suggestion_box.setStyleSheet("""
            background-color: #fff8dc;
            border: 1px solid #ffe4b5;
            border-radius: 10px;
            padding: 10px;
            font-size: 32px;
            font-weight: bold;
        """)
        self.suggestion_box.setWordWrap(True)
        top_right_layout.addWidget(self.suggestion_box, stretch=1)

        layout.addStretch()

        top_layout.addLayout(top_right_layout)
        layout.addLayout(top_layout, stretch=1)
        
        ## Middle
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.chat_container = QtWidgets.QWidget()
        self.chat_layout = QtWidgets.QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(QtCore.Qt.AlignTop)
        
        self.chat_container.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding
        )
        self.scroll_area.setWidget(self.chat_container)
        layout.addWidget(self.scroll_area, stretch=1)


        input_layout = QtWidgets.QHBoxLayout()
        self.input_line = QtWidgets.QTextEdit()
        #self.input_line.setFixedHeight(100)  # Can be adjusted or made resizable later
        self.input_line.setStyleSheet("font-size: 32px; padding: 8px;")
        self.input_line.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding  # 🔥 Allow vertical growth
        )

        # Create the "Send" button
        self.send_btn = QtWidgets.QPushButton("SEND")
        self.send_btn.setStyleSheet("font-size: 24px;")
        self.send_btn.setFixedSize(240,48)
        self.send_btn.clicked.connect(self.handle_text_input)

        # Create the "Switch to Voice" button
        self.mode = "text"
        self.toggle_btn = QtWidgets.QPushButton("🎤 Switch to Voice")
        self.toggle_btn.setStyleSheet("font-size: 24px;")
        self.toggle_btn.setFixedSize(240,48)
        self.toggle_btn.clicked.connect(self.toggle_input_mode)

        # Layout for buttons (Send above, Voice below, evenly spaced vertically)
        button_layout = QtWidgets.QVBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.send_btn, alignment=QtCore.Qt.AlignCenter)
        button_layout.addStretch()
        button_layout.addWidget(self.toggle_btn, alignment=QtCore.Qt.AlignCenter)
        button_layout.addStretch()

        # Add text input and button stack to main horizontal layout
        input_layout.addWidget(self.input_line, stretch=1)
        input_layout.addLayout(button_layout)

        # Wrap input layout in a QWidget and set padding/margin
        input_container = QtWidgets.QWidget()
        input_container.setLayout(input_layout)
        input_container.setContentsMargins(10, 10, 10, 20)
        layout.addWidget(input_container,stretch=1)

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
    
    def add_chat_bubble(self, role, message):
        bubble = ChatBubble(role, message)
        self.chat_layout.addWidget(bubble)
        QtCore.QTimer.singleShot(100, lambda:
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
        )


    def chatgpt_query(self, prompt):
        memory.add_user_input(prompt)
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=memory.get_messages()
            )
            reply = response.choices[0].message.content.strip()
            #reply = response.choices[0].message.content.strip()
            memory.add_assistant_response(reply)
            return reply
        except Exception as e:
            print("⚠️ ChatGPT error:", e)
            return "(ChatGPT failed to respond.)"

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
        history_text = f"Recent Emotions:\n {emoji_sequence}"
        
        self.emotion_history_label.setText(history_text)

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
        tip = suggestions.get(emotion, "💡 I'm here for you. Feel free to share anything on your mind.")
        self.suggestion_box.setText(tip)

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
                #chat_fn=self.chatgpt_query,
                final_emotion=self.final_emotion,
                add_bubble_fn=self.add_chat_bubble
            )
            #Stop the user's any possible change
            self.send_btn.setDisabled(True)
            self.input_line.setDisabled(True)
            self.toggle_btn.setDisabled(True)
            QtCore.QTimer.singleShot(100, lambda: self.speak_and_resume(voice_reply))

        except Exception as e:
            print("Voice input error:", e)
            voice_choice.speak("Something went wrong during voice input.")
        
        #if self.voice_mode_active:
            #self.voice_loop_timer.start(300) 
    
    def speak_and_resume(self, reply):
        try:
            self.speak_and_reenable(reply) 
        except Exception as e:
            print("Voice speak error:", e)

        if self.voice_mode_active:
            self.voice_loop_timer.start(300)

            
    
    def handle_text_input(self):
        user_text = self.input_line.toPlainText().strip()
        if user_text:
            self.add_chat_bubble("user", user_text)
            self.input_line.clear()

            #Stop the user's any possible change
            self.send_btn.setDisabled(True)
            self.input_line.setDisabled(True)
            self.toggle_btn.setDisabled(True)
            

            emotion_context = f"(The user seems to be feeling: {self.final_emotion}.)\n"
            full_prompt = emotion_context + user_text
            reply = self.chatgpt_query(full_prompt)
            self.add_chat_bubble("bot", reply)
            voice_choice.voice_keyword= "United States" 
            QtCore.QTimer.singleShot(100, lambda: self.speak_and_reenable(reply))
            


    def speak_and_reenable(self, reply):
        

        def speak_and_restore():
            try:
                voice_choice.speak(reply)
            except Exception as e:
                print("Voice speak error:", e)
            finally:
                # Re-enable buttons on the main thread
                QtCore.QTimer.singleShot(0, self.enable_input_buttons)

        threading.Thread(target=speak_and_restore).start()

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

    def enable_input_buttons(self):
        self.send_btn.setDisabled(False)
        self.input_line.setDisabled(False)
        self.toggle_btn.setDisabled(False)
        


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = EmotionChatApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
