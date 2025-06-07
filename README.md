# ECEC: Design and Implementation of an Emotional Chatbot for Elderly Companionship

![Project Logo](./emotion_model/TeamLogo.jpg)

## Project Overview

**Emotionally Chatbot for Elderly Companionship** is a desktop emotional companion robot designed for elderly users. The project takes real-time user expression recognition as its core, combines voice/text input, integrates into ChatGPT's natural conversations and various user-friendly interfaces, aiming to provide the elderly with a warm, intelligent and easy-to-use digital companionship experience and reduce their sense of loneliness.

---

## ✨ Key Features

- **Real-Time Emotion Recognition:** Webcam-based detection and recognize of Ekman emotions.
- **Smart Chat:** Integrated OpenAI ChatGPT API for natural, empathetic conversation.
- **Multimodal Input:** Switch between text and voice input.
- **User-Friendly GUI:** Includes video, emotion prediction, chat area, and easy voice/text controls.
- **Friendly Branding:** Heartwarming logo and colors, emphasizing companionship and elderly-friendliness.
- **Accessibility:** The large font and buttons are easy for elderly users to use.

---

## 🚀 Quick Start: Main Application
Launch the desktop application with:  

Go to emotion_model/ and run:
```
python EmotionChatUI.py
```
Opens a GUI, starts the webcam, detects And recognizes faces, and predicts emotions in real time.

Chat via text or switch to voice input anytime.

**Description of main functional areas:**

- **On the left:** Real-time camera footage and emotion tags

- **Upper right:** Current sentiment and historical sentiment trends

- **Bottom right:** Care tips

- **Middle to lower:** Chat record window

- **Below:** Text input box and voice switch button

---

## 📝 Feature List

- **Real-Time Webcam Emotion Detection:**  
  Face detection and prediction of Ekman emotions (Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise).
- **ChatGPT-Powered Dialogue:**  
  Open-domain, context-aware conversation using the latest OpenAI GPT model.
- **Text & Voice Modes:**  
  Effortless switching between keyboard and microphone input.
- **Speech Synthesis Input (Whisper):**  
  Chatbot responses are spoken aloud using system default English voice.
- **Speech Synthesis Output (Pyttsx3):**  
  Converts text to speech quickly, suitable for real-time conversations.
- **Emotion History Tracking:**  
  GUI displays current and recent emotions as emoji trends.
- **Elderly-Friendly UI:**  
  Large fonts, clear icons, and supportive prompt area.
- **Flexible Model Benchmarking:**  
  Easily benchmark EfficientNet-B0, MobileNetV2-100, and ResNet50.
- **Automatic Data Preparation:**  
  Scripts to preprocess AffectNet-HQ and RAF-DB datasets.
- **OpenAI API Integration:**  
  Secure API key handling and persistent chat memory.

---

## 📂 Directory Structure

```
CS731-2025-PROJECT-GROUP-8/
├── chatbot/                         # Voice and dialogue logic
│   ├── ffmpeg/                      # Audio/video support binaries and presets
│   ├── TTS-dev/                     # Text-to-Speech (TTS) engine and scripts
│   ├── vosk-model-small-en-us-0.15/ # Offline Speech-to-Text (STT) model files
│   ├── chat_memory.py
│   ├── final_chatbot.py             # Main chatbot script (for pipeline integration)
│   ├── voice_choice.py
│   ├── voice_test.py
│   └── ... (other modules, audio logs, etc.)
├── data/
│   ├── original/                    # Original dataset
│   │   ├── AffectNet-HQ/
│   │   └── RAF-DB/
│   └── emotion_dataset/             # Dataset for training and validation 
│       ├── train/
│       └── val/
├── emotion_model/                   # Model training, inference, UI code, and checkpoints
│   ├── checkpoints/                 # Result of trained model
│   ├── EmotionChatUI.py             # Main application entry (GUI)
│   ├── train.py                     # Training code for emotion models
│   ├── evaluate_model.py            # Validation code
│   ├── inference.py                 # Inference for single/batch images
│   ├── process_dataset.py           # Dataset preprocessing code
│   └── ... (other modules)
├── face_detection/                  # Face detection code and YOLOv8 model
│   └── yolov8n-face.pt
├── requirements.txt                 # Dependency list
└── README.md
```

---

## ⚙️ Environment Setup

Requires **Python 3.10+**


# 1. Create and activate a virtual environment (conda or venv recommended)
```
conda create -n emotion_env python=3.10
conda activate emotion_env
```
# OR
```
python3 -m venv emotion_env
source emotion_env/bin/activate
```

# 2. Install dependencies
```
pip install -r requirements.txt
```
Major dependencies: PyTorch, torchvision, timm, opencv-python, ultralytics, scikit-learn, matplotlib, seaborn, openai, pyttsx3, etc.

---
## 📁 Data Preparation
Download and extract datasets
Download AffectNet-HQ and RAF-DB, then place them as follows:
```
#data/original/AffectNet-HQ/
#data/original/RAF-DB/
```
Each emotion should be a subfolder containing category images.

# Preprocess datasets
From the project root, run:
```
python process_dataset.py
```
After the operation is completed, the training/validation set in the standard format will be automatically generated::
```
data/emotion_dataset/train/
data/emotion_dataset/val/
```

---
## 🏋️ Model Training
Go to emotion_model/ and run:  
```
python train.py
```
Trains EfficientNet-B0, MobileNetV2-100, and ResNet50 models on your data.

Best weights are saved to:
```
emotion_model/checkpoints/
    best_model_efficientnet_b0.pt
    best_model_mobilenetv2_100.pt
    best_model_resnet50.pt
```

---

## 📊 Model Evaluation & Inference
1. Evaluation, Go to emotion_model/ and run:  
```
python evaluate_model.py
```
Outputs a classification report, accuracy, and confusion matrix.

2. Inference (Single/Bulk Images)
See inference.py for examples of using any trained model for facial emotion recognition on images or webcam input.

---

## ❓ FAQ
```
Q: What should I do if there is an error in the dataset path during runtime?
A: Please ensure that all paths are relative paths and that all scripts run from the project root directory or the corresponding module directory.

Q: GPU/CPU switching?
A: Scripts automatically use CUDA GPU if available.

Q: How do I set my ChatGPT API Key?
A: Place your OpenAI API key in emotion_model/openai_key.txt. For specific instructions, please refer to the code.

Q: Switching inference models?
A: In inference.py, set the model name and checkpoint (EfficientNet-B0, MobileNetV2-100, or ResNet50).
```
---

## 📚 Citations & Credits
AffectNet: https://paperswithcode.com/dataset/affectnet

RAF-DB: https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset

OpenAI ChatGPT API

YOLOv8 by Ultralytics

All open source dependencies

