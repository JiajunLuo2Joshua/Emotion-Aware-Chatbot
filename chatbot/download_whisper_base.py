import whisper

print("📦 Downloading Whisper base model...")
model = whisper.load_model("base")
#model = whisper.load_model("tiny")
print("✅ Model downloaded and loaded successfully!")
