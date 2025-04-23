import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import os

# 💡 Add local ffmpeg to system PATH for subprocess calls
ffmpeg_path = os.path.abspath("./ffmpeg/bin")
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

# Load Whisper base model
print("📦 Loading local Whisper base model...")
model = whisper.load_model("base")  # Load from default cache path
print("✅ Model loaded!")

# Recording settings
fs = 16000              # Sample rate (Hz)
duration = 5            # Duration of each recording (seconds)
counter = 1             # Segment counter

print("🎤 Whisper Real-Time Listening Started (press Ctrl+C to stop)")
print("-----------------------------------------------------------")

try:
    while True:
        print(f"\n⏺️  Recording... (segment {counter})")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        # Save as 16-bit PCM format (recommended)
        filename = f"temp_{counter}.wav"
        audio = recording.astype(np.float32) / np.max(np.abs(recording))
        sf.write(filename, audio, fs, subtype='PCM_16')

        # Playback the recorded audio
        print("🔊 Playing back your recording...")
        data, _ = sf.read(filename)
        sd.play(data, fs)
        sd.wait()

        # Transcribe using Whisper
        print("🧠 Transcribing...")
        result = model.transcribe(filename)

        if result["text"].strip():
            print("📝 You said:", result["text"])
        else:
            print("😶 No speech detected.")

        counter += 1
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user. Exiting...")
