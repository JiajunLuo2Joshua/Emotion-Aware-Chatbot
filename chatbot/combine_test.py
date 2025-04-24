import os
import openai
import pyttsx3
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import time

# 💡 Add local ffmpeg to system PATH for subprocess calls
ffmpeg_path = os.path.abspath("./ffmpeg/bin")
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

# ✅ Load OpenAI API key from a local text file (openai_key.txt)
def load_api_key(file_path="openai_key.txt"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read().strip()
    raise ValueError("❌ Please create an 'openai_key.txt' file with your OpenAI API key.")

openai.api_key = load_api_key()
client = openai.OpenAI(api_key=openai.api_key)

# ✅ Initialize Whisper
print("📦 Loading Whisper model...")
model = whisper.load_model("base")
print("✅ Whisper model loaded!")

# ✅ Initialize pyttsx3 TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 140)
tts_engine.setProperty("volume", 0.75)
# Optional: Try to select a gentle voice
voices = tts_engine.getProperty("voices")
if len(voices) > 1:
    tts_engine.setProperty("voice", voices[1].id)

def say(text):
    print("🗣️ ChatGPT is speaking...")
    tts_engine.say(text)
    tts_engine.runAndWait()

# ✅ ChatGPT query
def chatgpt_query(prompt, model="gpt-4o", temperature=0.5, max_tokens=150):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ Error communicating with ChatGPT API:", e)
        return None

# ✅ Main loop: record → transcribe → chat → speak
fs = 16000
duration = 5
counter = 1

print("🎤 Speak when you're ready! (Ctrl+C to stop)")
print("------------------------------------------------")

try:
    while True:
        print(f"\n⏺️  Recording... (segment {counter})")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        filename = f"temp_{counter}.wav"
        audio = recording.astype(np.float32) / np.max(np.abs(recording))
        sf.write(filename, audio, fs, subtype='PCM_16')

        print("🔊 Playing back...")
        data, _ = sf.read(filename)
        sd.play(data, fs)
        sd.wait()

        print("🧠 Transcribing...")
        result = model.transcribe(filename)
        transcript = result["text"].strip()

        if transcript:
            print(f"📝 You said: {transcript}")
            response = chatgpt_query(transcript)
            if response:
                print("🤖 ChatGPT:", response)
                say(response)
        else:
            print("😶 No speech detected.")

        counter += 1
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user. Goodbye!")
