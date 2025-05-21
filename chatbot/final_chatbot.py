import sounddevice as sd
import numpy as np
import queue
import time
import whisper
import openai
import warnings

import soundfile as sf
import os
import voice_choice
from chat_memory import ChatMemory

# Optional: list available voices
# voice_choice.list_voices()
voice_choice.voice_keyword = "Zira"

from vosk import  KaldiRecognizer
from vosk import Model as voskModel

import json
trigger = "hi blueberry"

vosk_model = voskModel("vosk-model-small-en-us-0.15")  
rec = KaldiRecognizer(vosk_model, 16000)

voice_actived = False

def listen_and_trigger():
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1) as stream:
        print("🎧 Listening for:  "+ trigger)
        while True:
            raw_data = stream.read(4000)[0]
            data = np.frombuffer(raw_data, dtype=np.int16)
            audio_bytes = data.tobytes()   
            if rec.AcceptWaveform(audio_bytes):
                result = json.loads(rec.Result())
                text = result.get("text", "").lower()
                print(f"📝 Heard: {text}")
                if trigger in text:
                    voice_actived = True
                    print("✅ Wake word detected!")
                    break


# === OpenAI key ===
def load_api_key(file_path="openai_key.txt"):
    with open(file_path, "r") as f:
        return f.read().strip()

openai.api_key = load_api_key()
client = openai.OpenAI(api_key=openai.api_key)

# === Whisper ===
model = whisper.load_model("base")

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


# === Audio Params ===
fs = 16000
block_duration = 0.1  # seconds
block_size = int(fs * block_duration)
silence_db_threshold = -50 #scilence is around -80 dB
max_silence_time = 2.0  # seconds
silence_blocks_limit = int(max_silence_time / block_duration)

# === Queues and Buffers ===
audio_queue = queue.Queue()
recorded_frames = []
recording = False
silence_counter = 0

# === Helper Functions ===
def rms_db(samples, ref=1.0):
    rms = np.sqrt(np.mean(samples**2))
    if rms < 1e-10:
        return -float('inf')
    return 20 * np.log10(rms / ref)

def save_conversation_log(history, filename="chat_log.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for m in history:
            if m["role"] == "user":
                f.write(f"🧑 User: {m['content']}\n")
            elif m["role"] == "assistant":
                f.write(f"🤖 Assistant: {m['content']}\n")
            elif m["role"] == "system":
                f.write(f"[System Prompt]\n{m['content']}\n\n")

system_prompt = (
    "You are a warm, patient, and emotionally supportive assistant. "
    "Your job is to comfort and accompany elderly users who may be feeling anxious, lonely, or upset. "
    "Speak gently, offer encouragement, and use simple and kind words. "
    "Avoid giving medical advice. If the user sounds very distressed, encourage them to talk to a family member or doctor."
)
memory = ChatMemory(system_prompt)

def chatgpt_query(prompt):
    memory.add_user_input(prompt)
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=memory.get_messages()
        )
        reply = response.choices[0].message.content.strip()
        memory.add_assistant_response(reply)
        return reply
    except Exception as e:
        print("⚠️ ChatGPT error:", e)
        return None
    
def text_input():
    user_input = input("⌨️ Please type your message: ").strip()
    if user_input:
        print(f"📝 You typed: {user_input}")
        reply = chatgpt_query(user_input)
        if reply:
            voice_choice.speak(reply)
    else:
        print("⚠️ No input detected.")

should_exit = False

# === Record Once Function ===
def record_one_utterance():
    
    audio_queue.queue.clear()
    recording = False
    recorded_frames = []
    silence_counter = 0

    def callback(indata, frames, time_info, status):
        nonlocal recording, recorded_frames, silence_counter
        samples = indata[:, 0]
        db = rms_db(samples)

        #print(f"🎚️ Volume: {db:.2f} dB")

        if db > silence_db_threshold:
            if not recording:
                print("🎙️ Detected speech start!")
                recording = True
                recorded_frames = []
                silence_counter = 0
            recorded_frames.append(samples.copy())
            silence_counter = 0
        elif recording:
            recorded_frames.append(samples.copy())
            silence_counter += 1
            if silence_counter >= silence_blocks_limit:
                print("🔴 Speech ended.")
                audio_queue.put(np.concatenate(recorded_frames))
                raise sd.CallbackStop()  # ✅ Stop stream cleanly

    print("🎤 Speak now...")
    with sd.InputStream(samplerate=fs, channels=1, blocksize=block_size, callback=callback):
        try:
            while audio_queue.empty():
                time.sleep(0.1)
        except sd.CallbackStop:
            pass  # Graceful stop

# === Main Interaction Loop ===
segment = 1
try:
    while True:
        # ✅ Step 1: Ask for user input
        text_input()
        listen_and_trigger()
        if voice_actived:
            record_one_utterance()
        else:
            text_input()
            continue

        # ✅ Step 2: Process recorded audio
        audio_data = audio_queue.get()
        filename = f"temp_{segment}.wav"
        sf.write(filename, audio_data, fs)

        print("🧠 Transcribing...")
        result = model.transcribe(filename)
        transcript = result["text"].strip()

        if transcript:
            print(f"📝 You said: {transcript}")
            reply = chatgpt_query(transcript)
            if reply:
                voice_choice.speak(reply)
        else:
            print("😶 No speech detected.")

        segment += 1

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
    save_conversation_log(memory.get_messages())
