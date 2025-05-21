import sounddevice as sd
import numpy as np
import queue
import time
import whisper
import openai
import warnings
import threading


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
wake_trigger = "hi nicole"
exit_trigger = "no"

vosk_model = voskModel("vosk-model-small-en-us-0.15")  
rec = KaldiRecognizer(vosk_model, 16000)




def background_trigger_listener():
    """Background thread that continuously listens for the trigger or exit phrase using VOSK."""
    global voice_actived, should_exit
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1) as stream:
        while not should_exit and not voice_actived:
            raw_data = stream.read(4000)[0]
            data = np.frombuffer(raw_data, dtype=np.int16)
            audio_bytes = data.tobytes()
            if rec.AcceptWaveform(audio_bytes):
                result = json.loads(rec.Result())
                text = result.get("text", "").lower()
                #print(f"📝 Heard: {text}")

                # Trigger voice mode
                if wake_trigger in text:
                    voice_actived = True
                    print("✅ Wake word detected!")

                # Graceful program exit
                elif exit_trigger in text:
                    should_exit = True
                    print("🛑 Exit command received.")
                    break


def text_input():
    global voice_actived
    print("⌨️ You can type a message below, or say 'hi nicole' to switch to voice:")

    while not voice_actived:
        try:
            user_input = input("> ").strip()
            if user_input:
                reply = chatgpt_query(user_input)
                if reply:
                    voice_choice.speak(reply)
        except KeyboardInterrupt:
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
    



# === Record Once Function ===
def record_one_utterance():
    audio_queue.queue.clear()
    recording = False
    recorded_frames = []
    silence_counter = 0

    def callback(indata, frames, time_info, status):
        nonlocal recording, recorded_frames, silence_counter
        global voice_actived,should_exit
        

        # Extract audio samples from the input stream
        samples = indata[:, 0]
        db = rms_db(samples)

        # Convert audio to bytes for VOSK real-time speech recognition
        audio_bytes = samples.astype(np.int16).tobytes()
        if rec.AcceptWaveform(audio_bytes):
            result = json.loads(rec.Result())
            text = result.get("text", "").lower()
            print(f"🗣️ Real-time heard: {text}")

            # Check for exit command during voice interaction
            if exit_trigger in text:
                print("🔁 Exit trigger detected. Returning to text input.")
                should_exit = True
                voice_actived = False
                raise sd.CallbackStop()  # Immediately stop recording

        # If sound is above threshold, start or continue recording
        if db > silence_db_threshold:
            if not recording:
                print("🎙️ Detected speech start!")
                recording = True
                recorded_frames = []
                silence_counter = 0
            recorded_frames.append(samples.copy())
            silence_counter = 0

        # If user stops speaking, count silence frames and stop after threshold
        elif recording:
            recorded_frames.append(samples.copy())
            silence_counter += 1
            if silence_counter >= silence_blocks_limit:
                print("🔴 Speech ended.")
                audio_queue.put(np.concatenate(recorded_frames))
                raise sd.CallbackStop()  # Gracefully stop after silence

    print("🎤 Speak now...")
    with sd.InputStream(samplerate=fs, channels=1, blocksize=block_size, callback=callback):
        try:
            while audio_queue.empty() and voice_actived:
                time.sleep(0.1)
        except sd.CallbackStop:
            pass  # Stream stopped by speech end or exit trigger


# === Main Interaction Loop ===
segment = 1
try:
    while True:
        # 🔁 Reset voice assistant state
        voice_actived = False
        should_exit = False

        # 🔊 Start background thread to listen for wake word
        trigger_thread = threading.Thread(target=background_trigger_listener)
        trigger_thread.start()

        # 🖥️ Allow multiple rounds of text input until voice is triggered
        while not voice_actived:
            text_input()

        # Wait for background listener to finish
        trigger_thread.join()

        

        # 🎙️ Voice mode interaction
        while voice_actived:
            record_one_utterance()

            if should_exit:
                print("🔁 Switching back to text input mode.\n")
                voice_actived = False  # ✅ leave voice mode
                should_exit = False    # ✅ avoid breaking outer loop
                continue

            # 🧠 Transcribe and process voice input
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
