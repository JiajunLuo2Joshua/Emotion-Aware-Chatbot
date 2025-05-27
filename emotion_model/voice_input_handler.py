# voice_input_handler.py
import os
import sounddevice as sd
import whisper
import numpy as np
import soundfile as sf
import voice_choice

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class VoiceInputHandler:
    def __init__(self, fs=16000, silence_db_threshold=-50, silence_seconds=2.0):
        self.fs = fs
        self.block_duration = 0.1
        self.block_size = int(self.fs * self.block_duration)
        self.silence_db_threshold = silence_db_threshold
        self.silence_blocks_limit = int(silence_seconds / self.block_duration)
        self.audio_queue = []
        self.model = whisper.load_model("base")
        self.active = False

    def record_utterance(self):
        import queue
        q = queue.Queue()
        silence_counter = 0
        recording = False
        recorded_frames = []

        def rms_db(samples, ref=1.0):
            rms = np.sqrt(np.mean(samples ** 2))
            return -float('inf') if rms < 1e-10 else 20 * np.log10(rms / ref)

        def callback(indata, frames, time_info, status):
            nonlocal recording, recorded_frames, silence_counter
            samples = indata[:, 0]
            db = rms_db(samples)

            if db > self.silence_db_threshold:
                if not recording:
                    print("🎙️ Start speaking...")
                    recording = True
                    recorded_frames = []
                    silence_counter = 0
                recorded_frames.append(samples.copy())
                silence_counter = 0
            elif recording:
                recorded_frames.append(samples.copy())
                silence_counter += 1
                if silence_counter >= self.silence_blocks_limit:
                    q.put(np.concatenate(recorded_frames))
                    raise sd.CallbackStop()

        with sd.InputStream(samplerate=self.fs, channels=1, blocksize=self.block_size, callback=callback):
            while q.empty():
                sd.sleep(100)
        return q.get()

    def transcribe_and_respond222(self, chat_fn, final_emotion, chat_history_widget):
        try:
            audio = self.record_utterance()
            temp_path = os.path.join(BASE_DIR, "temp_voice.wav")
            sf.write(temp_path, audio, self.fs)

            result = self.model.transcribe(temp_path)
            transcript = result["text"].strip()

            if transcript:
                chat_history_widget.append(f"🧑 (voice): {transcript}")
                emotion_prompt = f"(The user seems to be feeling: {final_emotion}.)\n"
                reply = chat_fn(emotion_prompt + transcript)
                chat_history_widget.append(f"🤖: {reply}")
                voice_choice.speak(reply)
            else:
                chat_history_widget.append("⚠️ No speech detected.")
                voice_choice.speak("Sorry, I didn't catch that.")
        except Exception as e:
            print("Voice input failed:", e)
            voice_choice.speak("There was a problem with voice input.")

    def transcribe_and_respond(self, chat_fn, final_emotion, add_bubble_fn):
        try:
            audio = self.record_utterance()
            temp_path = os.path.join(BASE_DIR, "temp_voice.wav")
            sf.write(temp_path, audio, self.fs)

            result = self.model.transcribe(temp_path)
            transcript = result["text"].strip()

            if transcript:
                add_bubble_fn("user", transcript)
                emotion_prompt = f"(The user seems to be feeling: {final_emotion}.)\n"
                reply = chat_fn(emotion_prompt + transcript)
                add_bubble_fn("bot", reply)
                return reply 
            else:
                add_bubble_fn("bot", "⚠️ No speech detected.")
                return "Sorry, I didn't catch that."
        except Exception as e:
            print("Voice input failed:", e)
            return "There was a problem with voice input."

    def start_conversation_loop(self, chat_fn, final_emotion_fn, chat_history_widget):
        self.active = True
        voice_choice.speak("Voice mode started. You may speak anytime.")
        
        while self.active:
            try:
                audio = self.record_utterance()
                temp_path = os.path.join(BASE_DIR, "temp_voice.wav")
                sf.write(temp_path, audio, self.fs)

                result = self.model.transcribe(temp_path)
                transcript = result["text"].strip()

                if transcript:
                    chat_history_widget.append(f"🧑 (voice): {transcript}")
                    emotion = final_emotion_fn()
                    emotion_prompt = f"(The user seems to be feeling: {emotion}.)\n"
                    reply = chat_fn(emotion_prompt + transcript)
                    chat_history_widget.append(f"🤖: {reply}")
                    voice_choice.speak(reply)
                else:
                    chat_history_widget.append("⚠️ No speech detected.")
                    voice_choice.speak("Sorry, I didn't catch that.")
            except Exception as e:
                print("Voice loop error:", e)
                voice_choice.speak("An error occurred during voice input.")

    def stop_conversation_loop(self):
        self.active = False

