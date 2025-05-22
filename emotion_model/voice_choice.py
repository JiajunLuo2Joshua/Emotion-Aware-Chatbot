import pyttsx3

# Global TTS engine and config
tts = pyttsx3.init()
voice_keyword = "United States"   # Default voice keyword (can be changed externally)
rate = 160
volume = 0.75

def list_voices():
    """Print all available voices."""
    voices = tts.getProperty('voices')
    print("Available voices:\n")
    for i, voice in enumerate(voices):
        print(f"{i}: {voice.name} ({voice.id})")

def apply_voice_settings():
    """Apply the selected voice, rate, and volume."""
    voices = tts.getProperty('voices')
    selected = None
    for voice in voices:
        if voice_keyword.lower() in voice.name.lower():
            selected = voice
            break

    if selected:
        tts.setProperty('voice', selected.id)
        print(f"✅ Selected voice: {selected.name}")
    else:
        tts.setProperty('voice', voices[0].id)
        print(f"⚠️ No match for '{voice_keyword}', using default: {voices[0].name}")

    tts.setProperty('rate', rate)
    tts.setProperty('volume', volume)

def speak(text):
    """Speak the given text using current voice settings."""
    apply_voice_settings()
    print(f"🗣️ Speaking: \"{text}\"\n")
    tts.say(text)
    tts.runAndWait()
