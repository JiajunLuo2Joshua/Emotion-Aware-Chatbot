import pyttsx3

# Initialize the text-to-speech engine
tts = pyttsx3.init()

# Retrieve the list of available voices from the system
voices = tts.getProperty('voices')
print("Available voices:\n")
for i, voice in enumerate(voices):
    print(f"{i}: {voice.name} ({voice.id})")

# === Automatically search for a voice containing 'Ziri' in its name ===
ziri_voice = None
for voice in voices:
    if "ziri" in voice.name.lower():  # Case-insensitive match
        ziri_voice = voice
        break  # Stop after finding the first match

# === Voice settings ===
rate = 160        # Set speech rate (default ~200 words per minute)
volume = 0.75     # Set volume (range: 0.0 to 1.0)
test_phrase = "Hello, I'm your companion robot. How are you feeling today?"

# If 'Ziri' voice was found, use it; otherwise fallback to default voice
if ziri_voice:
    tts.setProperty('voice', ziri_voice.id)
    print(f"\n✅ Selected voice: {ziri_voice.name}")
else:
    print("\n⚠️ Voice 'Ziri' not found, using default voice.")
    tts.setProperty('voice', voices[0].id)  # Fallback to the first available voice

# Apply rate and volume settings
tts.setProperty('rate', rate)
tts.setProperty('volume', volume)

# Speak the test phrase
print(f"🗣️  Speaking: \"{test_phrase}\"\n")
tts.say(test_phrase)
tts.runAndWait()
