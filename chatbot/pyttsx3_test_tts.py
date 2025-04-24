import pyttsx3

# Initialize TTS engine
tts = pyttsx3.init()

# List available voices
voices = tts.getProperty('voices')
print("Available voices:\n")
for i, voice in enumerate(voices):
    print(f"{i}: {voice.name} ({voice.id})")

# === ADJUST HERE ===
voice_index = 1      # Change this to try different voices (e.g. 0, 1, 2...)
rate = 140           # Speech rate (default ~200)
volume = 0.75         # Volume (0.0 to 1.0)
test_phrase = "Hello, I'm your companion robot. How are you feeling today?"

# Apply voice settings
tts.setProperty('voice', voices[voice_index].id)
tts.setProperty('rate', rate)
tts.setProperty('volume', volume)

print("\n🗣️  Speaking with the selected settings...")
print(f"Voice: {voices[voice_index].name}")
print(f"Rate: {rate}")
print(f"Volume: {volume}")
print(f"Text: \"{test_phrase}\"\n")

# Speak
tts.say(test_phrase)
tts.runAndWait()
