import voice_choice

# Change voice BEFORE speaking
voice_choice.voice_keyword = "David"   # Or "Zira", "Google", etc.

# Optional: list available voices
voice_choice.list_voices()
# Speak some text
voice_choice.speak("Hey there! This is a dynamic voice selection demo.")
