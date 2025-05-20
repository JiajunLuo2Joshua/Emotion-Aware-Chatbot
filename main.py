import sys
sys.path.append("./emotion_model")  # Add the parent directory to the system path
from gui_emotion import run_emotion_capture_gui

if __name__ == "__main__":
    run_emotion_capture_gui(log_interval=5)  # Run the GUI
