import cv2
import torch
import numpy as np
from ultralytics import YOLO

# loading model
try:
    face_model = YOLO('face_detection/yolov8n-face.pt')  # load the face detection model
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Model loading failed: {e}")

def detect_face_from_camera(show_window=False):
    cap = cv2.VideoCapture(0)  # turn on the camera

    if not cap.isOpened():
        print(" Error: Camera not accessible.")
        return None

    face_img = None

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print(" Error: Failed to grab frame.")
            break

        # Run YOLOv8 inference on the frame for face detection
        results = face_model(frame)

        # Extract bounding boxes, classify emotions, and plot them
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])

                # Extract face region
                face_img = frame[y1:y2, x1:x2]
                print(f" Face detected at ({x1},{y1}) to ({x2},{y2})")

                # Draw bounding box
                if show_window:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if show_window:
            cv2.imshow("Face Detection - Press 'q' to exit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # if face_img is not None:
        #     print(" Face detected!")
        #     break

    # Release the webcam and close windows
    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    return face_img

if __name__ == "__main__":
    detect_face_from_camera(show_window=True)
