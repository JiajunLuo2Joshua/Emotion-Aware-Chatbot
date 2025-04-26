from face_detection.detect import detect_face_from_camera
import cv2

face = detect_face_from_camera(show_window=True)
if face is not None:
    cv2.imwrite("detected_face.jpg", face)  # save the detected face images
