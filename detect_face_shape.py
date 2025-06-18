import face_recognition
import numpy as np

def detect_face_shape(image_path):
    image = face_recognition.load_image_file(image_path)
    landmarks = face_recognition.face_landmarks(image)

    if not landmarks:
        return "Unknown"

    points = landmarks[0]['chin']
    jaw_width = np.linalg.norm(np.array(points[0]) - np.array(points[-1]))
    chin_point = points[len(points)//2]
    jaw_height = np.linalg.norm(np.array(points[0]) - np.array(chin_point))

    ratio = jaw_width / jaw_height

    if ratio > 2.0:
        return "round"
    elif 1.6 < ratio <= 2.0:
        return "oval"
    elif ratio <= 1.6:
        return "square"
    else:
        return "unknown"

