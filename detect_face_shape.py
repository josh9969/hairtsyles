import dlib
import numpy as np
import cv2

# Load pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_face_shape(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    face = faces[0]
    landmarks = predictor(gray, face)

    # Facial points
    jaw = [landmarks.part(n) for n in range(0, 17)]
    jaw_width = jaw[-1].x - jaw[0].x

    cheekbone_width = landmarks.part(15).x - landmarks.part(1).x
    face_height = landmarks.part(8).y - landmarks.part(27).y
    forehead = landmarks.part(21).x - landmarks.part(22).x

    ratio = round(jaw_width / face_height, 2)

    # Simple rule-based classification
    if ratio >= 1.5:
        return "Round"
    elif jaw_width < cheekbone_width:
        return "Oval"
    elif cheekbone_width > jaw_width and cheekbone_width > forehead:
        return "Heart"
    elif jaw_width > cheekbone_width:
        return "Square"
    else:
        return "Oblong"

