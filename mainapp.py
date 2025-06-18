import streamlit as st
import mediapipe as mp
import cv2
import json
import numpy as np
from PIL import Image

# Load hairstyle suggestions
with open("hairstyles.json") as f:
    hairstyle_data = json.load(f)

# Initialize face mesh
mp_face_mesh = mp.solutions.face_mesh

def detect_face_shape(landmarks):
    face_width = landmarks[234][0] - landmarks[454][0]
    forehead_to_chin = landmarks[10][1] - landmarks[152][1]
    ratio = abs(face_width / forehead_to_chin)

    if ratio > 1.4:
        return "round"
    elif 1.2 <= ratio <= 1.4:
        return "square"
    elif ratio < 1.2:
        return "oval"
    else:
        return "unknown"

def get_landmarks(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            h, w, _ = image.shape
            landmarks = []
            for lm in results.multi_face_landmarks[0].landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append((x, y))
            return landmarks
    return None

# Streamlit UI
st.set_page_config(page_title="Hairstyle Recommender")
st.title("📷 Hairstyle Recommender")

gender = st.radio("Select your gender:", ["Male", "Female"], horizontal=True)

option = st.radio("Choose input method:", ["Upload a photo", "Use webcam"], horizontal=True)

image = None

if option == "Upload a photo":
    uploaded = st.file_uploader("Upload your face image", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

elif option == "Use webcam":
    picture = st.camera_input("Take a photo")
    if picture:
        image = cv2.imdecode(np.frombuffer(picture.read(), np.uint8), cv2.IMREAD_COLOR)

if image is not None:
    landmarks = get_landmarks(image)
    if landmarks:
        face_shape = detect_face_shape(landmarks)
        st.success(f"Detected face shape: **{face_shape.capitalize()}**")

        st.markdown("### 💇 Recommended Hairstyles:")
        styles = hairstyle_data.get(face_shape, {}).get(gender.lower(), [])
        if styles:
            for style in styles:
                st.write(f"• {style}")
        else:
            st.warning("No styles found for your face shape and gender.")
    else:
        st.error("Face not detected. Please upload a clear photo.")


