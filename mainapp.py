import streamlit as st
import cv2
import dlib
import json
import os
import numpy as np
from PIL import Image

from detect_face_shape import detect_face_shape

st.set_page_config(page_title="Hairstyle Recommender", layout="centered")
st.title("📸 Hairstyle Recommender")
st.markdown("Upload your photo or take one using the camera. Then select your gender to get hairstyle suggestions!")

# Load hairstyle recommendations
with open("hairstyles.json", "r") as f:
    hairstyles = json.load(f)

# Gender input
gender = st.radio("Select your gender:", ["Male", "Female"])

# Image input
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if image_file is None:
    img_data = st.camera_input("Or take a picture using your webcam")

    if img_data:
        image = Image.open(img_data)
    else:
        st.stop()
else:
    image = Image.open(image_file)

# Convert image for processing
image_np = np.array(image)
face_shape = detect_face_shape(image_np)

if face_shape:
    st.success(f"Detected Face Shape: **{face_shape}**")

    suggestions = hairstyles.get(gender.lower(), {}).get(face_shape.lower(), [])
    if suggestions:
        st.subheader("💇 Recommended Hairstyles:")
        for s in suggestions:
            st.write(f"• {s}")
    else:
        st.warning("No hairstyles found for this face shape.")
else:
    st.error("Could not detect face shape. Please try with a clearer photo.")


