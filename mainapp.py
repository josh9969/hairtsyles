import streamlit as st
import cv2
import os
from detect_face_shape import detect_face_shape
import json
import tempfile

st.set_page_config(page_title="Hairstyle Recommender", layout="centered")
st.title("💇‍♂️ Hairstyle Recommender")

# Take photo
st.write("📷 Click below to take a photo of your face.")

camera_image = st.camera_input("Take a picture")

if camera_image:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(camera_image.getvalue())
        img_path = temp_file.name

    st.image(img_path, caption="Your Photo", use_column_width=True)

    shape = detect_face_shape(img_path)
    st.success(f"Detected face shape: **{shape}**")

    # Load styles
    with open("hairstyles.json", "r") as file:
        styles = json.load(file)

    recommendations = styles.get(shape.lower(), [])
    
    if recommendations:
        st.subheader("Recommended Hairstyles:")
        for style in recommendations:
            st.markdown(f"- {style}")
    else:
        st.warning("No hairstyles found for this face shape.")
