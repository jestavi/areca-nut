
# app.py
import streamlit as st
from PIL import Image
from image_classification import classify_image

# Streamlit app
st.title("Areca Nut Image Classification")

uploaded_file = st.file_uploader("Choose an Areca nut image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")
    class_name, confidence_score = classify_image(uploaded_file)
    st.write(f"Class: {class_name}")
    st.write(f"Confidence Score: {confidence_score:.2f}")

# Additional UI elements, if needed
st.write("Upload an Areca nut image to classify it using the Keras model.")

