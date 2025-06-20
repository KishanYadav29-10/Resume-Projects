import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = load_model("models/garbage_classifier.h5")
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.set_page_config(page_title="EcoVision | Garbage Classifier", layout="centered")
st.title("♻️ EcoVision: Intelligent Garbage Classifier")
st.markdown("Upload an image of a waste item or use your webcam to identify its type (plastic, metal, paper, etc.).")

# Function to predict
def predict(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    idx = np.argmax(pred)
    label = class_labels[idx]
    confidence = float(pred[0][idx])
    return label, confidence

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    label, confidence = predict(image)
    st.success(f"🧠 Predicted: **{label.upper()}** ({confidence*100:.2f}%)")

# Webcam
if st.checkbox("📷 Use Webcam"):
    picture = st.camera_input("Take a photo")
    if picture:
        image = Image.open(picture)
        st.image(image, caption="Captured Photo", use_column_width=True)
        label, confidence = predict(image)
        st.success(f"🧠 Predicted: **{label.upper()}** ({confidence*100:.2f}%)")
