import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


current_dir = os.path.dirname(os.path.abspath(__file__))

eye_model_path = os.path.join(current_dir, "../model/eye_model.h5")
drowsy_model_path = os.path.join(current_dir, "../model/drowsy_model.h5")

eye_classes = ['Bulging Eyes', 'Cataracts', 'Crossed Eyes', 'Glaucoma', 'Uveitis']

st.set_page_config(page_title="CareLens AI", page_icon="👁️")
st.title("CareLens AI 👁️")
st.subheader("Eye Disease + Drowsiness Detection")

uploaded = st.file_uploader("Upload an image of the eye or face", type=["jpg", "png", "jpeg"])

if uploaded:

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    st.write("---")
    st.write("### 🔍 Analysis Results")

    eye_img = img.resize((128, 128))

    eye_array = np.array(eye_img)
    eye_input = np.expand_dims(eye_array, axis=0) 
    

    eye_pred = eye_model_path.predict(eye_input)
    eye_class_index = np.argmax(eye_pred)
    confidence = np.max(eye_pred) * 100

    st.write(f"**Eye Condition:** {eye_classes[eye_class_index]}")
    st.write(f"**Confidence:** {confidence:.2f}%")


    d_img = img.resize((64, 64))
    d_array = np.array(d_img)
    d_input = np.expand_dims(d_array, axis=0)


    d_pred = drowsy_model_path.predict(d_input)
    d_class_index = np.argmax(d_pred)
    

    status = "Unknown"
    if d_class_index == 0:
        status = "😴 Eyes Closed (Drowsy)"
    else:
        status = "👀 Eyes Open (Alert)"

    st.write(f"**Drowsiness Status:** {status}")
