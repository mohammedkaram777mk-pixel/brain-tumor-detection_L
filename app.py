import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("brain_tumor_model.h5")

st.title("Brain Tumor Detection")

uploaded_file = st.file_uploader("Upload MRI Image")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    if pred[0][0] > 0.5:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor")