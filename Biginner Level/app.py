import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# CIFAR-10 Classes
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load model (make sure you have saved your trained model as 'cifar10_model.h5')
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cifar10_model.h5")
    return model

model = load_model()

# Streamlit UI
st.title("🚀 CIFAR-10 Image Classifier")
st.write("Upload an image and let the model predict its class!")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and preprocess image
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader(f"✅ Prediction: {CLASS_NAMES[class_index]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
