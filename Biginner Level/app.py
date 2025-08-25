import os, zipfile

# Unzip the model if .h5 file is not already extracted
if not os.path.exists("cifar10_model.h5") and os.path.exists("cifar10_model.zip"):
    with zipfile.ZipFile("cifar10_model.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
        
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
import tensorflow as tf
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# CIFAR-10 Classes
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load or train model
@st.cache_resource
def load_model():
    if os.path.exists("cifar10_model.h5"):
        model = tf.keras.models.load_model("cifar10_model.h5")
    else:
        st.warning("⚠️ Model not found! Training a new model (this may take 1-2 mins)...")

        # Load CIFAR-10 dataset
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train/255.0, x_test/255.0
        y_train, y_test = y_train.flatten(), y_test.flatten()

        from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
        from tensorflow.keras.models import Model

        K = len(set(y_train))
        i = Input(shape=x_train[0].shape)
        x = Conv2D(32, (3,3), activation='relu', padding='same')(i)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(K, activation='softmax')(x)

        model = Model(i, x)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Train with fewer epochs for quick demo
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=64)
        model.save("cifar10_model.h5")

    return model

model = tf.keras.models.load_model("cifar10_model.h5")


# Streamlit UI
st.title("🚀 CIFAR-10 Image Classifier")
st.write("Upload an image and let the model predict its class!")

uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader(f"✅ Prediction: {CLASS_NAMES[class_index]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    # Show top-3 predictions
    top_indices = prediction[0].argsort()[-3:][::-1]
    st.write("### 🔎 Top 3 Predictions")
    for i in top_indices:
        st.write(f"- {CLASS_NAMES[i]} ({prediction[0][i]*100:.2f}%)")


