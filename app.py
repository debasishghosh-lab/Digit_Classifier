import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

from PIL import Image

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load Model
model = tf.keras.models.load_model(
    "digit_classifier.keras"
)

# Load Class Names
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# Page Title
st.title("🔢 Digit Classifier")

st.write(
    "Upload a digit image and get prediction."
)

# Upload Image
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess
    img_resized = img.resize((224,224))

    img_array = image.img_to_array(
        img_resized
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    # Prediction
    prediction = model.predict(
        img_array
    )

    predicted_class = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    )

    st.success(
        f"Predicted Digit: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence*100:.2f}%"
    )