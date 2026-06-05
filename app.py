import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Disaster Resilience Image Segmentation",
    page_icon="🌍",
    layout="wide"
)

# Title
st.title("🌍 Image Segmentation for Disaster Resilience")
st.markdown("""
Upload a satellite image and the trained U-Net model will generate
a segmentation mask highlighting detected building regions.
""")

# Load Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "disaster_segmentation_unet.h5",
        compile=False
    )
    return model

model = load_model()

# File Upload
uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Read Image
    image = Image.open(uploaded_file).convert("RGB")

    img = np.array(image)

    # Resize for Model
    img_resized = cv2.resize(img, (256, 256))

    # Normalize
    input_img = img_resized / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    # Prediction
    prediction = model.predict(input_img)

    prediction = prediction[0]

    # Binary Mask
    pred_mask = (prediction > 0.3).astype(np.uint8)

    pred_mask = pred_mask.squeeze()

    # Create Overlay
    overlay = img_resized.copy()

    overlay[pred_mask == 1] = [255, 0, 0]

    # Display Results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Input Image")
        st.image(
            img_resized,
            use_container_width=True
        )

    with col2:
        st.subheader("Predicted Mask")
        st.image(
            pred_mask * 255,
            use_container_width=True
        )

    with col3:
        st.subheader("Overlay Result")
        st.image(
            overlay,
            use_container_width=True
        )

    # Statistics
    st.markdown("---")

    detected_pixels = np.sum(pred_mask)

    total_pixels = pred_mask.size

    coverage = (detected_pixels / total_pixels) * 100

    st.subheader("📊 Prediction Statistics")

    st.write(f"Detected Region Pixels: {detected_pixels}")
    st.write(f"Total Pixels: {total_pixels}")
    st.write(f"Coverage Area: {coverage:.2f}%")

else:
    st.info("Upload a satellite image to generate segmentation results.")