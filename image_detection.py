import streamlit as st
from PIL import Image
import numpy as np
import cv2

def run_image_detection(model):
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Run YOLO Detection"):
            # Convert PIL image to OpenCV format
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            results = model.predict(frame, verbose=False)
            annotated_image = results[0].plot()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image, caption="Detected Image", use_column_width=True)
