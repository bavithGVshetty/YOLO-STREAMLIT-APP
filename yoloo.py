import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import numpy as np
import time
import os
# import imageio

st.set_page_config(page_title="YOLOv10 Detection App", layout="wide")
st.title("YOLOv10 Detection App")

# Load the YOLOv10 model once
model_path = "D:\\YOLO\\yolov10n.pt"  # adjust path
model = YOLO(model_path)

# Sidebar options
option = st.sidebar.selectbox(
    "Select Mode",
    ["Evaluate Image", "Evaluate Video", "Webcam Detection"]
)

# ----------------- IMAGE DETECTION -----------------
if option == "Evaluate Image":
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

# ----------------- VIDEO DETECTION -----------------
import streamlit as st
import tempfile
import cv2
import imageio

if option == "Evaluate Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        video_path = tfile.name

        if st.button("Run YOLO on Video"):
            reader = imageio.get_reader(video_path)
            fps = reader.get_meta_data()['fps']

            out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_file.close()
            writer = imageio.get_writer(out_file.name, fps=fps, codec='libx264')

            progress_bar = st.progress(0)
            total_frames = reader.count_frames()

            for i, frame in enumerate(reader):
                # YOLO processing
                frame_resized = cv2.resize(frame, (640, 640))
                results = model.predict(frame_resized, verbose=False)
                annotated_frame = results[0].plot()
                annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))
                writer.append_data(annotated_frame.astype('uint8'))
                progress_bar.progress((i+1)/total_frames)

            reader.close()
            writer.close()

            st.success("Video processed successfully!")
            # Autoplay video
            st.video(out_file.name, start_time=0)  # starts immediately


# ----------------- WEBCAM DETECTION -----------------
elif option == "Webcam Detection":
    st.warning("Webcam detection works locally only.")
    
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
        else:
            stframe = st.empty()
            stop = st.button("Stop Webcam")
            while cap.isOpened() and not stop:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(frame, verbose=False)
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame)
                time.sleep(0.03)
            cap.release()
