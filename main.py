import streamlit as st
from ultralytics import YOLO

# Import detection modules
from image_detection import run_image_detection
from video_detection import run_video_detection
from web_camera import run_webcam_detection

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

# Run corresponding module
if option == "Evaluate Image":
    run_image_detection(model)
elif option == "Evaluate Video":
    run_video_detection(model)
elif option == "Webcam Detection":
    run_webcam_detection(model)
