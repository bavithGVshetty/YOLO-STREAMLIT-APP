import streamlit as st
import tempfile
import cv2
import imageio
import numpy as np

def run_video_detection(model):
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
            st.video(out_file.name, start_time=0)
