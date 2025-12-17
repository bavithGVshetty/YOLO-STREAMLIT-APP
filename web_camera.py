import streamlit as st
import cv2
import time

def run_webcam_detection(model):
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
