import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time

st.set_page_config(
    page_title="Orange Quality Detection System",
    page_icon="🍊",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .quality-metric {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .fresh {
        background-color: #90EE90;
        color: #006400;
    }
    .rotten {
        background-color: #FFB6C1;
        color: #8B0000;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_model():
    return YOLO(r"D:\Fruit Ninja\runs\detect\orange_classifier_rtx40508\weights\best.pt").to('cuda')

def initialize_cameras():
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    return cap1, cap2

def detect_quality(frame, model):
    results = model(frame, conf=0.25)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            label = model.names[class_id]
            return label
    return None

def conclude_quality(quality_cam1, quality_cam2):
    if quality_cam1 == "fresh" and quality_cam2 == "fresh":
        return "fresh"
    else:
        return "rotten"

def main():
    st.title("🍊 Orange Quality Detection System")
    st.markdown("---")

    model = initialize_model()
    cap1, cap2 = initialize_cameras()

    if not cap1.isOpened() or not cap2.isOpened():
        st.error("Error: Could not open one or both cameras!")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Camera 1")
        camera1_placeholder = st.empty()
    
    with col2:
        st.subheader("Camera 2")
        camera2_placeholder = st.empty()

    quality_placeholder = st.empty()

    stop_button = st.button("Stop Detection")

    while not stop_button:
        try:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                st.error("Failed to capture frames from cameras")
                break

            results1 = model(frame1, conf=0.25)
            results2 = model(frame2, conf=0.25)

            annotated_frame1 = results1[0].plot()
            annotated_frame2 = results2[0].plot()

            rgb_frame1 = cv2.cvtColor(annotated_frame1, cv2.COLOR_BGR2RGB)
            rgb_frame2 = cv2.cvtColor(annotated_frame2, cv2.COLOR_BGR2RGB)

            camera1_placeholder.image(rgb_frame1, channels="RGB", use_container_width=True)
            camera2_placeholder.image(rgb_frame2, channels="RGB", use_container_width=True)

            quality_cam1 = detect_quality(frame1, model)
            quality_cam2 = detect_quality(frame2, model)

            if quality_cam1 is not None and quality_cam2 is not None:
                final_quality = conclude_quality(quality_cam1, quality_cam2)
                
                quality_html = f"""
                    <div class="quality-metric {'fresh' if final_quality == 'fresh' else 'rotten'}">
                        <h2>Current Detection: {final_quality.upper()}</h2>
                        <p>Camera 1: {quality_cam1}</p>
                        <p>Camera 2: {quality_cam2}</p>
                    </div>
                """
                quality_placeholder.markdown(quality_html, unsafe_allow_html=True)

            time.sleep(0.1)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            break

    cap1.release()
    cap2.release()

if __name__ == "__main__":
    main()
