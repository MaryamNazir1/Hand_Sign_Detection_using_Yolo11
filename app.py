import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import tempfile
import time
from ultralytics import YOLO
import subprocess
import os
import base64
# Load the YOLO model
st.set_page_config(
    page_title="Hand Sign Detection using Yolo11"
)
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Function to perform detection and draw bounding boxes on images
def perform_detection(image, model):
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    start_time = time.time()
    results = model(img_rgb)
    detection_time = time.time() - start_time
    boxes = results[0].boxes
    labels = results[0].names
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)  # Convert back to RGB for visualization

    for box in boxes:
        if hasattr(box, 'xyxy') and len(box.xyxy[0]) >= 4:
            x1, y1, x2, y2 = box.xyxy[0][:4]
            conf = box.conf[0]
            cls = int(box.cls[0])
            class_name = labels[cls]
            cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            label = f"{class_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(img_rgb, (int(x1), int(y1) - h - 10), (int(x1) + w, int(y1)), (0, 255, 0), -1)
            cv2.putText(img_rgb, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return img_rgb, detection_time

# Function to perform inference on videos and draw bounding boxes
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    
    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a temporary output video file
    temp_output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(temp_output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), original_fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB, perform detection
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        boxes = results[0].boxes
        labels = results[0].names

        for box in boxes:
            if hasattr(box, 'xyxy') and len(box.xyxy[0]) >= 4:
                x1, y1, x2, y2 = box.xyxy[0][:4]
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = labels[cls]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                label = f"{class_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - h - 10), (int(x1) + w, int(y1)), (0, 255, 0), -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Convert the processed video to H264 format using ffmpeg for better compatibility
    converted_video = './processed_video_h264.mp4'
    subprocess.call(args=f"ffmpeg -y -i {temp_output_video_path} -c:v libx264 {converted_video}".split())

    return converted_video
# Function to embed resized video using HTML
def display_video(video_path):
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    video_url = f'data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}'
    st.markdown(f"""
        <video width="100%"  controls>
            <source src="{video_url}" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)

# Streamlit App
st.title("Hand Sign Detection using Yolo11")

# Model selection option
model_path = "/content/drive/MyDrive/Yolo_dataset/runs/detect/train15/weights/best.onnx"
model = load_model(model_path)

# Sidebar for settings
st.sidebar.write("Settings")
batch_mode = st.sidebar.checkbox("Batch Mode (Multiple Images)", False)
save_results = st.sidebar.checkbox("Save Processed Images/Videos", False)

# Option to upload image(s) or video
upload_type = st.sidebar.radio("Upload Type", ("Image", "Video"))

if upload_type == "Image":
    if batch_mode:
        uploaded_files = st.sidebar.file_uploader("Upload images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    else:
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Process Images
    if batch_mode and uploaded_files:
        for uploaded_file in uploaded_files:
            input_image = Image.open(uploaded_file)
            st.write(f"Processing: {uploaded_file.name}")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Original Image")
                # resized_image = resize_image(input_image, target_height=300)  # Resize the image
                st.image(input_image, caption='Uploaded Image', use_column_width=True)

            processed_image, detection_time = perform_detection(input_image, model)

            with col2:
                st.write(f"Processed Image (Detection Time: {detection_time:.2f}s)")
                # resized_processed_image = resize_image(Image.fromarray(processed_image), target_height=300)  # Resize the processed image
                st.image(processed_image, caption='Detected Image with Bounding Boxes', use_column_width=True)

                # Provide download button if save_results is checked
                if save_results:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
                        img_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(tmpfile.name, img_bgr)
                        st.write(f"Processed image saved to: {tmpfile.name}")
                        st.download_button(
                            label="Download Processed Image",
                            data=open(tmpfile.name, 'rb').read(),
                            file_name=os.path.basename(tmpfile.name),
                            mime="image/jpeg"
                        )

    elif not batch_mode and uploaded_file:
        input_image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.write("Original Image")
            # resized_image = resize_image(input_image, target_height=300)  # Resize the image
            st.image(input_image, caption='Uploaded Image', use_column_width=True)

        processed_image, detection_time = perform_detection(input_image, model)

        with col2:
            st.write(f"Processed Image (Detection Time: {detection_time:.2f}s)")
            # resized_processed_image = resize_image(Image.fromarray(processed_image), target_height=300)  # Resize the processed image
            st.image(processed_image, caption='Detected Image with Bounding Boxes', use_column_width=True)

            # Provide download button if save_results is checked
            if save_results:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
                    img_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(tmpfile.name, img_bgr)
                    st.write(f"Processed image saved to: {tmpfile.name}")
                    st.download_button(
                        label="Download Processed Image",
                        data=open(tmpfile.name, 'rb').read(),
                        file_name=os.path.basename(tmpfile.name),
                        mime="image/jpeg"
                    )

elif upload_type == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])

    if uploaded_video:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.write(f"Processing: {uploaded_video.name}")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Original Video")
            display_video(video_path)

        # Process the video and get the path to the output video
        processed_video_path = process_video(video_path, model)

        with col2:
            st.write("Processed Video with Detections")
            display_video(processed_video_path)

            # Provide download button if save_results is checked
            if save_results:
                st.download_button(
                    label="Download Processed Video",
                    data=open(processed_video_path, 'rb').read(),
                    file_name=os.path.basename(processed_video_path),
                    mime="video/mp4"
                )
