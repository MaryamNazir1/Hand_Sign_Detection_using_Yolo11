# Hand Sign Detection

This repository provides a tool for detecting hand signs using a fine-tuned YOLO11 model and a Streamlit web application. The project allows users to upload images or videos and receive real-time hand sign detection results.

## Features

- **Real-time Hand Sign Detection**: Uses a YOLO11 model to detect various hand signs in uploaded images or videos.
- **User-Friendly Interface**: Built with Streamlit for easy interaction, allowing users to upload files and view results.
- **Downloadable Results**: Users can download processed images and videos with bounding boxes and confidence scores.
  
## Demo

<img src="src/demo.gif" alt="Hand Sign Detection Demo" width="600" height="350">

---

## Getting Started

### 1. Clone the Repository

Open **Command Prompt** or **Terminal** and clone the repository:

```bash
git clone https://github.com/maryamnazir1/Hand_Sign_Detection_using_Yolo11.git
cd Hand_Sign_Detection_using_Yolo11

```
### 2. Install Dependencies
To run the application, you need to install the required Python libraries. You can do this using pip:

```bash
pip install streamlit opencv-python-headless matplotlib ultralytics
```
### 3. Run the Application
Once the dependencies are installed, you can run the Streamlit application using the following command:

```bash
streamlit run app.py
```
This will start the app, and you can access it in your web browser at http://localhost:8501.

### 4. Model Training
If you want to train the YOLO11 model on your own dataset, follow these steps:

1. Prepare your dataset in the YOLO format.
2. Utilize the training scripts provided in the train/ directory to fine-tune the model.
3. Adjust the paths in the configuration files as necessary.
4. For detailed training instructions, please refer to the YOLOv8 documentation.
5. You can check the train model in this path 'runs/detect/train15/weights/best.onnx'.
