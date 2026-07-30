# 👤 Face Detection Streamlit Application

An interactive, premium web application built with **Streamlit** and **OpenCV** that detects human faces in real-time. It implements the face detection logic from **Project 3** of the `AI_Playground_4_Real_World_AI_Projects_v4.ipynb` notebook.

## 🚀 Features

- **Multi-Source Inputs**: 
  - Upload your own photographs (`.jpg`, `.jpeg`, `.png`).
  - Capture real-time images directly using your computer's webcam.
  - Test instantly using preloaded computer vision benchmark sample images (Lena, Group Photo).
- **Interactive Tuning**: 
  - Dynamic sliders to control Haar Cascade parameters: `Scale Factor`, `Min Neighbors`, and `Min Size`.
  - Bounding box customization: Color picker and line thickness adjustments.
- **Cropped Face Gallery**: Automatically crops and presents each detected face in an interactive gallery layout.
- **Educational Content**: Embedded explanations on how Haar Cascades work and interactive interview Q&A.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Clone/Move to the Project Directory
```bash
cd c:/Users/abhineetsinha/Documents/face-detection-streamlit
```

### 3. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Application
```bash
streamlit run app.py
```

---

## ⚙️ How Face Detection Parameters Work

- **Scale Factor**: Specifies how much the image size is reduced at each image scale. For example, `1.10` means the image is scaled down by 10% in each iteration. Lower values are more thorough but slower.
- **Min Neighbors**: Specifies how many neighbors each candidate rectangle should have to retain it. Higher values result in fewer detections but with higher quality (reduces false positives).
- **Min Size**: Minimum possible object size. Objects smaller than this are ignored.
