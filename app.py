"""
AI Playground: Face Detection App
Built with Streamlit and OpenCV's Haar Cascade classifier.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import urllib.request

# ── Page Setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Playground: Face Detection",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #fafafa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    .stApp header {
        background-color: transparent;
    }
    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

# ── Title & Intro ─────────────────────────────────────────────────────────
st.title("👤 AI Playground: Face Detection")
st.write(
    "Identify and locate human faces in images using **Computer Vision** and "
    "**OpenCV's Haar Cascade Classifier**. You can upload your own photo, "
    "use your webcam, or select from classic sample images."
)
st.divider()

# ── Robust Cascade Classifier Loader ──────────────────────────────────────
def load_cascade_classifier():
    cascade_filename = "haarcascade_frontalface_default.xml"
    
    # 1. Local project file (bundled in repository)
    local_path = os.path.join(os.path.dirname(__file__), cascade_filename)
    if os.path.exists(local_path):
        detector = cv2.CascadeClassifier(local_path)
        if not detector.empty():
            return detector

    # 2. Default OpenCV data paths
    default_path = os.path.join(cv2.data.haarcascades, cascade_filename)
    if os.path.exists(default_path):
        detector = cv2.CascadeClassifier(default_path)
        if not detector.empty():
            return detector
            
    # 3. Download fallback if not present
    local_backup_path = os.path.join(os.getcwd(), cascade_filename)
    if not os.path.exists(local_backup_path):
        github_url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{cascade_filename}"
        try:
            urllib.request.urlretrieve(github_url, local_backup_path)
        except Exception as e:
            st.error(f"Failed to download cascade classifier: {e}")
            return None
            
    detector = cv2.CascadeClassifier(local_backup_path)
    if detector.empty():
        return None
    return detector

face_detector = load_cascade_classifier()

if face_detector is None:
    st.error("❌ Critical Error: Could not load the Haar Cascade face detector XML. Please check your network connection.")
    st.stop()

# ── Sidebar Configuration ──────────────────────────────────────────────────
st.sidebar.header("⚙️ Detection Settings")
st.sidebar.write("Fine-tune the face detection algorithm parameters:")

# 1. scaleFactor
scale_factor = st.sidebar.slider(
    "Scale Factor",
    min_value=1.05,
    max_value=1.50,
    value=1.10,
    step=0.05,
    help="How much the image size is reduced at each image scale. Lower values are more thorough but slower."
)

# 2. minNeighbors
min_neighbors = st.sidebar.slider(
    "Min Neighbors",
    min_value=1,
    max_value=15,
    value=5,
    step=1,
    help="How many neighbors each candidate rectangle should have to retain it. Higher values reduce false positives."
)

# 3. minSize
min_size_val = st.sidebar.slider(
    "Min Size (pixels)",
    min_value=10,
    max_value=150,
    value=30,
    step=10,
    help="Minimum possible object size. Objects smaller than this are ignored."
)

st.sidebar.divider()
st.sidebar.header("🎨 Styling Settings")

# 4. Box Color
box_color_hex = st.sidebar.color_picker("Bounding Box Color", "#00E5FF")
# Convert Hex to BGR (OpenCV expects BGR tuple)
hex_clean = box_color_hex.lstrip('#')
box_color_rgb = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
box_color_bgr = (box_color_rgb[2], box_color_rgb[1], box_color_rgb[0])

# 5. Box Thickness
box_thickness = st.sidebar.slider(
    "Line Thickness",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

# ── Main Content Tabs ─────────────────────────────────────────────────────
tab_upload, tab_webcam, tab_samples = st.tabs([
    "📤 Upload an Image",
    "📸 Use Your Webcam",
    "🖼️ Classic Sample Images"
])

input_image = None
image_name = ""

# --- Upload Tab ---
with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a JPEG or PNG image...",
        type=["jpg", "jpeg", "png"],
        key="uploader"
    )
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            input_image = np.array(pil_image)
            image_name = uploaded_file.name
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")

# --- Webcam Tab ---
with tab_webcam:
    st.write("Capture a photo using your webcam to run the detector live!")
    camera_photo = st.camera_input("Take a picture", key="webcam")
    if camera_photo is not None:
        try:
            pil_image = Image.open(camera_photo).convert("RGB")
            input_image = np.array(pil_image)
            image_name = "webcam_capture.jpg"
        except Exception as e:
            st.error(f"Error capturing image from webcam: {e}")

# --- Sample Images Tab ---
with tab_samples:
    st.write("Test the algorithm instantly with standard Computer Vision datasets:")
    
    col_sample1, col_sample2 = st.columns(2)
    
    # Pre-defined OpenCV samples
    SAMPLES = {
        "Lena (Classic Face)": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "Group/Crowd Photo": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/group.jpg"
    }
    
    selected_sample = st.selectbox(
        "Choose a sample image:",
        options=list(SAMPLES.keys()),
        index=0
    )
    
    if st.button("Load Selected Sample"):
        sample_url = SAMPLES[selected_sample]
        sample_filename = f"sample_{selected_sample.replace(' ', '_').lower()}.jpg"
        
        # Download if not already cached
        if not os.path.exists(sample_filename):
            with st.spinner("Downloading sample image..."):
                try:
                    urllib.request.urlretrieve(sample_url, sample_filename)
                except Exception as e:
                    st.error(f"Error downloading sample image: {e}")
        
        if os.path.exists(sample_filename):
            try:
                pil_image = Image.open(sample_filename).convert("RGB")
                input_image = np.array(pil_image)
                image_name = sample_filename
                st.session_state["loaded_sample"] = input_image
                st.session_state["sample_name"] = sample_filename
            except Exception as e:
                st.error(f"Error parsing sample image: {e}")

    # Fallback/Keep state of loaded sample image
    if input_image is None and "loaded_sample" in st.session_state:
        input_image = st.session_state["loaded_sample"]
        image_name = st.session_state["sample_name"]

# ── Processing & Detection ───────────────────────────────────────────────
if input_image is not None:
    st.write("---")
    
    # Original image in BGR format for OpenCV compatibility
    # streamlit arrays are RGB, OpenCV expects BGR
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Run Detector
    with st.spinner("Analyzing image for faces..."):
        faces = face_detector.detectMultiScale(
            img_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size_val, min_size_val)
        )
    
    # Create Columns for Visual Outputs
    col_out, col_metrics = st.columns([3, 1])
    
    with col_metrics:
        st.subheader("📊 Statistics")
        st.metric(label="Faces Detected", value=len(faces))
        
        st.subheader("💡 Param Settings")
        st.write(f"**Scale Factor:** {scale_factor}")
        st.write(f"**Min Neighbors:** {min_neighbors}")
        st.write(f"**Min Size:** {min_size_val}px")
        
    with col_out:
        st.subheader("🔍 Detection Result")
        
        # Draw bounding boxes on a copy of BGR image
        output_bgr = img_bgr.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(
                output_bgr,
                (x, y),
                (x + w, y + h),
                box_color_bgr,
                box_thickness
            )
            
        # Convert back to RGB for display in Streamlit
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        
        # Display image side-by-side or tabs
        st.image(
            output_rgb,
            caption=f"Processed: {image_name} — Detected {len(faces)} face(s)",
            use_column_width=True
        )

    # ── Cropped Faces Gallery ─────────────────────────────────────────────
    if len(faces) > 0:
        st.subheader("✂️ Cropped Faces Gallery")
        st.write("Each detected face cropped individually from the original image:")
        
        # Determine columns layout
        gallery_cols = st.columns(min(len(faces), 5))
        
        for idx, (x, y, w, h) in enumerate(faces):
            # Crop the face from the original RGB image
            face_crop = input_image[y:y+h, x:x+w]
            
            # Put in corresponding column (wrap around if more than 5)
            col_idx = idx % 5
            with gallery_cols[col_idx]:
                st.image(
                    face_crop,
                    caption=f"Face #{idx + 1}",
                    use_column_width=True
                )
    else:
        st.info("ℹ️ No faces were detected. Try adjusting the **Scale Factor** or **Min Neighbors** in the sidebar.")

# ── Educational Section ──────────────────────────────────────────────────
st.divider()
st.subheader("📖 Learning & Explanation Corner")

col_edu1, col_edu2 = st.columns(2)

with col_edu1:
    st.markdown("""
    ### How Face Detection Works (Haar Cascades)
    Unlike deep learning models which process millions of weight parameters, a **Haar Cascade** is a classic machine learning method for object detection.
    
    1. **Grayscale Conversion**: Face shape and structural contrast are color-invariant, so the image is first converted to grayscale.
    2. **Haar Features**: It uses edge, line, and four-rectangle features to find contrasts. For instance, the eye region is typically darker than the bridge of the nose.
    3. **Integral Image**: A mathematical representation that makes calculating features extremely fast.
    4. **Cascade Classifier**: It uses a cascade of classifiers where weak learners are grouped. If a window fails a stage, it is discarded immediately to save compute time.
    """)

with col_edu2:
    with st.expander("💼 Interview Q&A Corner"):
        st.write("""
        **Q: What is the difference between face detection and face recognition?**
        * **Face Detection**: Finds the position/coordinates of human faces in an image ("Is there a face here, and where?").
        * **Face Recognition**: Identifies whose face it is ("Who is this specific person?").
        
        **Q: Why do Haar Cascades run on grayscale images?**
        * Color channels add computational overhead and do not contribute to locating structural facial features like shadows, eye sockets, and nose ridges.
        
        **Q: What happens if Min Neighbors is set too low?**
        * Setting it too low causes the algorithm to capture more false positives (detecting faces in random objects or textures) because it requires fewer overlapping matching windows.
        
        **Q: What are the limitations of Haar Cascades compared to deep learning detectors?**
        * Haar Cascades struggle with non-frontal faces (profiles), occlusions, poor lighting, and head rotation. Modern deep learning tools like MediaPipe or MTCNN are much more robust but require more computing power.
        """)
