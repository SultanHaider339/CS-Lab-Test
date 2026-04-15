import streamlit as st
import cv2
import numpy as np
from PIL import Image

# UI Design
st.set_page_config(page_title="Lightweight AI Scanner", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    </style>
    """, unsafe_allow_html=True)

st.title("👤 Fast & Light Face Detector")
st.write("Bilkul light-weight AI app! Sirf OpenCV use kar rahi hai (No PyTorch or heavy libraries).")

# Load Light-weight Pre-trained Model
@st.cache_resource
def load_model():
    # OpenCV ka built-in pre-trained face detection model (Sirf kuch KBs ka hota hai)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade

face_detector = load_model()

# Image Uploader
uploaded_file = st.file_uploader("Koi tasveer (image) upload karein jismein faces hon...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Original Image")
        st.image(image, use_column_width=True)

    if st.button("Detect Faces"):
        with st.spinner("AI is scanning..."):
            # Convert image for OpenCV processing
            img_array = np.array(image)
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Detect faces in the image
            faces = face_detector.detectMultiScale(
                gray_img, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # Draw green rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 4)
            
            with col2:
                st.success(f"Success! {len(faces)} face(s) detected. 🎉")
                st.image(img_array, caption="Processed Image", use_column_width=True)
                
else:
    st.info("App start karne ke liye koi image upload karein.")
