
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="AI Vision Scanner", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Description
st.title("🔍 AI Object Detector")
st.write("Upload an image and let the AI identify objects in real-time — **No API required!**")

# Load Model (Cache it so it doesn't reload every time)
@st.cache_resource
def load_model():
    # 'yolov8n.pt' is a pre-trained model (approx 6MB) 
    # It will download automatically on the first run
    model = YOLO('yolov8n.pt') 
    return model

model = load_model()

# Sidebar for settings
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4)

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and Display Image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Original Image")
        st.image(image, use_column_width=True)

    # Perform Detection
    if st.button("Analyze Image"):
        with st.spinner('AI is thinking...'):
            # Convert PIL to numpy
            img_array = np.array(image)
            
            # Run Inference
            results = model.predict(img_array, conf=confidence)
            
            # Plot results
            res_plotted = results[0].plot()
            
            with col2:
                st.success("AI Result")
                st.image(res_plotted, caption="Objects Detected", use_column_width=True)
                
            # Show details in a table
            st.write("### Detection Summary")
            det_data = []
            for box in results[0].boxes:
                name = model.names[int(box.cls)]
                conf = float(box.conf)
                det_data.append({"Object": name, "Confidence": f"{conf:.2%}"})
            
            if det_data:
                st.table(det_data)
            else:
                st.warning("No objects detected at this confidence level.")
else:
    st.info("Please upload an image to start.")
