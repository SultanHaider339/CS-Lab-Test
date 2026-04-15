import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw

# Page configuration
st.set_page_config(page_title="AI Vision Scanner", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 AI Object Detector")
st.write("Using Hugging Face Transformers — **No API required!**")

# Load Model (Cache it to avoid reloading)
@st.cache_resource
def load_model():
    # Facebook's DETR model pre-trained for object detection
    # Ye pehli dafa run hone par automatically background mein download ho ga
    return pipeline("object-detection", model="facebook/detr-resnet-50")

detector = load_model()

# Sidebar for settings
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open Image and ensure it's in RGB format
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Original Image")
        st.image(image, use_column_width=True)

    # Perform Detection
    if st.button("Analyze Image"):
        with st.spinner('AI is thinking (this might take a few seconds on first run)...'):
            
            # Run Inference
            results = detector(image)
            
            # Draw bounding boxes on the image using Pillow (No OpenCV needed)
            draw = ImageDraw.Draw(image)
            det_data = []
            
            for result in results:
                score = result['score']
                
                # Check if confidence is above the threshold
                if score >= confidence_threshold:
                    box = result['box']
                    label = result['label']
                    
                    # Draw Rectangle around object
                    draw.rectangle(
                        [(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])],
                        outline="red",
                        width=3
                    )
                    
                    # Write Label text
                    draw.text(
                        (box['xmin'], max(0, box['ymin'] - 15)), 
                        f"{label.capitalize()} ({round(score, 2)})", 
                        fill="red"
                    )
                    
                    # Save data for the table
                    det_data.append({"Object": label.capitalize(), "Confidence": f"{score:.2%}"})
            
            with col2:
                st.success("AI Result")
                st.image(image, caption="Objects Detected", use_column_width=True)
                
            # Show details in a table
            st.write("### Detection Summary")
            if det_data:
                st.table(det_data)
            else:
                st.warning("No objects detected at this confidence level. Try lowering the threshold from the sidebar.")
else:
    st.info("Please upload an image to start.")
