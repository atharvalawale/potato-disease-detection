import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import os
from io import BytesIO

# ---- SETTINGS ----
MODEL_PATH = "model/potato_disease_model.h5"
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']
PRIMARY_COLOR = "#4CAF50"  # Green
SECONDARY_COLOR = "#FF9800"  # Orange

# ---- DISEASE INFO ----
disease_info = {
    "Early Blight": {
        "description": "Fungal disease caused by Alternaria solani. Characterized by dark spots with concentric rings on older leaves.",
        "prescription": ["Remove infected leaves", "Apply fungicides (mancozeb/chlorothalonil)", "Rotate crops annually"],
        "prevention": ["Avoid overhead watering", "Ensure proper plant spacing", "Remove plant debris"]
    },
    "Late Blight": {
        "description": "Destructive disease caused by Phytophthora infestans. Shows large, dark brown lesions with a yellow halo.",
        "prescription": ["Destroy infected plants immediately", "Apply copper-based fungicides", "Avoid working with wet plants"],
        "prevention": ["Plant resistant varieties", "Improve air circulation", "Avoid night watering"]
    },
    "Healthy": {
        "description": "Your potato plant shows no signs of disease. Keep up the good work!",
        "prescription": ["Continue current care routine", "Monitor regularly for early signs", "Maintain proper nutrition"],
        "prevention": ["Practice crop rotation", "Use certified disease-free seeds", "Maintain soil health"]
    }
}

# ---- HELPER FUNCTIONS ----
def load_model_with_check():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {str(e)}")
        st.stop()

def is_valid_file(file):
    if not file: return False
    name = file.name.lower()
    return any(name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])

def plot_prediction(predictions):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(CLASS_NAMES, predictions[0]*100, color=[PRIMARY_COLOR, SECONDARY_COLOR, '#2196F3'])
    ax.set_ylabel('Confidence (%)', fontweight='bold')
    ax.set_title('Disease Prediction Confidence', pad=20, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="PotatoGuard: Disease Detection",
    page_icon="🥔",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---- SIDEBAR ----
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2909/2909495.png", width=80)
    st.title("PotatoGuard")
    st.markdown("""
    **About this app:**
    - Uses deep learning to detect potato diseases
    - Identifies Early Blight, Late Blight, or Healthy plants
    - Provides treatment recommendations
    """)
    

# ---- MAIN INTERFACE ----
st.title("🥔 PotatoGuard: Disease Detection")
st.markdown("Upload an image of potato leaves to detect diseases and get treatment recommendations.")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=None, 
                                help="Supported formats: JPG, JPEG, PNG",
                                key="file_uploader")

if uploaded_file and is_valid_file(uploaded_file):
    try:
        # Display original image
        img = Image.open(BytesIO(uploaded_file.getvalue()))
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="Original Image", use_column_width=True)
        
        # Preprocess image
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        if len(img_array.shape) == 2:  # Handle grayscale
            img_array = np.stack((img_array,)*3, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        model = load_model_with_check()
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = np.max(predictions) * 100
        
        # Display prediction results
        with col2:
            st.markdown("### 🔍 Analysis Results")
            
            # Prediction confidence chart
            st.pyplot(plot_prediction(predictions))
            
            # Prediction badge
            if predicted_label == "Healthy":
                st.success(f"## ✅ {predicted_label} ({confidence:.1f}%)")
            else:
                st.error(f"## ⚠️ {predicted_label} ({confidence:.1f}%)")
            
            st.markdown(f"**Description**: {disease_info[predicted_label]['description']}")
        
        # Treatment and prevention cards
        st.markdown("---")
        col3, col4 = st.columns([1, 1])
        
        with col3:
            with st.expander("💊 **Recommended Treatment**", expanded=True):
                for item in disease_info[predicted_label]['prescription']:
                    st.markdown(f"- {item}")
        
        with col4:
            with st.expander("🛡️ **Prevention Tips**", expanded=True):
                for item in disease_info[predicted_label]['prevention']:
                    st.markdown(f"- {item}")
        
        # Disease comparison table
        st.markdown("---")
        st.subheader("📊 Disease Comparison")
        df = pd.DataFrame({
            'Disease': CLASS_NAMES,
            'Confidence (%)': [x*100 for x in predictions[0]]
        }).sort_values('Confidence (%)', ascending=False)
        st.dataframe(df.style.highlight_max(axis=0, color='#E6F9E6'), 
                    use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.error("Please try with a different image file.")

elif uploaded_file and not is_valid_file(uploaded_file):
    st.error("⚠️ Invalid file type. Please upload a JPG, JPEG, or PNG image.")

# ---- FOOTER ----
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 0.8em;
    color: #777;
    text-align: center;
    padding: 10px;
}
</style>
<div class="footer">
    PotatoGuard v1.0 | For agricultural use only | Not a substitute for professional diagnosis
</div>
""", unsafe_allow_html=True)