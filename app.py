import streamlit as st
import numpy as np
from PIL import Image
from utils.helpers import load_hf_model, preprocess_image, predict_deepfake
import io

@st.cache_resource
def get_model():
    return load_hf_model("maheer24/xceptionnet-deepfake-detector-finetuned")

model = get_model()

st.set_page_config(page_title="DeepFake Detector 💕", page_icon="🎭", layout="centered")

# Custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

 
# HEADER
st.markdown(
    "<h1 class='main-title'>See What's Real.</h1>", unsafe_allow_html=True
)
st.markdown(
    """
<div class='subtitle'>
✨ Upload an image and let our Xception-powered model reveal whether it’s real or AI-generated - fast, accurate, and transparent. 🚀
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")


# IMAGE UPLOAD
st.markdown("### 🖼️ Upload an image to analyze:")
uploaded_file = st.file_uploader("Choose a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image 💖",  use_container_width=True)

    # Preprocess
    img_array = preprocess_image(image)

    if st.button("✨ Analyze DeepFake!"):
        with st.spinner("Thinking really hard 🧠💫..."):
            y_pred = predict_deepfake(model, img_array)
            prediction = "🧑 Real!" if y_pred > 0.5 else "🤖 DeepFake!"
            confidence = y_pred if y_pred > 0.5 else 1 - y_pred

        # DISPLAY RESULT
        st.markdown("<h2 class='result-title'>Result:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p class='prediction'>{prediction}</p>", unsafe_allow_html=True)
        st.progress(float(confidence))
        st.markdown(
            f"<p class='confidence'>Confidence: {confidence*100:.2f}%</p>",
            unsafe_allow_html=True,
        )

        st.balloons()

st.markdown("---")
st.markdown(
    """
<div class='footer'>
👩‍💻 Built by <b>Maheer, Saira & Alishba</b> | Powered by Hugging Face 🤗  
Detecting truth, one pixel at a time 💫
</div>
""",
    unsafe_allow_html=True,
)
