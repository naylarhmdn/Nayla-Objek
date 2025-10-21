import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/classifier_model.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI Styling
# ==========================
st.set_page_config(page_title="Deteksi Alpaca dan Non-Alpaca", layout="centered")

# Warna tema otomatis (dark / light)
theme = st.get_option("theme.base") if "theme.base" in st.session_state else "light"
if theme == "dark":
    text_color = "#FAF9F6"  # terang
else:
    text_color = "#3B2F2F"  # coklat tua

st.markdown(
    f"""
    <style>
        body {{
            background-color: #F5F2E7; /* broken white */
            color: {text_color};
            font-family: 'Poppins', sans-serif;
            text-align: center;
        }}
        .stApp {{
            background-color: #F5F2E7;
        }}
        h1 {{
            color: {text_color};
        }}
        .uploadedImage {{
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# UI
# ==========================
st.title("🦙 Deteksi Alpaca dan Non-Alpaca")

uploaded_file = st.file_uploader("Unggah gambar alpaca atau non-alpaca", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Gambar yang diunggah", use_container_width=True, output_format="auto")

    # YOLO Deteksi
    results = yolo_model(img)
    result_img = results[0].plot()
    st.image(result_img, caption="🎯 Hasil Deteksi Objek", use_container_width=True)

    # Klasifikasi
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = classifier.predict(img_array)
    class_index = np.argmax(prediction)
    labels = ["Non-Alpaca", "Alpaca"]

    st.subheader(f"Hasil Prediksi: **{labels[class_index]}**")
    st.write(f"Probabilitas: **{np.max(prediction)*100:.2f}%**")

# ==========================
# Footer Alpaca Image
# ==========================
st.markdown("---")
st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Alpaca_lying_down.jpg", caption="Alpaca 🦙", use_container_width=True)
