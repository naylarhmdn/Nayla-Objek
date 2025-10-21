import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import datetime

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(page_title="🦙 Deteksi Alpaca & Non-Alpaca", page_icon="🦙", layout="centered")

# ==========================
# Tema Adaptif (otomatis)
# ==========================
hour = datetime.datetime.now().hour
is_dark = hour >= 18 or hour < 6  # malam → gelap

if is_dark:
    bg_color = "#2b2118"  # coklat gelap
    text_color = "#f8f6f1"  # broken white
else:
    bg_color = "#f8f6f1"  # broken white
    text_color = "#3b2f1d"  # coklat tua

st.markdown(f"""
<style>
    body, .stApp {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'Poppins', sans-serif;
    }}
    h1, h2, h3, h4, h5, h6, p, label, span, div {{
        color: {text_color} !important;
    }}
    .stButton>button {{
        background-color: {'#b28b67' if not is_dark else '#8b6a4a'};
        color: white !important;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)

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
# Header
# ==========================
st.markdown(f"<h1 style='text-align:center;'>🦙 Deteksi Alpaca & Non-Alpaca</h1>", unsafe_allow_html=True)
st.image(
    "https://images.unsplash.com/photo-1608572253294-57a2a72b3b4b?auto=format&fit=crop&w=1000&q=80",
    use_container_width=True,
)

# ==========================
# Mode & Upload
# ==========================
mode = st.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# Proses Gambar
# ==========================
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar Diupload", use_container_width=True)

    if mode == "Deteksi Objek (YOLO)":
        results = yolo_model(img)
        st.image(results[0].plot(), caption="Hasil Deteksi", use_container_width=True)

    else:
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = classifier.predict(img_array)
        st.write("**Prediksi:**", np.argmax(prediction))
        st.write("**Probabilitas:**", f"{np.max(prediction)*100:.2f}%")

else:
    st.info("📂 Unggah gambar untuk mulai deteksi atau klasifikasi.")

# ==========================
# Footer
# ==========================
st.markdown(f"<p style='text-align:center;color:{text_color};'>© Nela & Ayi 🦙</p>", unsafe_allow_html=True)
