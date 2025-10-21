import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="🐾 Alpaca Dataset Dashboard",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ==========================
# Custom CSS untuk Tema
# ==========================
st.markdown("""
<style>
    body {
        background-color: #f8f6f1;
        color: #4b3b2b;
    }
    .stApp {
        background-color: #f8f6f1;
    }
    h1, h2, h3, h4 {
        color: #4b3b2b;
        font-family: 'Poppins', sans-serif;
    }
    .css-1v3fvcr, .css-1d391kg, .css-qrbaxs {
        background-color: #f0e8dc !important;
    }
    .uploadedFile {
        background-color: #f3ede5 !important;
    }
    .stButton>button {
        background-color: #b89674 !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #9a7b5a !important;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi alpaca
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi alpaca
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Header
# ==========================
st.markdown("<h1 style='text-align:center;'>🦙 Alpaca Dataset Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Deteksi & Klasifikasi Hewan Alpaca dengan Sentuhan Natural</h4>", unsafe_allow_html=True)
st.divider()

# ==========================
# Sidebar
# ==========================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=100)
st.sidebar.markdown("### 🌿 Tentang Proyek")
st.sidebar.write(
    """
    Dashboard ini dibuat untuk mengenali dan mendeteksi **alpaca** 
    serta hewan serupa menggunakan dua model:
    - 🧭 **YOLOv8** untuk deteksi objek  
    - 🧠 **CNN Classifier** untuk klasifikasi gambar  
    """
)

mode = st.sidebar.radio("🔍 Pilih Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
st.sidebar.divider()
st.sidebar.caption("💡 Tips: Gunakan gambar alpaca dengan pencahayaan alami agar hasil lebih akurat.")

# ==========================
# Upload Gambar
# ==========================
uploaded_file = st.file_uploader("📸 Unggah Gambar Alpaca atau Hewan Serupa", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Gambar Diupload", use_container_width=True)
    st.divider()

    if mode == "Deteksi Objek (YOLO)":
        st.subheader("🎯 Hasil Deteksi Objek")
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Deteksi Alpaca", use_container_width=True)
        st.success("✅ Deteksi selesai! Coba gambar lain untuk melihat variasi deteksi.")

    elif mode == "Klasifikasi Gambar":
        st.subheader("🧠 Hasil Klasifikasi Gambar")

        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        st.markdown(f"""
        <div style='background-color:#ede4d1;padding:15px;border-radius:10px;'>
        <h3 style='color:#4b3b2b;'>Prediksi Kelas: <b>{class_index}</b></h3>
        <p style='font-size:16px;'>Probabilitas: <b>{confidence*100:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

else:
    st.info("📂 Silakan unggah gambar terlebih dahulu untuk memulai analisis.")

# ==========================
# Footer
# ==========================
st.divider()
st.markdown(
    "<p style='text-align:center;color:#7a6b5b;'>Dibuat dengan ☕ dan cinta oleh Nela & Ayi 🦙</p>",
    unsafe_allow_html=True
)
