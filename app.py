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
    page_title="🦙 Deteksi Alpaca & Non-Alpaca",
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
        color: #3b2f1d;
    }
    .stApp {
        background-color: #f8f6f1;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #3b2f1d !important;
        font-family: 'Poppins', sans-serif;
    }
    .stButton>button {
        background-color: #b28b67 !important;
        color: #fff !important;
        border-radius: 10px !important;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #9c7651 !important;
        color: #fff;
    }
    .stSidebar {
        background-color: #f2ece3 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Header dengan Gambar Alpaca
# ==========================
st.markdown("<h1 style='text-align:center;'>🦙 Deteksi Objek Alpaca dan Non-Alpaca</h1>", unsafe_allow_html=True)
st.image(
    "https://images.unsplash.com/photo-1608572253294-57a2a72b3b4b?auto=format&fit=crop&w=1000&q=80",
    caption="Alpaca di padang rumput — sumber: Unsplash",
    use_container_width=True
)
st.markdown("<h4 style='text-align:center;'>Menggunakan Model YOLO untuk Deteksi dan CNN untuk Klasifikasi</h4>", unsafe_allow_html=True)
st.divider()

# ==========================
# Sidebar
# ==========================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=90)
st.sidebar.markdown("### 🌿 Tentang Aplikasi")
st.sidebar.write(
    """
    Aplikasi ini mendeteksi keberadaan **alpaca** 
    serta membedakannya dengan hewan lain menggunakan:
    - 🧭 **YOLOv8** untuk deteksi objek  
    - 🧠 **CNN Classifier** untuk klasifikasi gambar  
    """
)

mode = st.sidebar.radio("🔍 Pilih Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
st.sidebar.divider()
st.sidebar.caption("💡 Unggah gambar dengan pencahayaan alami untuk hasil terbaik.")

# ==========================
# Upload Gambar
# ==========================
uploaded_file = st.file_uploader("📸 Unggah Gambar Alpaca atau Hewan Lain", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Gambar yang Diupload", use_container_width=True)
    st.divider()

    if mode == "Deteksi Objek (YOLO)":
        st.subheader("🎯 Hasil Deteksi Objek")
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Deteksi Objek Alpaca dan Non-Alpaca", use_container_width=True)
        st.success("✅ Deteksi selesai! Coba gambar lain untuk melihat hasil berbeda.")

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
        <div style='background-color:#efe7dc;padding:15px;border-radius:10px;'>
        <h3 style='color:#3b2f1d;'>Prediksi Kelas: <b>{class_index}</b></h3>
        <p style='font-size:16px;color:#3b2f1d;'>Probabilitas: <b>{confidence*100:.2f}%</b></p>
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
    "<p style='text-align:center;color:#3b2f1d;'>Dibuat dengan ☕ dan semangat oleh Nela & Ayi 🦙</p>",
    unsafe_allow_html=True
)
