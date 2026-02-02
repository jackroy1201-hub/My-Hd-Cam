import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# 1. Page Config
st.set_page_config(page_title="Family AI Master Studio", layout="wide")

# --- Functions Area ---

def enhance_to_8k(img):
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    kernel = np.array([[-0.1,-0.1,-0.1], [-0.1,1.8,-0.1], [-0.1,-0.1,-0.1]])
    return cv2.filter2D(upscaled, -1, kernel)

def apply_face_wash(img):
    # نیچرل گلو: کنٹراسٹ بڑھائے بغیر صفائی
    smoothed = cv2.bilateralFilter(img, 15, 50, 50)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, 8) 
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def apply_natural_night(img):
    b, g, r = cv2.split(img)
    g = cv2.add(g, 35)
    r = cv2.subtract(r, 10)
    b = cv2.subtract(b, 10)
    return cv2.merge((b, g, r))

# --- UI Styling ---
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>💎 AI Family Master Studio</h1>", unsafe_allow_html=True)

img_file = st.file_uploader("تصویر اپلوڈ کریں", type=["jpg", "png", "jpeg"])

if img_file:
    if 'original' not in st.session_state or st.session_state.get('last_file') != img_file.name:
        raw_img = Image.open(img_file).convert("RGB")
        st.session_state.original = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
        st.session_state.processed = st.session_state.original.copy()
        st.session_state.last_file = img_file.name

if 'processed' in st.session_state:
    # --- Action Buttons Row 1 ---
    st.write("### 🛠 مین پاور ٹولز")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧼 Face Glow (Natural)"):
            st.session_state.processed = apply_face_wash(st.session_state.processed)
    with col2:
        if st.button("🧵 Silk Hair (سیٹ بال)"):
            st.session_state.processed = cv2.medianBlur(st.session_state.processed, 3)
            st.session_state.processed = cv2.bilateralFilter(st.session_state.processed, 10, 40, 40)
    with col3:
        if st.button("🚀 8K Ultra HD"):
            st.session_state.processed = enhance_to_8k(st.session_state.processed)

    # --- Filters Row 2 ---
    st.write("### 📱 اسمارٹ موبائل فلٹرز")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        if st.button("🍎 iPhone Mode"):
            img = st.session_state.processed.astype(np.float32)
            img[:, :, 2] *= 1.05 # Warmth
            st.session_state.processed = np.clip(img, 0, 255).astype(np.uint8)
    with f2:
        if st.button("📸 Oppo Beauty"):
            st.session_state.processed = cv2.convertScaleAbs(st.session_state.processed, alpha=1.05, beta=10)
    with f3:
        if st.button("🌙 Natural Night"):
            st.session_state.processed = apply_natural_night(st.session_state.processed)
    with f4:
        if st.button("🔄 Reset All"):
            st.session_state.processed = st.session_state.original.copy()
            st.rerun()

    # --- Advanced Tools (Hair Color & Zoom) ---
    st.divider()
    st.write("### 🎨 ایڈوانسڈ کلرنگ اور زوم")
    c_h1, c_h2, c_h3 = st.columns(3)
    with c_h1:
        hair_shades = {"Default": None, "Jet Black": [10, 10, 10], "Deep Brown": [30, 50, 90], "Golden": [40, 180, 220], "Pink": [150, 80, 250]}
        h_col = st.selectbox("بالوں کا رنگ بدلیں", list(hair_shades.keys()))
    with c_h2:
        h_int = st.slider("رنگ کی شدت (Intensity)", 0.0, 1.0, 0.5)
    with c_h3:
        zoom = st.slider("زوم لیول (Zoom)", 1.0, 3.0, 1.0)

    # --- Final Manual Tuning ---
    st.write("### 🏁 فائنل ٹچ")
    m1, m2 = st.columns(2)
    with m1:
        bright = st.slider("برائٹنس", 0.8, 1.3, 1.0)
    with m2:
        vibrance = st.slider("رنگوں کی گہرائی (Vibrance)", 1.0, 1.3, 1.05)

    # --- Processing Final Image ---
    final = st.session_state.processed.copy()
    
    # 1. Zoom
    if zoom > 1.0:
        h, w = final.shape[:2]
        nh, nw = int(h/zoom), int(w/zoom)
        sh, sw = (h-nh)//2, (w-nw)//2
        final = cv2.resize(final[sh:sh+nh, sw:sw+nw], (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    # 2. Hair Coloring
    if h_col != "Default":
        hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        mask_3d = np.stack([cv2.GaussianBlur(mask, (15,15), 0)]*3, axis=-1) / 255.0
        final = (final * (1 - mask_3d * h_int) + np.array(hair_shades[h_col]) * (mask_3d * h_int)).astype(np.uint8)

    # 3. Vibrance
    if vibrance > 1.0:
        hsv_v = cv2.cvtColor(final, cv2.COLOR_BGR2HSV).astype("float32")
        hsv_v[:,:,1] *= vibrance
        final = cv2.cvtColor(np.clip(hsv_v, 0, 255).astype("uint8"), cv2.COLOR_HSV2BGR)
    
    final = cv2.convertScaleAbs(final, alpha=bright, beta=0)

    # --- Display & Download ---
    st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    _, buffer = cv2.imencode(".jpg", final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    st.download_button("📥 DOWNLOAD 8K STUDIO PHOTO", buffer.tobytes(), "Family_AI_Pro.jpg", "image/jpeg")

st.info("💡 آپ کے تمام پسندیدہ فیچرز اب ایک ہی جگہ موجود ہیں۔ پہلے 'Face Glow' اور 'Silk Hair' استعمال کریں، پھر '8K' کریں۔")
