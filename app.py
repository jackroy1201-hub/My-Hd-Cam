import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# 1. Page Config
st.set_page_config(page_title="Family AI Pro Studio", layout="wide")

# 2. Advanced Functions
def enhance_to_8k(img):
    width = int(img.shape[1] * 2)
    height = int(img.shape[0] * 2)
    upscaled = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(upscaled, -1, kernel)

def apply_face_wash(img):
    smooth = cv2.bilateralFilter(img, 25, 80, 80)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, 25) 
    s = cv2.add(s, 10) 
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def apply_natural_night_vision(img):
    img_dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-5)
    b, g, r = cv2.split(img_dark)
    g = cv2.add(g, 45) # Natural green tint
    r = cv2.subtract(r, 20)
    return cv2.merge((b, g, r))

# 3. Custom Styling
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 10px; font-weight: bold; transition: 0.3s; height: 3em; }
    .main-btn button { background: linear-gradient(135deg, #00F2EA, #FF0050); color: white; border: none; height: 4em; font-size: 18px; }
    .main-title { text-align: center; color: #1E1E1E; font-size: 35px !important; font-weight: 900; margin-bottom: 10px; }
    .section-head { background: #f0f2f6; padding: 5px 15px; border-radius: 5px; font-weight: bold; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>💎 Family AI Master Studio</h1>", unsafe_allow_html=True)

img_file = st.file_uploader("تصویر منتخب کریں", type=["jpg", "png", "jpeg"])

if img_file:
    raw_img = Image.open(img_file).convert("RGB")
    original_frame = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
    
    if 'processed_img' not in st.session_state:
        st.session_state.processed_img = original_frame.copy()

    # --- MAIN AI ACTIONS ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="main-btn">', unsafe_allow_html=True)
        if st.button("🧼 FACE WASH (Fresh Glow)"):
            st.session_state.processed_img = apply_face_wash(st.session_state.processed_img)
            st.toast("Skin Smooth & Bright! ✨")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="main-btn">', unsafe_allow_html=True)
        if st.button("🚀 8K ULTRA HD (Ad Quality)"):
            st.session_state.processed_img = enhance_to_8k(st.session_state.processed_img)
            st.success("Upscaled to High Definition!")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- FILTERS ROW ---
    st.markdown('<div class="section-head">Smart Filters</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        if st.button("📱 iPhone"):
            img = st.session_state.processed_img
            img = cv2.convertScaleAbs(img, alpha=1.05, beta=5)
            img[:, :, 2] = np.clip(img[:, :, 2] * 1.1, 0, 255)
            st.session_state.processed_img = img
    with f2:
        if st.button("📸 Oppo"):
            st.session_state.processed_img = cv2.convertScaleAbs(cv2.bilateralFilter(st.session_state.processed_img, 15, 75, 75), alpha=1.1, beta=15)
    with f3:
        if st.button("🌙 Night Vision"):
            st.session_state.processed_img = apply_natural_night_vision(st.session_state.processed_img)
    with f4:
        if st.button("🔄 Reset"):
            st.session_state.processed_img = original_frame.copy()
            st.rerun()

    # --- ADVANCED TOOLS (HAIR & ZOOM) ---
    st.markdown('<div class="section-head">Advanced Coloring & Zoom</div>', unsafe_allow_html=True)
    col_t1, col_t2, col_t3 = st.columns(3)
    
    with col_t1:
        hair_shades = {"Default": None, "Jet Black": [10, 10, 10], "Deep Brown": [30, 50, 90], "Vibrant Red": [20, 20, 200], "Golden": [40, 180, 220], "Pink": [150, 80, 250]}
        h_col = st.selectbox("Hair Color Shade", list(hair_shades.keys()))
    with col_t2:
        h_int = st.slider("Color Intensity", 0.0, 1.0, 0.6)
    with col_t3:
        zoom = st.slider("Zoom Level", 1.0, 3.0, 1.0)

    # Final Processing
    final = st.session_state.processed_img.copy()

    # 1. Apply Zoom
    if zoom > 1.0:
        h, w = final.shape[:2]
        nh, nw = int(h/zoom), int(w/zoom)
        sh, sw = (h-nh)//2, (w-nw)//2
        final = cv2.resize(final[sh:sh+nh, sw:sw+nw], (w, h), interpolation=cv2.INTER_LANCZOS4)

    # 2. Apply Hair Color (Optimized)
    if h_col != "Default":
        hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
        # Target dark areas (usually hair)
        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        mask_3d = np.stack([cv2.GaussianBlur(mask, (15,15), 0)]*3, axis=-1) / 255.0
        final = (final * (1 - mask_3d * h_int) + np.array(hair_shades[h_col]) * (mask_3d * h_int)).astype(np.uint8)

    # 3. Manual Sliders
    st.markdown('<div class="section-head">Manual Finishing</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        bright = st.slider("Final Brightness", 0.5, 2.0, 1.0)
    with m2:
        vibrance = st.slider("Color Vibrance", 1.0, 3.0, 1.2)

    # Final Adjustments
    if vibrance > 1.0:
        hsv_f = cv2.cvtColor(final, cv2.COLOR_BGR2HSV).astype("float32")
        hsv_f[:,:,1] *= vibrance
        final = cv2.cvtColor(np.clip(hsv_f, 0, 255).astype("uint8"), cv2.COLOR_HSV2BGR)
    
    final = cv2.convertScaleAbs(final, alpha=bright, beta=0)

    # --- DISPLAY & DOWNLOAD ---
    st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    is_success, buffer = cv2.imencode(".jpg", final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if is_success:
        st.download_button("📥 DOWNLOAD 8K STUDIO QUALITY IMAGE", buffer.tobytes(), "UltraHD_Photo.jpg", "image/jpeg")

st.info("💡 بہترین رزلٹ کے لیے: پہلے 'Face Wash' کریں، پھر '8K' بٹن دبائیں، اور آخر میں زوم سیٹ کریں۔")
