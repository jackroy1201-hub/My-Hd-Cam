import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

# --- 1. پیج کنفیگریشن (موبائل ویو کے لیے بہترین) ---
st.set_page_config(
    page_title="Family AI Studio Pro",
    page_icon="📸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. ماڈلز لوڈ کرنا (MediaPipe for Portrait Mode) ---
@st.cache_resource
def load_models():
    try:
        import mediapipe.python.solutions.selfie_segmentation as mp_selfie
        return mp_selfie.SelfieSegmentation(model_selection=1)
    except Exception:
        return None

selfie_seg = load_models()

# --- 3. موبائل فرینڈلی ڈیزائن (Custom CSS) ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(135deg, #007bff, #0056b3);
        color: white;
        font-weight: bold;
        border: none;
        margin-top: 10px;
    }
    .main-title {
        text-align: center;
        font-size: 28px !important;
        color: #1E1E1E;
        margin-bottom: 20px;
    }
    .stSelectbox, .stSlider {
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. سیکیورٹی اور لاگ ان سسٹم ---
if 'auth' not in st.session_state: st.session_state.auth = False
user_db = {"Admin": "12@24", "Family": "4590$"}

if not st.session_state.auth:
    st.markdown("<h1 class='main-title'>🔐 Family Secure Login</h1>", unsafe_allow_html=True)
    u = st.text_input("صارف کا نام (Username)")
    p = st.text_input("پاس ورڈ (Password)", type="password")
    if st.button("Unlock Studio"):
        if u in user_db and p == user_db[u]:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("صارف کا نام یا پاس ورڈ غلط ہے")
else:
    # --- مین ایپ انٹرفیس ---
    st.markdown("<h1 class='main-title'>📸 TikTok AI HD Studio</h1>", unsafe_allow_html=True)
    
    # تصویر لینے یا اپلوڈ کرنے کا انتخاب
    source = st.radio("تصویر کہاں سے لیں؟", ["Gallery Upload 📂", "Live Camera 🤳"], horizontal=True)
    
    img_file = None
    if source == "Gallery Upload 📂":
        img_file = st.file_uploader("تصویر منتخب کریں", type=["jpg", "png", "jpeg"])
    else:
        img_file = st.camera_input("کیمرے سے تصویر لیں")

    if img_file:
        # تصویر کو لوڈ کرنا
        raw_img = Image.open(img_file).convert("RGB")
        frame = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
        
        st.write("---")
        st.write("### 🎨 فلٹرز اور ایڈجسٹمنٹ")
        
        col1, col2 = st.columns(2)
        with col1:
            mode = st.selectbox("ٹک ٹاک فلٹرز منتخب کریں:", 
                ["Natural HD", "Portrait Blur (AI)", "Night Vision 🌙", "TikTok Soft Glow", "Anime Cartoon", "Retro Aesthetic"])
            bright = st.slider("چمک (Brightness)", 0.5, 2.0, 1.0)
            
        with col2:
            hair_color = st.selectbox("بالوں کا رنگ بدلیں:", ["None", "Brown", "Golden", "Red", "Purple", "Pink"])
            hair_int = st.slider("رنگ کی شدت (Intensity)", 0.0, 1.0, 0.6)

        # --- پروسیسنگ انجن شروع ---
        processed = frame.copy()

        # 1. Night Vision (پرانا فیچر - CLAHE Lighting)
        if mode == "Night Vision 🌙":
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
            processed = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

        # 2. TikTok Soft Glow (نیا فلٹر)
        elif mode == "TikTok Soft Glow":
            blur = cv2.GaussianBlur(processed, (25, 25), 0)
            processed = cv2.addWeighted(processed, 1.3, blur, 0.4, 0)

        # 3. Anime Cartoon (کارٹون لک)
        elif mode == "Anime Cartoon":
            color = cv2.bilateralFilter(processed, 9, 250, 250)
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            processed = cv2.bitwise_and(color, color, mask=edges)

        # 4. Retro Aesthetic
        elif mode == "Retro Aesthetic":
            processed = cv2.applyColorMap(processed, cv2.COLORMAP_PINK)

        # 5. برائٹنس اور ایچ ڈی نکھار
        processed = cv2.convertScaleAbs(processed, alpha=bright, beta=0)
        if mode == "Natural HD":
            processed = cv2.detailEnhance(processed, sigma_s=10, sigma_r=0.15)

        # 6. بالوں کا رنگ (Advanced Hair Masking)
        if hair_color != "None":
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 90]))
            mask_3d = np.stack([cv2.GaussianBlur(mask, (15,15), 0)]*3, axis=-1) / 255.0
            
            shades = {
                "Brown": [30, 60, 100], 
                "Golden": [50, 190, 230], 
                "Red": [40, 40, 200], 
                "Purple": [130, 0, 130],
                "Pink": [160, 120, 255]
            }
            target = np.array(shades[hair_color], dtype=np.uint8)
            processed = (processed * (1 - mask_3d * hair_int) + target * (mask_3d * hair_int)).astype(np.uint8)

        # 7. Portrait Blur (AI Background Removal)
        if mode == "Portrait Blur (AI)" and selfie_seg:
            rgb_f = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            res = selfie_seg.process(rgb_f)
            if res.segmentation_mask is not None:
                mask = np.stack((res.segmentation_mask,) * 3, axis=-1) > 0.5
                blur_bg = cv2.GaussianBlur(processed, (55, 55), 0)
                # چہرے کو صاف کرنا (Bilateral) اور بیک گراؤنڈ بلر کرنا
                processed = np.where(mask, cv2.bilateralFilter(processed, 9, 75, 75), blur_bg)

        # --- فائنل رزلٹ ڈسپلے ---
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="HD AI Result", use_container_width=True)
        
        # ڈاؤنلوڈ سیکشن
        final_pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        final_pil.save(buf, format="JPEG", quality=100)
        
        st.download_button("📥 Save HD Photo", buf.getvalue(), "Family_Studio_HD.jpg", "image/jpeg")
        
        if st.button("🔒 Logout"):
            st.session_state.auth = False
            st.rerun()
    else:
        st.info("شروع کرنے کے لیے گیلری سے فوٹو اپلوڈ کریں یا کیمرہ استعمال کریں۔")
