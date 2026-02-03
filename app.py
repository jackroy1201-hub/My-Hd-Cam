import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import io

# 1. پیج سیٹنگ اور اسٹائلنگ
st.set_page_config(page_title="Roman Studio Pro", layout="wide", page_icon="🎨")

st.markdown("""
<style>
    .main-header { text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #1e3c72, #2a5298); border-radius: 15px; color: white; margin-bottom: 20px; }
    .stButton>button { width:100%; border-radius:12px; font-weight:bold; height: 3.5em; transition: 0.3s; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎨 Roman Studio Pro</h1><p>مکمل ایچ ڈی ایڈیٹنگ اسٹوڈیو</p></div>', unsafe_allow_html=True)

# 2. سیفٹی فنکشن (ایرر سے بچنے کے لیے)
def get_safe_numpy(pil_img):
    return np.array(pil_img.convert("RGB"))

# 3. سائیڈ بار
with st.sidebar:
    st.title("⚙️ کنٹرول پینل")
    quality = st.slider("ڈاؤنلوڈ کوالٹی", 80, 100, 95)
    st.info("Roman Studio: آپ کا ڈیٹا مکمل محفوظ ہے۔")

col1, col2 = st.columns([1, 2])

with col1:
    pic = st.file_uploader("تصویر اپلوڈ کریں", type=["jpg", "png", "jpeg", "webp"])
    if pic:
        original = Image.open(pic).convert("RGB")
        if "img" not in st.session_state:
            st.session_state.img = original
            st.session_state.original = original

with col2:
    if pic:
        # موازنہ پریویو
        p1, p2 = st.columns(2)
        with p1: st.image(st.session_state.original, caption="Before (اصل)", use_container_width=True)
        with p2: st.image(st.session_state.img, caption="After (ایڈیٹ شدہ)", use_container_width=True)

        # 4. تمام فیچرز کے ٹیبز
        tabs = st.tabs(["✨ AI میجک", "👔 ڈریس کلر", "💇 ہیئر کلر", "💄 بیوٹی", "🎬 فلٹرز"])

        # --- AI میجک ---
        with tabs[0]:
            if st.button("🚀 اسمارٹ HD نکھار لاگو کریں"):
                img_np = get_safe_numpy(st.session_state.img)
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
                res_np = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
                st.session_state.img = Image.fromarray(res_np)
                st.rerun()

        # --- ڈریس کلر ---
        with tabs[1]:
            d_color = st.color_picker("کپڑوں کا نیا رنگ چنیں", "#3498db")
            d_intensity = st.slider("رنگ کی شدت (Dress)", 0.0, 1.0, 0.5)
            if st.button("👔 ڈریس کلر تبدیل کریں"):
                rgb = tuple(int(d_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                img_np = get_safe_numpy(st.session_state.img)
                hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 255, 255]))
                mask_3d = np.stack([cv2.GaussianBlur(mask, (15, 15), 0)/255.0]*3, axis=-1)
                res_np = (img_np * (1 - mask_3d * d_intensity) + np.array(rgb) * (mask_3d * d_intensity)).astype(np.uint8)
                st.session_state.img = Image.fromarray(res_np)
                st.rerun()

        # --- ہیئر کلر ---
        with tabs[2]:
            h_opt = {"جیٹ بلیک": [20,20,20], "سنہرا (Gold)": [190,150,50], "بھورا (Brown)": [100,60,40], "سرخ": [180,40,40]}
            h_choice = st.selectbox("بالوں کا رنگ منتخب کریں", list(h_opt.keys()))
            h_int = st.slider("رنگ کی شدت (Hair)", 0.1, 1.0, 0.4)
            if st.button("💇 ہیئر کلر لاگو کریں"):
                img_np = get_safe_numpy(st.session_state.img)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                mask = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)[1]
                mask_3d = np.stack([cv2.GaussianBlur(mask, (21, 21), 0)/255.0]*3, axis=-1)
                res_np = (img_np * (1 - mask_3d * h_int) + np.array(h_opt[h_choice]) * (mask_3d * h_int)).astype(np.uint8)
                st.session_state.img = Image.fromarray(res_np)
                st.rerun()

        # --- بیوٹی ٹچ اپ ---
        with tabs[3]:
            smooth = st.slider("جلد کا نکھار", 0, 25, 10)
            bright = st.slider("چہرے کی چمک", 0.5, 2.0, 1.0)
            if st.button("💄 بیوٹی ٹچ اپ لاگو کریں"):
                img_np = get_safe_numpy(st.session_state.img)
                img_np = cv2.bilateralFilter(img_np, smooth, 75, 75)
                res = Image.fromarray(img_np)
                st.session_state.img = ImageEnhance.Brightness(res).enhance(bright)
                st.rerun()

        # --- سنیماٹک فلٹرز ---
        with tabs[4]:
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                if st.button("🖤 کلاسک Noir (B&W)"):
                    st.session_state.img = ImageOps.grayscale(st.session_state.img)
                    st.rerun()
                if st.button("🌅 سنہری رنگ (Golden Hour)"):
                    st.session_state.img = ImageEnhance.Color(st.session_state.img).enhance(1.6)
                    st.rerun()
            with f_col2:
                if st.button("🌈 شوخ رنگ (Vivid)"):
                    st.session_state.img = ImageEnhance.Color(st.session_state.img).enhance(1.5)
                    st.rerun()
                if st.button("📜 پرانا انداز (Retro)"):
                    img_np = get_safe_numpy(st.session_state.img)
                    sepia = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
                    st.session_state.img = Image.fromarray(np.clip(cv2.transform(img_np, sepia), 0, 255).astype(np.uint8))
                    st.rerun()

        # 5. ڈاؤنلوڈ اور ری سیٹ
        st.markdown("---")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            buf = io.BytesIO()
            st.session_state.img.save(buf, format="JPEG", quality=quality, subsampling=0)
            st.download_button("📥 ایچ ڈی تصویر سیو کریں", buf.getvalue(), "Roman_Studio_Final.jpg", "image/jpeg")
        with d_col2:
            if st.button("🔄 تصویر اصل حالت میں لائیں"):
                st.session_state.img = st.session_state.original
                st.rerun()

st.markdown("<center><p style='color:gray;'>Roman Studio Pro - 2026<br>آپ کی ضرورت کا بہترین اسٹوڈیو</p></center>", unsafe_allow_html=True)
