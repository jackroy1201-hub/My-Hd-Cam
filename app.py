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
    .stButton>button { width:100%; border-radius:12px; font-weight:bold; height: 3.5em; transition: 0.3s; margin-bottom: 5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎨 Roman Studio Pro</h1><p>الٹرا ایچ ڈی سوشل میڈیا ایڈیٹنگ اسٹوڈیو</p></div>', unsafe_allow_html=True)

# 2. سیفٹی اور کوالٹی فنکشنز
def get_safe_numpy(pil_img):
    return np.array(pil_img.convert("RGB"))

def apply_sharpness(pil_img, factor=1.5):
    enhancer = ImageEnhance.Sharpness(pil_img)
    return enhancer.enhance(factor)

# 3. سائیڈ بار
with st.sidebar:
    st.title("⚙️ کنٹرول پینل")
    quality_slider = st.slider("ایکسپورٹ کوالٹی (HD)", 80, 100, 100)
    st.markdown("---")
    st.write("✅ تمام فیچرز ایکٹیو ہیں")
    st.write("✅ کوالٹی پروٹیکشن آن ہے")

col1, col2 = st.columns([1, 2])

with col1:
    pic = st.file_uploader("تصویر یہاں اپلوڈ کریں", type=["jpg", "png", "jpeg", "webp"])
    if pic:
        original = Image.open(pic).convert("RGB")
        if "img" not in st.session_state:
            st.session_state.img = original
            st.session_state.original = original
        st.info("💡 آپ کی تصویر Roman Studio میں محفوظ ہے۔")

with col2:
    if pic:
        # موازنہ پریویو (موبائل فرینڈلی)
        p1, p2 = st.columns(2)
        with p1: st.image(st.session_state.original, caption="اصل تصویر", use_container_width=True)
        with p2: st.image(st.session_state.img, caption="ایڈیٹ شدہ (HD)", use_container_width=True)

        # 4. تمام فیچرز کے ٹیبز (Old + New)
        tabs = st.tabs(["✨ AI میجک", "💄 بیوٹی & نکھار", "👔 ڈریس & ہیئر", "🎬 سوشل میڈیا", "🎞️ پروفیشنل"])

        # --- AI میجک (HDR + Sharpness) ---
        with tabs[0]:
            if st.button("🚀 الٹرا HD نکھار (Ultra HD)"):
                img_np = get_safe_numpy(st.session_state.img)
                # Detail Enhancement
                dst = cv2.detailEnhance(img_np, sigma_s=12, sigma_r=0.15)
                res = Image.fromarray(dst)
                st.session_state.img = apply_sharpness(res)
                st.rerun()
            if st.button("🌟 HDR Mode"):
                img_np = get_safe_numpy(st.session_state.img)
                res_np = cv2.detailEnhance(img_np, sigma_s=20, sigma_r=0.20)
                st.session_state.img = Image.fromarray(res_np)
                st.rerun()

        # --- بیوٹی ٹچ اپ ---
        with tabs[1]:
            smooth = st.slider("جلد کی صفائی (Smoothing)", 0, 25, 10)
            bright = st.slider("چہرے کی چمک (Brightness)", 0.5, 2.0, 1.0)
            if st.button("💄 بیوٹی ٹچ اپ لاگو کریں"):
                img_np = get_safe_numpy(st.session_state.img)
                clean_np = cv2.bilateralFilter(img_np, smooth, 75, 75)
                res = Image.fromarray(clean_np)
                st.session_state.img = ImageEnhance.Brightness(res).enhance(bright)
                st.rerun()

        # --- ڈریس اور ہیئر کلر ---
        with tabs[2]:
            c_d, c_h = st.columns(2)
            with c_d:
                d_color = st.color_picker("کپڑوں کا رنگ", "#3498db")
                if st.button("👔 ڈریس کلر بدلیں"):
                    rgb = tuple(int(d_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    img_np = get_safe_numpy(st.session_state.img)
                    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                    mask = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 255, 255]))
                    mask_3d = np.stack([cv2.GaussianBlur(mask, (15, 15), 0)/255.0]*3, axis=-1)
                    res_np = (img_np * (1 - mask_3d * 0.45) + np.array(rgb) * (mask_3d * 0.45)).astype(np.uint8)
                    st.session_state.img = Image.fromarray(res_np)
                    st.rerun()
            with c_h:
                h_opt = {"Black": [20,20,20], "Gold": [190,150,50], "Brown": [100,60,40]}
                choice = st.selectbox("بالوں کا رنگ", list(h_opt.keys()))
                if st.button("💇 ہیئر کلر بدلیں"):
                    img_np = get_safe_numpy(st.session_state.img)
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    mask = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)[1]
                    mask_3d = np.stack([cv2.GaussianBlur(mask, (21,21), 0)/255.0]*3, axis=-1)
                    res_np = (img_np * (1 - mask_3d * 0.4) + np.array(h_opt[choice]) * (mask_3d * 0.4)).astype(np.uint8)
                    st.session_state.img = Image.fromarray(res_np)
                    st.rerun()

        # --- سوشل میڈیا فلٹرز (New Buttons) ---
        with tabs[3]:
            s1, s2 = st.columns(2)
            with s1:
                if st.button("📱 iPhone Cam"):
                    img = st.session_state.img
                    img = ImageEnhance.Color(img).enhance(1.15)
                    img = ImageEnhance.Sharpness(img).enhance(1.6)
                    st.session_state.img = ImageEnhance.Contrast(img).enhance(1.08)
                    st.rerun()
                if st.button("✨ TikTok Glow"):
                    img_np = get_safe_numpy(st.session_state.img)
                    glow = cv2.GaussianBlur(img_np, (25, 25), 0)
                    st.session_state.img = Image.fromarray(cv2.addWeighted(img_np, 0.75, glow, 0.25, 0))
                    st.rerun()
                if st.button("📸 Snapchat Filter"):
                    img = st.session_state.img
                    img = ImageEnhance.Brightness(img).enhance(1.1)
                    st.session_state.img = ImageEnhance.Color(img).enhance(1.25)
                    st.rerun()
            with s2:
                if st.button("📸 Insta Filter"):
                    img = st.session_state.img
                    img = ImageEnhance.Contrast(img).enhance(1.2)
                    st.session_state.img = ImageEnhance.Color(img).enhance(1.3)
                    st.rerun()
                if st.button("🎭 Dramatic"):
                    img = st.session_state.img
                    st.session_state.img = ImageEnhance.Contrast(img).enhance(1.6)
                    st.rerun()
                if st.button("🎬 Cinema Mode"):
                    img_np = get_safe_numpy(st.session_state.img).astype(float)
                    img_np[:,:,0] *= 0.85 # Teal effect
                    img_np[:,:,2] *= 1.1 # Orange/Warm effect
                    st.session_state.img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                    st.rerun()

        # --- پروفیشنل اور کلاسک ---
        with tabs[4]:
            if st.button("🖤 کلاسک Noir (B&W)"):
                st.session_state.img = ImageOps.grayscale(st.session_state.img)
                st.rerun()
            if st.button("🌅 Golden Hour"):
                st.session_state.img = ImageEnhance.Color(st.session_state.img).enhance(1.7)
                st.rerun()

        # 5. ڈاؤنلوڈ اور ری سیٹ (High Quality)
        st.markdown("---")
        d1, d2 = st.columns(2)
        with d1:
            # سیونگ کے وقت کوالٹی کو بہترین بنانا
            final_img = st.session_state.img
            buf = io.BytesIO()
            final_img.save(buf, format="JPEG", quality=quality_slider, subsampling=0)
            st.download_button("📥 ایچ ڈی تصویر محفوظ کریں", buf.getvalue(), "Roman_Studio_Final.jpg", "image/jpeg")
        with d2:
            if st.button("🔄 تصویر اصل حالت میں لائیں"):
                st.session_state.img = st.session_state.original
                st.rerun()

st.markdown("<center><p style='color:gray;'>Roman Studio Pro - 2026<br>All Features Active: HD, Social, Beauty & More</p></center>", unsafe_allow_html=True)
