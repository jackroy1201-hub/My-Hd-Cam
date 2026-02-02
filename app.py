import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import io

# --- 1. الٹرا ڈیزائن اور موبائل فرینڈلی بٹنز ---
st.set_page_config(page_title="Roman HD Studio Pro", layout="centered")

st.markdown("""
    <style>
    .stButton > button {
        width: 100%; border-radius: 12px; height: 3.5em;
        font-weight: bold; border: 1px solid #d1d5db;
        transition: 0.3s; margin-bottom: 5px;
    }
    /* AI Auto Bot Button */
    div[data-testid="stVerticalBlock"] > div:nth-child(2) .stButton > button {
        background: linear-gradient(135deg, #FFD700, #FFA500) !important;
        color: black !important; border: none !important; font-size: 1.1em;
    }
    /* Filter Buttons Hover Effect */
    .stButton > button:hover { border-color: #FF4B4B; color: #FF4B4B; background-color: #fffafa; }
    </style>
    """, unsafe_allow_html=True)

st.title("📸 Roman HD Studio Pro")

pic_up = st.file_uploader("تصویر منتخب کریں", type=['jpg', 'png', 'jpeg'])

if pic_up:
    original = Image.open(pic_up).convert('RGB')
    if 'img' not in st.session_state:
        st.session_state.img = original

    # --- 🤖 AI AUTO BOT (Ultra HD & Glow) ---
    if st.button("🤖 AI Auto-Bot: Full HD Glow"):
        img_np = np.array(original)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        smooth = cv2.bilateralFilter(img_bgr, 15, 75, 75)
        gaussian = cv2.GaussianBlur(smooth, (0, 0), 3.0)
        unsharp = cv2.addWeighted(smooth, 1.5, gaussian, -0.5, 0)
        res = Image.fromarray(cv2.cvtColor(unsharp, cv2.COLOR_BGR2RGB))
        res = ImageEnhance.Brightness(res).enhance(1.1)
        res = ImageEnhance.Color(res).enhance(1.3)
        st.session_state.img = ImageEnhance.Sharpness(res).enhance(1.6)
        st.success("ماڈل لک اور ایچ ڈی گلو اپلائی ہو گیا!")

    st.write("### 🎨 سوشل میڈیا فلٹرز")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🌈 Vivid Mode (Bright)"):
            # رنگوں کو گہرا اور وائبرینٹ بنانا
            img = ImageEnhance.Color(original).enhance(1.8)
            st.session_state.img = ImageEnhance.Contrast(img).enhance(1.2)
        
        if st.button("🎵 TikTok Soft Glow"):
            # چہرے پر ملائم چمک اور سافٹ لائٹ
            img_np = np.array(original)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            dst = cv2.detailEnhance(img_bgr, sigma_s=10, sigma_r=0.15)
            soft = cv2.GaussianBlur(dst, (5, 5), 0)
            st.session_state.img = Image.fromarray(cv2.cvtColor(soft, cv2.COLOR_BGR2RGB))

        if st.button("🌟 Model Look"):
            img = ImageEnhance.Color(original).enhance(1.4)
            st.session_state.img = ImageEnhance.Brightness(img).enhance(1.1)

    with col2:
        if st.button("👻 Snapchat Clear"):
            # تصویر کو بالکل صاف اور شارپ کرنا
            img = ImageOps.autocontrast(original)
            st.session_state.img = ImageEnhance.Sharpness(img).enhance(2.0)

        if st.button("🍏 iPhone HD Cam"):
            img = ImageEnhance.Sharpness(original).enhance(2.2)
            st.session_state.img = ImageOps.autocontrast(img)

        if st.button("🎭 Dramatic"):
            st.session_state.img = ImageEnhance.Contrast(original).enhance(1.7)

    # --- 📷 مینول بلر سلائیڈر ---
    st.write("### 📷 Manual DSLR Blur")
    blur_val = st.slider("بلر کی مقدار (بیک گراؤنڈ)", 0, 20, 0)
    if blur_val > 0:
        img_np = np.array(original)
        h, w, _ = img_np.shape
        blurred_img = cv2.GaussianBlur(img_np, (blur_val*2+1, blur_val*2+1), 0)
        mask = np.zeros((h, w), dtype=np.uint8)
        # مرکز میں ماسک تاکہ چہرہ صاف رہے
        cv2.circle(mask, (w//2, h//2-50), min(w, h)//3, 255, -1)
        mask_3d = cv2.cvtColor(cv2.GaussianBlur(mask, (101, 101), 0), cv2.COLOR_GRAY2RGB) / 255.0
        final = (img_np * mask_3d + blurred_img * (1 - mask_3d)).astype(np.uint8)
        st.session_state.img = Image.fromarray(final)

    # ڈسپلے
    st.image(st.session_state.img, use_container_width=True)

    # ایکشن بٹنز
    buf = io.BytesIO()
    st.session_state.img.save(buf, format="JPEG", quality=100, subsampling=0)
    st.download_button("📥 Save HD Photo", buf.getvalue(), "Roman_Studio.jpg", "image/jpeg")
    
    if st.button("🔄 Reset"):
        st.session_state.img = original
        st.rerun()
else:
    st.info("اوپر بٹن سے تصویر اپ لوڈ کریں تاکہ ایڈیٹنگ شروع ہو سکے۔")
