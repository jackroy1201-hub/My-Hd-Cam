import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import io

# 1. پیج سیٹنگ
st.set_page_config(page_title="Roman HD Studio Pro", page_icon="🎨", layout="wide")

st.markdown("""
<style>
    .main-header {
        text-align: center; padding: 1.5rem;
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 15px; color: white; margin-bottom: 1.5rem;
    }
    .stButton>button { width:100%; border-radius:12px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎨 Roman Studio Pro</h1><p>ہائی ڈیفینیشن (HD) ایڈیٹنگ - بغیر پکسلز خراب کیے</p></div>', unsafe_allow_html=True)

# 2. سائیڈ بار
with st.sidebar:
    st.title("⚙️ ایچ ڈی سیٹنگز")
    quality = st.slider("کوالٹی برقرار رکھیں", 80, 100, 100)
    st.warning("ٹپ: تصویر کو بار بار ری سیٹ کرنے کے بجائے ہسٹری استعمال کریں۔")

col1, col2 = st.columns([1, 2])

with col1:
    pic = st.file_uploader("تصویر اپلوڈ کریں", type=["jpg", "png", "jpeg", "webp"])
    if pic:
        # تصویر کو ہائی کوالٹی میں لوڈ کرنا
        original = Image.open(pic).convert("RGB")
        if "img" not in st.session_state:
            st.session_state.img = original
            st.session_state.original = original

with col2:
    if pic:
        # پریویو کالمز
        p1, p2 = st.columns(2)
        with p1: st.image(st.session_state.original, caption="اصل تصویر", use_container_width=True)
        with p2: st.image(st.session_state.img, caption="ایچ ڈی رزلٹ", use_container_width=True)

        # ایڈیٹنگ ٹیبز
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["💎 HD نکھار", "🎬 پروفیشنل فلٹرز", "👔 ڈریس کلر", "💇 ہیئر کلر", "💄 بیوٹی"])

        # 1. HD Enhancement (بہتر کیا گیا تاکہ تصویر نہ پھٹے)
        with tab1:
            if st.button("✨ اسمارٹ ایچ ڈی نکھار"):
                img_np = np.array(st.session_state.img)
                # لاب کلر اسپیس میں پروسیسنگ تاکہ پکسل نہ پھٹیں
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                enhanced = cv2.merge((cl, a, b))
                final_np = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
                # شارپننگ بغیر شور (Noise) کے
                res = Image.fromarray(final_np)
                st.session_state.img = ImageEnhance.Sharpness(res).enhance(1.2)
                st.rerun()

        # 2. Cinematic (پریمیم فلٹرز)
        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🎥 سینما موڈ"):
                    img = ImageEnhance.Color(st.session_state.img).enhance(1.4)
                    st.session_state.img = ImageEnhance.Contrast(img).enhance(1.1)
                    st.rerun()
                if st.button("🌑 کلاسک بلیک"):
                    st.session_state.img = ImageOps.grayscale(st.session_state.img)
                    st.rerun()
            with c2:
                if st.button("🔆 برائٹ وائٹ"):
                    st.session_state.img = ImageEnhance.Brightness(st.session_state.img).enhance(1.2)
                    st.rerun()
                if st.button("🍂 وارم ٹون"):
                    img_np = np.array(st.session_state.img).astype(np.float32)
                    img_np[:, :, 0] *= 1.1 # Red بڑھائیں
                    img_np[:, :, 2] *= 0.9 # Blue کم کریں
                    st.session_state.img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                    st.rerun()

        # 3. Dress Color (ماسکنگ بہتر کی گئی)
        with tab3:
            d_color = st.color_picker("نیا رنگ منتخب کریں", "#3498db")
            if st.button("👔 کپڑوں کا رنگ بدلیں"):
                rgb = tuple(int(d_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                img_np = np.array(st.session_state.img)
                hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 255, 255]))
                mask = cv2.medianBlur(mask, 7) # ہموار کرنے کے لیے
                mask_3d = np.stack([mask/255.0]*3, axis=-1)
                res_np = (img_np * (1 - mask_3d * 0.5) + np.array(rgb) * (mask_3d * 0.5)).astype(np.uint8)
                st.session_state.img = Image.fromarray(res_np)
                st.rerun()

        # 4. Hair Color
        with tab4:
            h_opt = {"جیٹ بلیک": [20,20,20], "گولڈن": [190,150,50], "بھورا": [100,60,40]}
            choice = st.selectbox("بالوں کا رنگ", list(h_opt.keys()))
            if st.button("💇 رنگ لاگو کریں"):
                img_np = np.array(st.session_state.img)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)[1]
                mask_3d = np.stack([cv2.GaussianBlur(mask, (15,15), 0)/255.0]*3, axis=-1)
                res_np = (img_np * (1 - mask_3d*0.3) + np.array(h_opt[choice])*(mask_3d*0.3)).astype(np.uint8)
                st.session_state.img = Image.fromarray(res_np)
                st.rerun()

        # 5. Beauty (Anti-Blur Smoothing)
        with tab5:
            smooth_val = st.slider("جلد کا نکھار", 0, 20, 10)
            if st.button("💄 فیس ری ٹچ"):
                img_np = np.array(st.session_state.img)
                # بیلیٹرل فلٹر جو کناروں کو محفوظ رکھتا ہے اور تصویر نہیں پھٹتی
                clean = cv2.bilateralFilter(img_np, smooth_val, 75, 75)
                st.session_state.img = Image.fromarray(clean)
                st.rerun()

        # ڈاؤنلوڈ سیکشن
        st.markdown("---")
        d1, d2 = st.columns(2)
        with d1:
            buf = io.BytesIO()
            # ہائی کوالٹی سیونگ پیرامیٹرز
            st.session_state.img.save(buf, format="JPEG", quality=quality, subsampling=0, qtables="web_high")
            st.download_button("📥 ایچ ڈی تصویر سیو کریں", buf.getvalue(), "Roman_Studio_HD.jpg", "image/jpeg")
        with d2:
            if st.button("🔄 اصل تصویر پر واپس جائیں"):
                st.session_state.img = st.session_state.original
                st.rerun()

st.markdown("<center>Roman Studio Pro - 2026 | No Quality Loss Technology</center>", unsafe_allow_html=True)
