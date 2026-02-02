import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import io

# --- 1. پیج سیٹنگز اور الٹرا ایچ ڈی ڈیزائن ---
st.set_page_config(page_title="Roman Studio Final HD", layout="centered")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.8em; font-weight: bold;
        color: white; border: none; transition: 0.3s; margin-bottom: 8px;
    }
    .btn-autobot button { background: linear-gradient(135deg, #FFD700, #FFA500) !important; color: black !important; border: 2px solid white !important; }
    .btn-iphone button { background: linear-gradient(135deg, #a1c4fd, #c2e9fb) !important; color: #333 !important; }
    .btn-tiktok button { background: linear-gradient(135deg, #000000, #25f4ee) !important; }
    .btn-dslr button { background: linear-gradient(135deg, #ff9a9e, #fecfef) !important; color: #333 !important; }
    .btn-dramatic button { background: linear-gradient(135deg, #667eea, #764ba2) !important; }
    .btn-model button { background: linear-gradient(135deg, #f6d365, #fda085) !important; }
    .btn-dress button { background: linear-gradient(135deg, #434343, #000000) !important; }
    
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #11998e, #38ef7d) !important;
        border-radius: 15px; width: 100%; height: 3.5em; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📸 Roman Studio Final HD")

pic_up = st.file_uploader("تصویر منتخب کریں", type=['jpg', 'png', 'jpeg'])

if pic_up:
    original = Image.open(pic_up).convert('RGB')
    img_np = np.array(original)
    
    if 'img' not in st.session_state:
        st.session_state.img = original

    # --- 🤖 AI AUTO-BOT: ULTRA SHARP HD ---
    st.markdown('<div class="btn-autobot">', unsafe_allow_html=True)
    if st.button("🤖 AI Auto-Bot: Ultra Sharp HD"):
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 1. Noise Reduction (پہلے شور صاف کرنا)
        denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
        
        # 2. Unsharp Masking (تیز ترین ڈیٹیلز کے لیے)
        gaussian_3 = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        unsharp = cv2.addWeighted(denoised, 2.5, gaussian_3, -1.5, 0)
        
        # 3. Contrast Improvement (CLAHE)
        lab = cv2.cvtColor(unsharp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        final_lab = cv2.merge((cl,a,b))
        final_rgb = cv2.cvtColor(final_lab, cv2.COLOR_LAB2RGB)
        
        # 4. Final Polish with PIL
        res = Image.fromarray(final_rgb)
        res = ImageEnhance.Color(res).enhance(1.2) # Natural Colors
        st.session_state.img = res
        st.success("🤖 Ultra Sharp: تصویر اب پہلے سے زیادہ کلئیر اور HD ہے!")
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("### 🛠️ پروفیشنل ایچ ڈی فلٹرز")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="btn-iphone">', unsafe_allow_html=True)
        if st.button("🍏 iPhone HD Cam"):
            enhancer = ImageEnhance.Sharpness(original).enhance(2.5)
            enhancer = ImageEnhance.Contrast(enhancer).enhance(1.2)
            st.session_state.img = ImageOps.autocontrast(enhancer)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="btn-tiktok">', unsafe_allow_html=True)
        if st.button("🎵 Tiktok Beauty"):
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            smooth = cv2.bilateralFilter(img_bgr, 15, 80, 80)
            final = cv2.addWeighted(img_bgr, 0.3, smooth, 0.7, 0)
            st.session_state.img = Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="btn-model">', unsafe_allow_html=True)
        if st.button("🌟 Model Look"):
            enhancer = ImageEnhance.Brightness(original).enhance(1.05)
            enhancer = ImageEnhance.Contrast(enhancer).enhance(1.3)
            st.session_state.img = ImageEnhance.Color(enhancer).enhance(1.4)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="btn-dslr">', unsafe_allow_html=True)
        if st.button("📷 DSLR HD Blur"):
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR).astype(float)
            h, w, _ = img_bgr.shape
            blur = cv2.GaussianBlur(img_bgr, (51, 51), 0)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (w//2, h//2-50), min(w, h)//3, 255, -1)
            mask = cv2.GaussianBlur(mask, (101, 101), 0).astype(float) / 255.0
            final = (img_bgr * mask[:,:,np.newaxis] + blur * (1 - mask[:,:,np.newaxis]))
            st.session_state.img = Image.fromarray(cv2.cvtColor(final.astype(np.uint8), cv2.COLOR_BGR2RGB))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="btn-dramatic">', unsafe_allow_html=True)
        if st.button("🎭 Dramatic Look"):
            img = ImageOps.autocontrast(original, cutoff=2)
            st.session_state.img = ImageEnhance.Color(img).enhance(1.5)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="btn-dress">', unsafe_allow_html=True)
        if st.button("👗 Dress Tone"):
            img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            img_hsv[:,:,1] = cv2.multiply(img_hsv[:,:,1], 1.4)
            st.session_state.img = Image.fromarray(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ڈسپلے اور سیو ---
    st.image(st.session_state.img, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Reset Image"):
            st.session_state.img = original
            st.rerun()
    with c2:
        buf = io.BytesIO()
        # کوالٹی کو 100 پر رکھنا ضروری ہے تاکہ تصویر نہ پھٹے
        st.session_state.img.save(buf, format="JPEG", quality=100, subsampling=0)
        st.download_button("📥 Download Final HD", buf.getvalue(), "Roman_Final_HD.jpg", "image/jpeg")

else:
    st.info("اعلیٰ کوالٹی ایڈیٹنگ کے لیے تصویر اپ لوڈ کریں۔")
