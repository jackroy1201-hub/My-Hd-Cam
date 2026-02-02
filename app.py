import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

# --- 1. پیج کنفیگریشن ---
st.set_page_config(page_title="Family AI Pro Studio", layout="centered")

# --- 2. موبائل فرینڈلی ڈیزائن (CSS) ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em;
        font-weight: bold; transition: 0.3s;
    }
    .auto-btn button {
        background: linear-gradient(135deg, #FF0050, #00f2ea);
        color: white; border: none; font-size: 1.2em;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-title { text-align: center; color: #1E1E1E; font-size: 28px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 3. مین ایپ انٹرفیس ---
st.markdown("<h1 class='main-title'>📸 TikTok AI Photo Studio</h1>", unsafe_allow_html=True)

# تصویر اپلوڈ کرنے کا سیکشن (پرانے کیمرے کی جگہ صرف اپلوڈ)
img_file = st.file_uploader("تصویر اپلوڈ کریں (Gallery)", type=["jpg", "png", "jpeg"])

if img_file:
    raw_img = Image.open(img_file).convert("RGB")
    original_frame = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
    
    # سیشن اسٹیٹ تاکہ ایڈیٹنگ مکس نہ ہو
    if 'processed_img' not in st.session_state:
        st.session_state.processed_img = original_frame.copy()

    # --- [نیو فیچر] AI AUTO BEAUTY بٹن ---
    st.markdown('<div class="auto-btn">', unsafe_allow_html=True)
    if st.button("🪄 AI AUTO BEAUTY & HD (آٹو نکھار)"):
        img = st.session_state.processed_img
        img = cv2.bilateralFilter(img, 12, 80, 80) # اسکن صاف کرنا
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.5).apply(l) # ایچ ڈی لائٹنگ
        img = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
        st.session_state.processed_img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
        st.toast("AI Magic Applied! ✨")
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("---")
    
    # --- [پرانے اور نیو فیچرز] کوئیک فلٹر بٹنز ---
    st.write("### 🎨 تمام فلٹرز (Quick Buttons)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌟 TikTok Glow"): # نیو
            blur = cv2.GaussianBlur(st.session_state.processed_img, (25, 25), 0)
            st.session_state.processed_img = cv2.addWeighted(st.session_state.processed_img, 1.4, blur, 0.3, 0)
        if st.button("🌙 Night Vision"): # پرانا
            lab = cv2.cvtColor(st.session_state.processed_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=4.0).apply(l)
            st.session_state.processed_img = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

    with col2:
        if st.button("🎭 Anime Look"): # نیو
            img = st.session_state.processed_img
            color = cv2.bilateralFilter(img, 9, 250, 250)
            edges = cv2.adaptiveThreshold(cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            st.session_state.processed_img = cv2.bitwise_and(color, color, mask=edges)
        if st.button("☁️ Soft Portrait"): # پورٹریٹ بلر کا متبادل
            h, w = st.session_state.processed_img.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            cv2.circle(mask, (w//2, h//2), min(w,h)//2, 255, -1)
            mask = cv2.GaussianBlur(mask, (101, 101), 0) / 255
            blur_bg = cv2.GaussianBlur(st.session_state.processed_img, (45, 45), 0)
            st.session_state.processed_img = (st.session_state.processed_img * mask[..., None] + blur_bg * (1 - mask[..., None])).astype(np.uint8)

    with col3:
        if st.button("🎞 Retro Aesthetic"): # نیو
            st.session_state.processed_img = cv2.applyColorMap(st.session_state.processed_img, cv2.COLORMAP_PINK)
        if st.button("🔄 Reset (اصلی حالت)"): # پرانا ری سیٹ
            st.session_state.processed_img = original_frame.copy()

    # --- [پرانے مینوئل کنٹرولز] ---
    st.write("---")
    st.write("### ⚙️ مینوئل ایڈجسٹمنٹ اور کلر")
    c_left, c_right = st.columns(2)
    with c_left:
        bright = st.slider("Brightness (روشنی)", 0.5, 2.0, 1.0) # پرانا
        zoom = st.slider("Zoom (زوم کریں)", 1.0, 3.0, 1.0) # پرانا
    
    with c_right:
        # بالوں کا جدید انجن (Solid Color Change)
        hair_shades = {
            "None": None,
            "Jet Black": [10, 10, 10], "Deep Brown": [30, 50, 90],
            "Vibrant Red": [20, 20, 200], "Golden Blonde": [40, 180, 220],
            "Hot Pink": [150, 80, 250], "Neon Blue": [200, 50, 20]
        }
        h_col_name = st.selectbox("بالوں کا نیا رنگ چنیں:", list(hair_shades.keys()))
        h_int = st.slider("Intensity (رنگ کتنا گہرا ہو)", 0.0, 1.0, 0.8)

    # فائنل رینڈرنگ سیکشن (تمام تبدیلیاں ایک ساتھ اپلائی کرنا)
    final_view = st.session_state.processed_img.copy()

    # زوم اپلائی کرنا
    if zoom > 1.0:
        h, w = final_view.shape[:2]
        nh, nw = int(h/zoom), int(w/zoom)
        sh, sw = (h-nh)//2, (w-nw)//2
        final_view = cv2.resize(final_view[sh:sh+nh, sw:sw+nw], (w, h))

    # بالوں کا رنگ (Solid Overlay)
    if h_col_name != "None":
        hsv = cv2.cvtColor(final_view, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 120]))
        mask_3d = np.stack([cv2.GaussianBlur(mask, (15,15), 0)]*3, axis=-1) / 255.0
        target_rgb = np.array(hair_shades[h_col_name], dtype=np.uint8)
        final_view = (final_view * (1 - mask_3d * h_int) + target_rgb * (mask_3d * h_int)).astype(np.uint8)

    # برائٹنس اپلائی کرنا
    final_view = cv2.convertScaleAbs(final_view, alpha=bright, beta=0)

    # رزلٹ دکھانا
    st.image(cv2.cvtColor(final_view, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Final HD Result")
    
    # ڈاؤنلوڈ بٹن (پرانا سیو فیچر)
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(final_view, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG", quality=100)
    st.download_button("📥 Save HD Image (گیلری میں محفوظ کریں)", buf.getvalue(), "Family_AI_Studio.jpg", "image/jpeg")

else:
    st.info("شروع کرنے کے لیے گیلری سے کوئی تصویر اپلوڈ کریں۔")
