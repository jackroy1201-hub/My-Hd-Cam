import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# 1. Page Config
st.set_page_config(page_title="Roman Studio - AI Photo Editor Pro", layout="wide")

# --- Image Processing Functions ---

def enhance_to_8k_advanced(img):
    """Advanced 8K enhancement"""
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    denoised = cv2.bilateralFilter(upscaled, 9, 80, 80)
    
    low_pass = cv2.GaussianBlur(denoised, (0, 0), 3.0)
    high_pass = cv2.subtract(denoised, low_pass)
    
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_mask = cv2.GaussianBlur(edges, (5, 5), 0) / 255.0
    edge_mask = np.stack([edge_mask] * 3, axis=2)
    
    sharpened = cv2.addWeighted(denoised, 1.0, high_pass, 0.3 + 0.2 * edge_mask, 0)
    
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    l_final = cv2.addWeighted(l, 0.6, l_enhanced, 0.4, 0)
    final = cv2.cvtColor(cv2.merge((l_final, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(final, 7, 30, 30)

def apply_face_wash_pro(img):
    """Professional Face enhancement"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    skin_mask_soft = cv2.GaussianBlur(skin_mask.astype(np.float32), (21, 21), 0) / 255.0
    skin_mask_soft = np.stack([skin_mask_soft] * 3, axis=2)
    
    skin_smoothed = cv2.bilateralFilter(img, 11, 70, 70)
    lab = cv2.cvtColor(skin_smoothed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.add(a, 3) # Slight pinkish tone for health
    
    skin_final = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    result = skin_final * skin_mask_soft + img * (1 - skin_mask_soft)
    return result.astype(np.uint8)

def apply_hair_color_change(img, color_type="brown", intensity=0.7):
    """AI Hair color change simulation"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    hair_mask = cv2.inRange(l, 10, 120) # Basic hair luma range
    
    hair_mask_soft = cv2.GaussianBlur(hair_mask.astype(np.float32), (15, 15), 0) / 255.0
    hair_mask_soft = np.stack([hair_mask_soft] * 3, axis=2)
    
    color_map = {
        "black": (0, 0, 0.8), "brown": (10, 20, 1.1), 
        "blonde": (-5, 40, 1.3), "burgundy": (40, -10, 1.0)
    }
    
    da, db, dl = color_map.get(color_type, (0,0,1.0))
    a_new = np.clip(a + da, 0, 255).astype(np.uint8)
    b_new = np.clip(b + db, 0, 255).astype(np.uint8)
    l_new = np.clip(l * dl, 0, 255).astype(np.uint8)
    
    colored_img = cv2.cvtColor(cv2.merge((l_new, a_new, b_new)), cv2.COLOR_LAB2BGR)
    result = colored_img * hair_mask_soft + img * (1 - hair_mask_soft)
    return cv2.addWeighted(img, 1 - intensity, result.astype(np.uint8), intensity, 0)

def apply_cinematic_look(img):
    img_float = img.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_float)
    r = r * 1.1
    b = b * 0.9
    result = cv2.merge([b, g, r]) * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_ai_portrait_mode(img):
    blur = cv2.GaussianBlur(img, (25, 25), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0) / 255.0
    mask = np.stack([mask] * 3, axis=2)
    result = img * mask + blur * (1 - mask)
    return result.astype(np.uint8)

# --- UI Styling ---
st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 10px; padding: 10px 20px; transition: 0.3s;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }
    .effect-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px; border-radius: 10px; color: white; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Main App ---
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🎨 Roman Studio: AI Photo Pro</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🖼️ Editor", "🎨 Effects Gallery", "⚙️ Settings"])

with tab1:
    img_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])
    
    if img_file:
        if 'img_original' not in st.session_state or st.session_state.get('last_img') != img_file.name:
            raw_img = Image.open(img_file).convert("RGB")
            st.session_state.img_original = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
            st.session_state.img_processed = st.session_state.img_original.copy()
            st.session_state.img_history = [st.session_state.img_original.copy()]
            st.session_state.last_img = img_file.name

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Original")
            st.image(cv2.cvtColor(st.session_state.img_original, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col_right:
            st.subheader("Processed")
            st.image(cv2.cvtColor(st.session_state.img_processed, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")
        st.subheader("🛠️ AI Tools")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("🚀 8K Enhance", use_container_width=True):
                st.session_state.img_processed = enhance_to_8k_advanced(st.session_state.img_processed)
                st.session_state.img_history.append(st.session_state.img_processed.copy())
                st.rerun()
        with c2:
            if st.button("✨ Face Glow", use_container_width=True):
                st.session_state.img_processed = apply_face_wash_pro(st.session_state.img_processed)
                st.session_state.img_history.append(st.session_state.img_processed.copy())
                st.rerun()
        with c3:
            if st.button("🎬 Cinematic", use_container_width=True):
                st.session_state.img_processed = apply_cinematic_look(st.session_state.img_processed)
                st.session_state.img_history.append(st.session_state.img_processed.copy())
                st.rerun()
        with c4:
            if st.button("📸 Portrait", use_container_width=True):
                st.session_state.img_processed = apply_ai_portrait_mode(st.session_state.img_processed)
                st.session_state.img_history.append(st.session_state.img_processed.copy())
                st.rerun()

        st.subheader("👩‍🦰 Hair Style")
        h1, h2, h3, h4 = st.columns(4)
        hair_styles = [("🟫 Brown", "brown"), ("⬛ Black", "black"), ("👱 Blonde", "blonde"), ("🍷 Burgundy", "burgundy")]
        for i, (label, color) in enumerate(hair_styles):
            with [h1, h2, h3, h4][i]:
                if st.button(label, use_container_width=True):
                    st.session_state.img_processed = apply_hair_color_change(st.session_state.img_processed, color)
                    st.session_state.img_history.append(st.session_state.img_processed.copy())
                    st.rerun()

        st.markdown("---")
        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("↩️ Undo", use_container_width=True):
                if len(st.session_state.img_history) > 1:
                    st.session_state.img_history.pop()
                    st.session_state.img_processed = st.session_state.img_history[-1].copy()
                    st.rerun()
        with a2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.img_processed = st.session_state.img_original.copy()
                st.session_state.img_history = [st.session_state.img_original.copy()]
                st.rerun()
        with a3:
            _, buffer = cv2.imencode(".jpg", st.session_state.img_processed)
            st.download_button("💾 Download Image", buffer.tobytes(), "roman_studio_edit.jpg", "image/jpeg", use_container_width=True)

with tab2:
    st.header("🎨 Effects Gallery")
    st.markdown("""
    - **8K Ultra HD**: تصویر کی ریزولوشن اور ڈیٹیل کو بہتر بناتا ہے۔
    - **Face Glow**: چہرے کی جلد کو صاف اور چمکدار بناتا ہے۔
    - **AI Portrait**: پس منظر کو دھندلا کر کے فوکس چہرے پر لاتا ہے۔
    - **Hair Studio**: بالوں کا رنگ قدرتی انداز میں تبدیل کرتا ہے۔
    """)

with tab3:
    st.header("⚙️ Settings")
    st.checkbox("High Quality Rendering", value=True)
    st.checkbox("Auto-Save History", value=True)
    st.selectbox("Output Format", ["JPG", "PNG"])

# Sidebar
st.sidebar.title("Roman Studio")
st.sidebar.info("آپ کا اپنا پرسنل فوٹو اسٹوڈیو۔ ڈیٹا مکمل محفوظ ہے اور کہیں شیئر نہیں ہوتا۔")
st.sidebar.markdown("---")
st.sidebar.write("Developed for Roman Studio")
