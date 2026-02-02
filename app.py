import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Image comparison library check
try:
    from streamlit_image_comparison import image_comparison
    COMPARISON_AVAILABLE = True
except ImportError:
    COMPARISON_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="Roman Studio - Premium Enhancer",
    page_icon="✨",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .stButton>button {
        background: linear-gradient(45deg, #00dbde, #fc00ff);
        color: white; border: none; padding: 10px 20px;
        border-radius: 15px; font-weight: bold; width: 100%;
    }
    .enhance-card {
        background: #1e2130; padding: 20px;
        border-radius: 15px; border: 1px solid #3e4259;
    }
</style>
""", unsafe_allow_html=True)

# --- Core Enhancement Functions ---

def apply_skin_hair_magic(img_array, intensity=1.0):
    """جلد اور بالوں کے لیے ایڈوانسڈ فلٹرز"""
    # Skin Smoothing
    smooth = cv2.bilateralFilter(img_array, 9, 75, 75)
    
    # Hair and Detail Enhancement
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Glow Effect
    blur = cv2.GaussianBlur(result, (0,0), 10)
    result = cv2.addWeighted(result, 0.8, blur, 0.2 * intensity, 0)
    return result

def dslr_effect(img_array):
    """پروفیشنل ڈی ایس ایل آر لک"""
    detail = cv2.detailEnhance(img_array, sigma_s=15, sigma_r=0.15)
    # Cool tone adjustment
    detail[:,:,2] = cv2.multiply(detail[:,:,2], 1.1) # Blue boost
    return detail

def dark_moonlight(img_array):
    """ڈارک موڈ اور چاندنی ایفیکٹ"""
    dark = cv2.convertScaleAbs(img_array, alpha=0.7, beta=-10)
    # Blueish tint
    dark[:,:,2] = cv2.add(dark[:,:,2], 30) 
    return dark

def add_roman_studio_tag(img, text="Roman Studio"):
    """تصویر پر آپ کا برانڈ نام لکھنا"""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = w / 1200
    thickness = int(2 * scale)
    # Bottom right corner
    cv2.putText(img, text, (w - int(300*scale), h - 30), font, scale, (255,255,255), thickness, cv2.LINE_AA)
    return img

# --- Main Interface ---

def main():
    st.markdown("<h1 style='text-align: center; color: #00dbde;'>✨ Roman Studio</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>تصویر کو HD بنائیں اور ڈیٹا محفوظ رکھیں</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("📤 اپ لوڈ اور سیٹنگز")
        uploaded_file = st.file_uploader("تصویر منتخب کریں", type=['jpg', 'png', 'jpeg', 'webp'])
        
        st.divider()
        mode = st.radio("ایڈوانسڈ موڈز", ["iPhone شفاف ایفیکٹ", "DSLR پروفیشنل", "ڈارک موڈ / چاندنی", "نیچرل گلو"])
        intensity = st.slider("انہینسمنٹ شدت", 0.5, 2.0, 1.0)
        
        st.divider()
        watermark = st.checkbox("Roman Studio واٹرمارک لگائیں", value=True)

    if uploaded_file:
        # Load Image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process Image
        with st.spinner("تصویر کو جادوئی بنایا جا رہا ہے..."):
            if mode == "iPhone شفاف ایفیکٹ":
                processed = apply_skin_hair_magic(img, intensity)
            elif mode == "DSLR پروفیشنل":
                processed = dslr_effect(img)
            elif mode == "ڈارک موڈ / چاندنی":
                processed = dark_moonlight(img)
            else:
                processed = apply_skin_hair_magic(img, intensity * 0.5)

            if watermark:
                processed = add_roman_studio_tag(processed)

        # Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📷 اصل تصویر")
            st.image(img, use_container_width=True)
        
        with col2:
            st.markdown("### 🎨 Roman Studio ایڈٹ")
            st.image(processed, use_container_width=True)

        # Comparison Section
        if COMPARISON_AVAILABLE:
            st.divider()
            st.subheader("🔍 موازنہ سلائیڈر")
            image_comparison(
                img1=Image.fromarray(img),
                img2=Image.fromarray(processed),
                label1="Original",
                label2="Roman Studio"
            )

        # Download
        st.divider()
        result_pil = Image.fromarray(processed)
        buf = io.BytesIO()
        result_pil.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="📥 16K HD ڈاؤن لوڈ کریں",
            data=buf.getvalue(),
            file_name=f"RomanStudio_{uploaded_file.name}",
            mime="image/jpeg"
        )
    else:
        st.info("براہ کرم سائیڈ بار سے تصویر اپ لوڈ کریں تاکہ ہم اس پر کام شروع کر سکیں!")

if __name__ == "__main__":
    main()
