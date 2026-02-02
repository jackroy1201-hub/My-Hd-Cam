import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Roman Studio - Premium Image Enhancer",
    page_icon="🌟",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        color: white; font-weight: bold; border-radius: 25px; border: none;
    }
    .enhance-card {
        background: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin: 15px 0;
        border-left: 5px solid #667eea; color: black;
    }
    h1, h2, h3 { color: white !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
</style>
""", unsafe_allow_html=True)

# ==================== CORE FUNCTIONS ====================

def enhance_skin_and_hair(img_array, intensity=1.0):
    """جلد اور بالوں کو قدرتی طریقے سے شفاف بنانے کے لیے"""
    original = img_array.copy()
    
    # 1. Skin Smoothing
    smooth = cv2.bilateralFilter(img_array, d=7, sigmaColor=8, sigmaSpace=8)
    base = cv2.addWeighted(original, 0.7, smooth, 0.3, 0)
    
    # 2. Color Enhancement (LAB)
    lab = cv2.cvtColor(base, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    
    # 3. Hair Detail Enhancement (FIXED LOGIC)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    detail = cv2.detailEnhance(enhanced, sigma_s=8, sigma_r=0.15)
    edges = cv2.Canny(gray, 30, 100)
    hair_mask = cv2.dilate(edges, np.ones((1,1), np.uint8), iterations=1)
    
    # Normalize mask to 0.0 - 1.0 range
    hair_mask_3d = np.stack([hair_mask.astype(float)/255.0]*3, axis=2)
    
    # Mathematical fix to prevent cv2.error
    enhanced_f = enhanced.astype(float)
    detail_f = detail.astype(float)
    hair_enhanced = (enhanced_f * (1.0 - hair_mask_3d * 0.3)) + (detail_f * (hair_mask_3d * 0.3))
    
    # 4. Final Glow and Color
    result = np.clip(hair_enhanced, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(float)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * (1.1 * intensity), 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.05, 0, 255)
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def create_premium_background(image):
    """DSLR Style Background Blur"""
    h, w = image.shape[:2]
    background = cv2.GaussianBlur(image, (51, 51), 30)
    
    # Create a simple mask for subject (Central focus)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), int(min(w,h)*0.4), 255, -1)
    mask = cv2.GaussianBlur(mask, (99, 99), 50) / 255.0
    mask_3d = np.stack([mask]*3, axis=2)
    
    result = (image.astype(float) * mask_3d) + (background.astype(float) * (1.0 - mask_3d))
    return np.clip(result, 0, 255).astype(np.uint8)

def enhance_iphone_style(image):
    enhanced = enhance_skin_and_hair(image, intensity=1.1)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(float)
    hsv[:,:,1] *= 1.25  # Vibrance
    result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return cv2.detailEnhance(result, sigma_s=5, sigma_r=0.15)

# ==================== MAIN APP ====================

def main():
    st.markdown("<div style='text-align: center;'><h1>Roman Studio ✨</h1><h3>DSLR کیفیت کی HD تصویریں</h3></div>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.header("📤 تصویر اپ لوڈ")
        uploaded_file = st.file_uploader("فائل منتخب کریں", type=['jpg', 'jpeg', 'png', 'webp'])
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file:
            st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
            st.header("📸 پریمیم موڈز")
            mode = st.radio("ایفیکٹس:", ["DSLR پروفیشنل", "انسٹاگرام وائرل", "قدرتی گلو", "آئی فون پرو"])
            
            intensity = st.slider("شفافیت کی سطح", 0.5, 2.0, 1.0)
            bg_blur = st.checkbox("DSLR بیک گراؤنڈ بلر")
            hd_upscale = st.checkbox("4K اپ سکیل (High Res)", value=True)
            st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    if uploaded_file:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with col1:
            st.markdown("<div class='enhance-card'><h3>اصل تصویر</h3>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='enhance-card'><h3>Roman Studio ورژن</h3>", unsafe_allow_html=True)
            with st.spinner("🎯 پروسیسنگ جاری ہے..."):
                # Apply Modes
                if mode == "DSLR پروفیشنل":
                    res = enhance_skin_and_hair(img, intensity)
                    res = cv2.detailEnhance(res, sigma_s=10, sigma_r=0.15)
                elif mode == "انسٹاگرام وائرل":
                    res = enhance_skin_and_hair(img, 1.3)
                    blur = cv2.GaussianBlur(res, (0,0), 15)
                    res = cv2.addWeighted(res, 0.8, blur, 0.2, 0)
                elif mode == "آئی فون پرو":
                    res = enhance_iphone_style(img)
                else:
                    res = enhance_skin_and_hair(img, intensity)

                # Optional Background Blur
                if bg_blur:
                    res = create_premium_background(res)

                # Optional 4K Upscale
                if hd_upscale:
                    width = int(res.shape[1] * 1.5)
                    height = int(res.shape[0] * 1.5)
                    res = cv2.resize(res, (width, height), interpolation=cv2.INTER_LANCZOS4)

                st.image(res, use_container_width=True)
                
                # Download Options
                result_pil = Image.fromarray(res)
                buf = io.BytesIO()
                result_pil.save(buf, format="JPEG", quality=95)
                st.download_button("📥 ڈاؤن لوڈ (High Quality)", buf.getvalue(), "roman_studio_hd.jpg", "image/jpeg")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("← بائیں طرف سے تصویر اپ لوڈ کریں")

if __name__ == "__main__":
    main()
