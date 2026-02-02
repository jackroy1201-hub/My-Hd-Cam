import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from streamlit_image_comparison import image_comparison

# Page configuration
st.set_page_config(
    page_title="Roman Studio ✨ Premium Enhancer",
    page_icon="🌟",
    layout="wide"
)

# Custom CSS for Roman Studio
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        color: white; font-weight: bold; padding: 12px 24px;
        border-radius: 25px; border: none; transition: all 0.3s;
    }
    .enhance-card {
        background: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin: 15px 0;
        border-left: 5px solid #667eea; color: black;
    }
    h1, h2, h3 { color: white !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
</style>
""", unsafe_allow_html=True)

def enhance_skin_and_hair(image, intensity=1.0):
    """جلد اور بالوں کو بہتر بنانے والا فنکشن - موبائل ایرر فکسڈ اور موثر بنایا گیا"""
    img_array = np.array(image)
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Bilateral filter کو کم پیرامیٹرز کے ساتھ تیز بنایا
    smooth = cv2.bilateralFilter(img_array, d=5, sigmaColor=6, sigmaSpace=6)
    base = cv2.addWeighted(img_array, 0.7, smooth, 0.3, 0)
    
    # CLAHE کو تیز بنایا، tileGridSize کم کیا
    lab = cv2.cvtColor(base, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6,6))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    
    # Hair Enhancement: Canny کو سادہ کیا، کم thresholds
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    detail = cv2.detailEnhance(enhanced, sigma_s=6, sigma_r=0.12)
    edges = cv2.Canny(gray, 25, 80)
    hair_mask = cv2.dilate(edges, np.ones((1,1), np.uint8), iterations=1)
    
    hair_mask_3d = np.stack([hair_mask.astype(np.float32) / 255.0] * 3, axis=2)
    enhanced_f = enhanced.astype(np.float32)
    detail_f = detail.astype(np.float32)
    
    hair_enhanced = (enhanced_f * (1.0 - hair_mask_3d * 0.3)) + (detail_f * (hair_mask_3d * 0.3))
    result = np.clip(hair_enhanced, 0, 255).astype(np.uint8)
    
    # Color Adjustments: HSV کو موثر بنایا
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] *= (1.1 * intensity)
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    hsv[:,:,2] *= 1.05
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Sharpening کو conditional اور تیز kernel کے ساتھ
    if intensity >= 0.8:
        sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        result = cv2.filter2D(result, -1, sharp_kernel)
    
    return result

def create_premium_background(image):
    """پریمیم بیک گراؤنڈ ایفیکٹ (DSLR اسٹائل) - blur کو کم کیا تیز کرنے کے لیے"""
    h, w = image.shape[:2]
    background = cv2.GaussianBlur(image, (41, 41), 25)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), int(min(w,h)*0.4), 255, -1)
    mask_f = cv2.GaussianBlur(mask, (79, 79), 40).astype(np.float32) / 255.0
    mask_3d = np.stack([mask_f]*3, axis=2)
    
    result = (image.astype(np.float32) * mask_3d) + (background.astype(np.float32) * (1.0 - mask_3d))
    return np.clip(result, 0, 255).astype(np.uint8)

def enhance_iphone_style(image):
    res = enhance_skin_and_hair(image, intensity=1.1)
    return cv2.detailEnhance(res, sigma_s=4, sigma_r=0.12)  # کم sigma تیز کرنے کے لیے

def main():
    st.markdown("<h1 style='text-align: center;'>Roman Studio ✨</h1>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.header("📤 اپ لوڈ")
        uploaded_file = st.file_uploader("تصویر منتخب کریں", type=['jpg', 'jpeg', 'png', 'webp'])
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file:
            st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
            mode = st.radio("ایڈوانسڈ ایفیکٹس:", ["DSLR پروفیشنل", "انسٹاگرام وائرل", "قدرتی گلو", "آئی فون پرو"])
            intensity = st.slider("شفافیت کی سطح", 0.5, 2.0, 1.0)
            bg_effect = st.selectbox("بیک گراؤنڈ ایفیکٹ", ["سادہ", "DSLR بلر"])
            hd_upscale = st.checkbox("4K اپ سکیل", value=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        original_image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='enhance-card'><h3>اصل تصویر</h3>", unsafe_allow_html=True)
            st.image(original_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='enhance-card'><h3>Roman Studio ورژن</h3>", unsafe_allow_html=True)
            with st.spinner("🎯 پروسیسنگ..."):
                # Apply processing based on mode
                if mode == "DSLR پروفیشنل":
                    res = enhance_skin_and_hair(original_image, intensity)
                    res = cv2.detailEnhance(res, sigma_s=8, sigma_r=0.12)  # کم sigma
                elif mode == "انسٹاگرام وائرل":
                    res = enhance_skin_and_hair(original_image, 1.3)
                    blur = cv2.GaussianBlur(res, (0,0), 12)  # کم blur
                    res = cv2.addWeighted(res, 0.8, blur, 0.2, 0)
                elif mode == "آئی فون پرو":
                    res = enhance_iphone_style(original_image)
                else:
                    res = enhance_skin_and_hair(original_image, intensity)

                if bg_effect == "DSLR بلر":
                    res = create_premium_background(res)

                if hd_upscale:
                    scale = 2.0
                    h, w = res.shape[:2]
                    res = cv2.resize(res, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
                    
                    sharp_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
                    res = cv2.filter2D(res, -1, sharp_kernel)
                    
                    res = cv2.convertScaleAbs(res, alpha=1.1, beta=8)

                st.image(res, use_container_width=True)
                
                # Comparison feature
                st.markdown("### 🔍 قبل و بعد کا موازنہ")
                image_comparison(img1=original_image, img2=Image.fromarray(res), label1="اصل", label2="ایڈٹ شدہ")
                
                # Download
                buf = io.BytesIO()
                Image.fromarray(res).save(buf, format="JPEG", quality=95)
                st.download_button("📥 ڈاؤن لوڈ HD", buf.getvalue(), "roman_studio.jpg", "image/jpeg")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("← تصویر اپ لوڈ کریں")

if __name__ == "__main__":
    main()
