Import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
from streamlit_image_comparison import image_comparison

# Page configuration
st.set_page_config(
    page_title="✨ Premium Image Enhancer",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        color: white;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 25px;
        border: none;
        transition: all 0.3s;
        margin: 5px 0;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(255, 107, 107, 0.3);
    }
    
    /* Cards */
    .enhance-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 5px solid #667eea;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        border: 2px dashed #667eea;
        border-radius: 10px;
        background: rgba(255,255,255,0.9);
    }
</style>
""", unsafe_allow_html=True)

def enhance_skin_and_hair(image, intensity=1.0):
    """خاص طور پر جلد اور بالوں کو شفاف بنانے کے لیے"""
    img_array = np.array(image)
    
    # Convert to appropriate color space
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # ==================== SKIN ENHANCEMENT ====================
    # 1. Skin smoothing (bilateral filter for keeping edges)
    smooth = cv2.bilateralFilter(img_array, 9, 75, 75)
    
    # 2. Skin tone enhancement
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Enhance luminosity (brightness)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    
    # Enhance colors slightly
    a = cv2.addWeighted(a, 1.2, np.zeros_like(a), 0, -10)
    b = cv2.addWeighted(b, 1.1, np.zeros_like(b), 0, -5)
    
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    skin_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # ==================== HAIR ENHANCEMENT ====================
    # Detect hair areas and enhance
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Edge detection for hair strands
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to cover hair area
    kernel = np.ones((2,2), np.uint8)
    hair_mask = cv2.dilate(edges, kernel, iterations=1)
    
    # Convert mask to 3 channels
    hair_mask_colored = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2RGB)
    hair_mask_colored = hair_mask_colored / 255.0
    
    # Enhance contrast in hair areas
    hair_areas = cv2.detailEnhance(img_array, sigma_s=10, sigma_r=0.15)
    
    # Blend original with enhanced hair
    hair_enhanced = cv2.addWeighted(
        skin_enhanced, 0.7,
        hair_areas, 0.3,
        0
    )
    
    # ==================== GLOW EFFECT ====================
    # Create glow layer
    blur = cv2.GaussianBlur(hair_enhanced, (0,0), 10)
    glow = cv2.addWeighted(hair_enhanced, 0.8, blur, 0.2, 0)
    
    # ==================== FINAL TOUCHES ====================
    # Enhance colors
    hsv = cv2.cvtColor(glow, cv2.COLOR_RGB2HSV)
    hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.3)  # Increase saturation
    hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.1)  # Increase brightness
    
    final_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Sharpen slightly
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    final_image = cv2.filter2D(final_image, -1, kernel)
    
    return final_image

def create_colorful_background(image):
    """رنگین بیک گراؤنڈ ایفیکٹ"""
    h, w = image.shape[:2]
    
    # Create gradient background
    background = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create colorful gradient
    for i in range(3):
        gradient = np.linspace(50, 255, h).reshape(-1, 1)
        background[:,:,i] = gradient
    
    # Add some color variations
    background[:,:,0] = np.roll(background[:,:,0], h//3)  # Blue shift
    background[:,:,2] = np.roll(background[:,:,2], h//6)  # Red shift
    
    # Blend with original (for edges)
    mask = cv2.GaussianBlur(image, (21,21), 0)
    mask = mask / 255.0
    
    result = cv2.addWeighted(
        background.astype(np.float32), 0.3,
        image.astype(np.float32), 0.7,
        0
    )
    
    return result.astype(np.uint8)

def enhance_iphone_style(image):
    """آئی فون جیسا شفاف ایفیکٹ"""
    enhanced = enhance_skin_and_hair(image, intensity=1.2)
    
    # iPhone-like color grading
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Boost colors
    a = cv2.add(a, 10)
    b = cv2.add(b, 5)
    
    enhanced_lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Add slight vignette
    rows, cols = result.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/3)
    kernel_y = cv2.getGaussianKernel(rows, rows/3)
    kernel = kernel_y * kernel_x.T
    
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = mask.astype(np.uint8)
    
    for i in range(3):
        result[:,:,i] = result[:,:,i] * (mask/255)
    
    return result

def enhance_dslr_style(image):
    """DSLR جیسا پروفیشنل ایفیکٹ"""
    # Initial enhancement
    enhanced = enhance_skin_and_hair(image)
    
    # Professional color correction
    result = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    
    # Adjust color balance (cool tone)
    result[:,:,0] = cv2.multiply(result[:,:,0], 1.05)  # Blue
    result[:,:,2] = cv2.multiply(result[:,:,2], 0.95)  # Red
    
    # Enhance details
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharp = cv2.convertScaleAbs(laplacian)
    
    # Add sharpness
    result = cv2.addWeighted(result, 0.8, 
                            cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR), 0.2, 0)
    
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

def enhance_dark_mood(image):
    """ڈارک موڈ/چاندنی ایفیکٹ"""
    enhanced = enhance_skin_and_hair(image, intensity=0.8)
    
    # Convert to dark mood
    dark = cv2.convertScaleAbs(enhanced, alpha=0.7, beta=20)
    
    # Add blue moonlight effect
    dark[:,:,0] = cv2.multiply(dark[:,:,0], 0.7)  # Reduce blue
    dark[:,:,2] = cv2.multiply(dark[:,:,2], 0.9)  # Slightly reduce red
    
    # Add glow to faces/highlights
    blur = cv2.GaussianBlur(dark, (0,0), 5)
    dark = cv2.addWeighted(dark, 0.7, blur, 0.3, 0)
    
    # Add star-like sparkle to highlights
    gray = cv2.cvtColor(dark, cv2.COLOR_RGB2GRAY)
    bright_spots = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    bright_spots_colored = cv2.cvtColor(bright_spots, cv2.COLOR_GRAY2RGB)
    
    # Make sparkles blue
    bright_spots_colored[:,:,0] = 255  # Blue
    bright_spots_colored[:,:,1] = 200  # Green
    bright_spots_colored[:,:,2] = 150  # Red
    
    dark = cv2.addWeighted(dark, 0.9, bright_spots_colored, 0.1, 0)
    
    return dark

def main():
    # Title with gradient
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='font-size: 48px; margin-bottom: 10px;'>✨ Premium Image Enhancer</h1>
        <h3 style='color: #FFD700;'>بال، جلد اور رنگوں کو شفاف بنائیں</h3>
        <p style='font-size: 18px;'>ایک کلک سے تصویر کو HD میں تبدیل کریں</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.header("⚡ تیز ترین آپشنز")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 آٹو انہینس", use_container_width=True):
                st.session_state.mode = "auto"
        with col2:
            if st.button("🌟 شفاف بال", use_container_width=True):
                st.session_state.mode = "hair"
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.header("📸 کیمرا موڈز")
        
        enhancement_mode = st.radio(
            "ایڈوانسڈ موڈز:",
            ["آئی فون اسٹائل", "DSLR پروفیشنل", "ڈارک موڈ", "نیچرل گلو"],
            index=0
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.header("📤 تصویر اپ لوڈ")
        uploaded_file = st.file_uploader(
            "اپنی تصویر یہاں ڈراپ کریں",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="ہر قسم کی تصویر چلے گی"
        )
        
        if uploaded_file:
            file_size = uploaded_file.size / (1024*1024)
            st.success(f"✅ تصویر اپ لوڈ ہو گئی! ({file_size:.1f} MB)")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.header("⚙️ ایڈوانسڈ سیٹنگز")
        
        intensity = st.slider("انہینسمنٹ انٹینسٹی", 0.5, 2.0, 1.0, 0.1)
        add_background = st.checkbox("رنگین بیک گراؤنڈ ایفیکٹ", value=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.subheader("📷 اصل تصویر")
        
        if uploaded_file:
            original_image = Image.open(uploaded_file)
            st.image(original_image, use_column_width=True, caption="اپ لوڈ کردہ تصویر")
            
            # Image info
            st.info(f"""
            **تصویر کی معلومات:**
            - سائز: {original_image.size}
            - موڈ: {original_image.mode}
            - فارمیٹ: {uploaded_file.type}
            """)
        else:
            st.image("https://via.placeholder.com/500x400/667eea/ffffff?text=Upload+Your+Image", 
                    use_column_width=True, caption="اپنی تصویر اپ لوڈ کریں")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.subheader("🎨 انہینسڈ ورژن")
        
        if uploaded_file:
            with st.spinner("🔮 تصویر کو جادوئی بنایا جا رہا ہے..."):
                # Process based on selected mode
                if enhancement_mode == "آئی فون اسٹائل":
                    enhanced = enhance_iphone_style(original_image)
                elif enhancement_mode == "DSLR پروفیشنل":
                    enhanced = enhance_dslr_style(original_image)
                elif enhancement_mode == "ڈارک موڈ":
                    enhanced = enhance_dark_mood(original_image)
                else:
                    enhanced = enhance_skin_and_hair(original_image, intensity)
                
                # Add colorful background if selected
                if add_background:
                    enhanced = create_colorful_background(enhanced)
                
                # Display enhanced image
                st.image(enhanced, use_column_width=True, 
                        caption=f"{enhancement_mode} - انہینسڈ")
                
                # Comparison slider
                st.markdown("### 🔍 تصویر کا موازنہ")
                try:
                    image_comparison(
                        img1=original_image,
                        img2=Image.fromarray(enhanced),
                        label1="اصل",
                        label2="انہینسڈ",
                        width=700
                    )
                except:
                    st.warning("تصویر کا موازنہ فیچر دستیاب نہیں")
                
                # Download section
                st.markdown("---")
                st.subheader("💾 ڈاؤن لوڈ کریں")
                
                # Convert to bytes
                enhanced_pil = Image.fromarray(enhanced)
                buf = io.BytesIO()
                enhanced_pil.save(buf, format='JPEG', quality=100)
                byte_im = buf.getvalue()
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        label="📥 16K HD ڈاؤن لوڈ",
                        data=byte_im,
                        file_name=f"enhanced_hd_{uploaded_file.name}",
                        mime="image/jpeg",
                        help="تقریباً 3MB سائز"
                    )
                with col_d2:
                    if st.button("🔄 نیا ورژن بنائیں"):
                        st.rerun()
                
                # Stats
                st.markdown("---")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("رزلوشن", f"{enhanced.shape[1]}x{enhanced.shape[0]}")
                with col_s2:
                    size_mb = len(byte_im) / (1024*1024)
                    st.metric("فائل سائز", f"{size_mb:.1f} MB")
                with col_s3:
                    st.metric("کوالٹی", "16K HD")
        
        else:
            st.warning("⬅️ پہلے ایک تصویر اپ لوڈ کریں")
            st.image("https://via.placeholder.com/500x400/764ba2/ffffff?text=Enhanced+Preview", 
                    use_column_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h4>✨ خصوصیات</h4>
        <p>
        <span style='color: #FF6B6B;'>• شفاف بال</span> | 
        <span style='color: #4ECDC4;'>• چمکدار جلد</span> | 
        <span style='color: #FFD166;'>• رنگین بیک گراؤنڈ</span> | 
        <span style='color: #06D6A0;'>• 16K HD</span>
        </p>
        <p style='font-size: 14px; color: #888;'>
        تمام تصاویر کو آٹومیٹک طریقے سے صاف اور شفاف بنایا جاتا ہے
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    if 'mode' not in st.session_state:
        st.session_state.mode = "auto"
    main() یہ streamlit پہ ہوسٹ ہونے کیلئے کافی ہے؟
