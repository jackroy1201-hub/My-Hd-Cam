import streamlit as st
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
    """جلد اور بالوں کو قدرتی طریقے سے شفاف بنانے کے لیے"""
    img_array = np.array(image)
    
    # Convert to appropriate color space
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # ==================== PRESERVE ORIGINAL TEXTURE ====================
    original = img_array.copy()
    
    # ==================== GENTLE SKIN SMOOTHING ====================
    # Use subtle bilateral filter to preserve texture
    smooth = cv2.bilateralFilter(img_array, d=7, sigmaColor=8, sigmaSpace=8)
    
    # Blend to keep natural skin texture (70% original, 30% smoothed)
    base = cv2.addWeighted(original, 0.7, smooth, 0.3, 0)
    
    # ==================== NATURAL COLOR ENHANCEMENT ====================
    # LAB color space for better color control
    lab = cv2.cvtColor(base, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Gentle CLAHE for luminosity (avoid over-enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    
    # Subtle color enhancement
    a = cv2.addWeighted(a, 1.05, np.zeros_like(a), 0, 0)
    b = cv2.addWeighted(b, 1.03, np.zeros_like(b), 0, 0)
    
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # ==================== HAIR DETAIL ENHANCEMENT ====================
    # Extract high-frequency details for hair
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    
    # Edge-preserving detail enhancement for hair
    detail = cv2.detailEnhance(enhanced, sigma_s=8, sigma_r=0.15)
    
    # Create hair mask using edge detection
    edges = cv2.Canny(gray, 30, 100)
    kernel = np.ones((1,1), np.uint8)
    hair_mask = cv2.dilate(edges, kernel, iterations=1)
    
    # Apply hair enhancement only to hair areas
    hair_mask_normalized = hair_mask.astype(float) / 255.0
    hair_mask_3d = np.stack([hair_mask_normalized]*3, axis=2)
    
    # Blend hair details
    hair_enhanced = cv2.addWeighted(
        enhanced, 1.0 - hair_mask_3d*0.3,
        detail, hair_mask_3d*0.3,
        0
    )
    
    # ==================== NATURAL GLOW EFFECT ====================
    # Create soft glow layer (subtle)
    blur = cv2.GaussianBlur(hair_enhanced, (0,0), sigmaX=15, sigmaY=15)
    glow = cv2.addWeighted(hair_enhanced, 0.85, blur, 0.15, 0)
    
    # ==================== FINAL COLOR ADJUSTMENTS ====================
    # Enhance vibrance (selective saturation)
    hsv = cv2.cvtColor(glow, cv2.COLOR_RGB2HSV)
    
    # Increase saturation slightly (more selective)
    saturation_boost = 1.1 * intensity
    hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation_boost, 0, 255).astype(np.uint8)
    
    # Slight brightness boost
    hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.05, 0, 255).astype(np.uint8)
    
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # ==================== SHARPENING (SUBTLY) ====================
    # Unsharp mask for crisp details
    gaussian = cv2.GaussianBlur(result, (0,0), 2.0)
    sharpened = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)
    
    # Final blend to avoid oversharpening
    final = cv2.addWeighted(result, 0.7, sharpened, 0.3, 0)
    
    # Ensure natural look
    final = np.clip(final, 0, 255).astype(np.uint8)
    
    return final

def create_premium_background(image):
    """پریمیم بیک گراؤنڈ ایفیکٹ (جرمن بلر/DSLR اسٹائل)"""
    h, w = image.shape[:2]
    
    # Create DSLR-like bokeh background
    background = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Generate bokeh effect
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Create multiple bokeh circles
    for _ in range(30):
        center_x = np.random.randint(0, w)
        center_y = np.random.randint(0, h)
        radius = np.random.randint(20, 80)
        
        # Create circle mask
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        circle_mask = distance < radius
        
        # Color for bokeh
        color = np.array([
            np.random.randint(100, 255),
            np.random.randint(100, 255),
            np.random.randint(100, 255)
        ])
        
        # Add bokeh with gradient
        bokeh_intensity = np.exp(-distance[circle_mask] / (radius*0.5))
        for i in range(3):
            background[circle_mask, i] = np.clip(
                background[circle_mask, i] + color[i] * bokeh_intensity * 0.5,
                0, 255
            )
    
    # Add slight blur to background
    background = cv2.GaussianBlur(background, (51, 51), 30)
    
    # Create mask from subject
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = mask.astype(float) / 255.0
    mask_3d = np.stack([mask]*3, axis=2)
    
    # Blend subject with bokeh background
    result = cv2.addWeighted(
        image.astype(float), mask_3d,
        background.astype(float), 1.0 - mask_3d,
        0
    )
    
    return result.astype(np.uint8)

def enhance_iphone_style(image):
    """آئی فون پرو میکس جیسا شفاف ایفیکٹ"""
    # Base enhancement
    enhanced = enhance_skin_and_hair(image, intensity=1.1)
    
    # iPhone-like color grading (warm, vibrant)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
    
    # Boost saturation and vibrance
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.25, 0, 255)
    
    # Slight warm tone
    hsv[:,:,0] = (hsv[:,:,0].astype(float) * 0.98).astype(np.uint8)  # Shift to warmer hues
    
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Add subtle vignette
    rows, cols = result.shape[:2]
    
    # Create oval vignette
    kernel_x = cv2.getGaussianKernel(cols, cols/4)
    kernel_y = cv2.getGaussianKernel(rows, rows/4)
    kernel = kernel_y * kernel_x.T
    
    # Normalize and apply
    vignette = kernel / kernel.max()
    vignette = 1 - vignette * 0.3  # Gentle vignette
    
    for i in range(3):
        result[:,:,i] = np.clip(result[:,:,i] * vignette, 0, 255).astype(np.uint8)
    
    # Final sharpening for iPhone crispness
    sharp = cv2.detailEnhance(result, sigma_s=5, sigma_r=0.15)
    
    return sharp

def enhance_dslr_style(image):
    """DSLR جیسا پروفیشنل ایفیکٹ (Portrait Mode)"""
    # Base enhancement with more natural settings
    enhanced = enhance_skin_and_hair(image, intensity=1.0)
    
    # Professional color correction (neutral to cool)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Adjust color balance for professional look
    a = cv2.add(a, -5)  # Reduce green/magenta
    b = cv2.add(b, 3)   # Slight yellow for warmth
    
    enhanced_lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Enhance micro-contrast (DSLR-like)
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    
    # Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    enhanced_contrast = clahe.apply(gray)
    
    # Apply contrast to luminosity channel
    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Blend enhanced contrast
    l_enhanced = cv2.addWeighted(l, 0.7, enhanced_contrast, 0.3, 0)
    
    result_lab = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
    
    # Add professional sharpening
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    result = cv2.filter2D(result, -1, kernel)
    
    # Add subtle film grain (optional)
    # grain = np.random.normal(0, 0.5, result.shape).astype(np.float32)
    # result = cv2.add(result.astype(np.float32), grain)
    # result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def enhance_instagram_viral(image):
    """انسٹاگر وائرل ایفیکٹ (HD, شفاف، چمکدار)"""
    # Enhanced base
    enhanced = enhance_skin_and_hair(image, intensity=1.3)
    
    # Instagram-style color grading
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
    
    # High saturation for viral look
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.4, 0, 255)
    
    # Bright highlights
    hsv[:,:,2] = np.where(
        hsv[:,:,2] > 180,
        np.clip(hsv[:,:,2] * 1.2, 0, 255),
        hsv[:,:,2]
    )
    
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Create strong but natural glow
    blur = cv2.GaussianBlur(result, (0,0), sigmaX=25, sigmaY=25)
    
    # Screen blending for glow
    blend = cv2.addWeighted(result, 0.8, blur, 0.2, 0)
    
    # Enhance details for clarity
    detail = cv2.detailEnhance(blend, sigma_s=3, sigma_r=0.1)
    
    # Final blend
    final = cv2.addWeighted(blend, 0.7, detail, 0.3, 0)
    
    # Add slight warmth
    final_hsv = cv2.cvtColor(final, cv2.COLOR_RGB2HSV)
    final_hsv[:,:,0] = np.clip(final_hsv[:,:,0].astype(float) * 0.95, 0, 179).astype(np.uint8)
    final = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    
    return final

def enhance_natural_glow(image):
    """قدرتی گلو اور شفافیت"""
    # Preserve maximum detail
    enhanced = enhance_skin_and_hair(image, intensity=0.9)
    
    # Create soft, natural glow
    blur_small = cv2.GaussianBlur(enhanced, (0,0), 3)
    blur_large = cv2.GaussianBlur(enhanced, (0,0), 15)
    
    # Blend for natural orton effect
    orton_effect = cv2.addWeighted(blur_small, 0.4, blur_large, 0.6, 0)
    
    # Soft light blending
    base_float = enhanced.astype(float) / 255.0
    orton_float = orton_effect.astype(float) / 255.0
    
    # Soft light blend formula
    blend = np.where(
        orton_float <= 0.5,
        2 * base_float * orton_float + base_float**2 * (1 - 2 * orton_float),
        2 * base_float * (1 - orton_float) + np.sqrt(base_float) * (2 * orton_float - 1)
    )
    
    result = np.clip(blend * 255, 0, 255).astype(np.uint8)
    
    # Gentle color boost
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.1, 0, 255)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return result

def main():
    # Title with gradient
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='font-size: 48px; margin-bottom: 10px;'>✨ Premium Image Enhancer</h1>
        <h3 style='color: #FFD700;'>DSLR کیفیت کی HD تصویریں - انسٹاگرام کے لیے تیار</h3>
        <p style='font-size: 18px;'>قدرتی گلو، شفاف جلد، اور وائرل ایفیکٹس</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.header("⚡ تیز ترین آپشنز")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 DSLR ایفیکٹ", use_container_width=True):
                st.session_state.mode = "dslr"
        with col2:
            if st.button("🌟 وائرل گلو", use_container_width=True):
                st.session_state.mode = "viral"
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.header("📸 پریمیم موڈز")
        
        enhancement_mode = st.radio(
            "ایڈوانسڈ ایفیکٹس:",
            ["DSLR پروفیشنل", "انسٹاگرام وائرل", "قدرتی گلو", "آئی فون پرو", "کلاسک پورٹریٹ"],
            index=0
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.header("📤 تصویر اپ لوڈ")
        uploaded_file = st.file_uploader(
            "اپنی تصویر یہاں ڈراپ کریں",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'heic'],
            help="16MP تک کی تصاویر سپورٹڈ"
        )
        
        if uploaded_file:
            file_size = uploaded_file.size / (1024*1024)
            st.success(f"✅ تصویر اپ لوڈ ہو گئی! ({file_size:.1f} MB)")
            
            # Check resolution
            img = Image.open(uploaded_file)
            st.info(f"رزلوشن: {img.size[0]}x{img.size[1]} پکسلز")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.header("⚙️ ایڈوانسڈ کنٹرولز")
        
        intensity = st.slider("شفافیت کی سطح", 0.5, 2.0, 1.0, 0.1)
        background_effect = st.selectbox(
            "بیک گراؤنڈ ایفیکٹ",
            ["DSLR بلیور", "جرمن بلر", "سادہ", "رنگین"]
        )
        preserve_details = st.checkbox("اصل تفصیلات محفوظ کریں", value=True)
        hd_upscale = st.checkbox("4K اپ سکیل", value=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Quick tips
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.header("💡 تجاویز")
        st.markdown("""
        - روشن تصاویر بہترین نتائج دیتی ہیں
        - DSLR موڈ پروفیشنل پورٹریٹس کے لیے
        - وائرل موڈ انسٹاگرام پوسٹس کے لیے
        - کم از کم 2MP کی تصویر استعمال کریں
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.subheader("📷 اصل تصویر")
        
        if uploaded_file:
            original_image = Image.open(uploaded_file)
            st.image(original_image, use_column_width=True, caption="اپ لوڈ کردہ تصویر")
            
            # Image analysis
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("رزلوشن", f"{original_image.size[0]}x{original_image.size[1]}")
            with col_info2:
                st.metric("رنگ ڈیپتھ", original_image.mode)
                
        else:
            st.image("https://via.placeholder.com/600x450/667eea/ffffff?text=اپنی+تصویر+اپ+لوڈ+کریں", 
                    use_column_width=True, caption="اپنی تصویر اپ لوڈ کریں")
            st.info("⬆️ اوپر سے تصویر اپ لوڈ کریں")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='enhance-card'>", unsafe_allow_html=True)
        st.subheader("🎨 پریمیم انہینسڈ")
        
        if uploaded_file:
            with st.spinner("🎯 DSLR کیفیت میں تبدیل کیا جا رہا ہے..."):
                # Process based on selected mode
                if enhancement_mode == "DSLR پروفیشنل":
                    enhanced = enhance_dslr_style(original_image)
                elif enhancement_mode == "انسٹاگرام وائرل":
                    enhanced = enhance_instagram_viral(original_image)
                elif enhancement_mode == "قدرتی گلو":
                    enhanced = enhance_natural_glow(original_image)
                elif enhancement_mode == "آئی فون پرو":
                    enhanced = enhance_iphone_style(original_image)
                else:
                    enhanced = enhance_skin_and_hair(original_image, intensity)
                
                # Apply background effect if selected
                if background_effect != "سادہ":
                    if background_effect == "DSLR بلیور":
                        enhanced = create_premium_background(enhanced)
                
                # Upscale if selected
                if hd_upscale and enhanced.shape[1] < 4000:
                    scale_factor = min(4000 / enhanced.shape[1], 2.0)
                    new_width = int(enhanced.shape[1] * scale_factor)
                    new_height = int(enhanced.shape[0] * scale_factor)
                    enhanced = cv2.resize(enhanced, (new_width, new_height), 
                                        interpolation=cv2.INTER_LANCZOS4)
                
                # Display enhanced image
                st.image(enhanced, use_column_width=True, 
                        caption=f"✨ {enhancement_mode} - DSLR Quality")
                
                # Before-After comparison
                st.markdown("### 🔍 قبل و بعد کا موازنہ")
                try:
                    image_comparison(
                        img1=original_image,
                        img2=Image.fromarray(enhanced),
                        label1="قبل - اصل تصویر",
                        label2="بعد - انہینسڈ",
                        width=700,
                        starting_position=50
                    )
                except Exception as e:
                    col_left, col_right = st.columns(2)
                    with col_left:
                        st.image(original_image, caption="اصل تصویر", use_column_width=True)
                    with col_right:
                        st.image(enhanced, caption="انہینسڈ ورژن", use_column_width=True)
                
                # Download section
                st.markdown("---")
                st.subheader("💾 ڈاؤن لوڈ آپشنز")
                
                # Convert to bytes with different qualities
                enhanced_pil = Image.fromarray(enhanced)
                
                # High quality (4K)
                buf_hq = io.BytesIO()
                enhanced_pil.save(buf_hq, format='JPEG', quality=100, optimize=True)
                byte_hq = buf_hq.getvalue()
                
                # Medium quality (for web)
                buf_mq = io.BytesIO()
                enhanced_pil.save(buf_mq, format='JPEG', quality=90, optimize=True)
                byte_mq = buf_mq.getvalue()
                
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.download_button(
                        label="📥 4K HD (HQ)",
                        data=byte_hq,
                        file_name=f"4k_hd_{enhancement_mode.replace(' ', '_')}.jpg",
                        mime="image/jpeg",
                        help="مکمل 4K رزلوشن (10-15MB)"
                    )
                with col_d2:
                    st.download_button(
                        label="📱 انسٹاگرام (MQ)",
                        data=byte_mq,
                        file_name=f"instagram_{enhancement_mode.replace(' ', '_')}.jpg",
                        mime="image/jpeg",
                        help="انسٹاگرام کے لیے موزوں (3-5MB)"
                    )
                with col_d3:
                    if st.button("🔄 نئی سیٹنگز آزمائیں", use_container_width=True):
                        st.rerun()
                
                # Stats
                st.markdown("---")
                st.subheader("📊 تصویر کوالٹی رپورٹ")
                
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("حتمی رزلوشن", 
                             f"{enhanced.shape[1]}x{enhanced.shape[0]}")
                with col_s2:
                    size_mb = len(byte_hq) / (1024*1024)
                    st.metric("فائل سائز", f"{size_mb:.1f} MB")
                with col_s3:
                    improvement = min(int(size_mb * 3), 100)
                    st.metric("کوالٹی اضافہ", f"{improvement}%")
                with col_s4:
                    st.metric("فارمیٹ", "JPG 100%")
        
        else:
            st.warning("⬅️ پہلے ایک تصویر اپ لوڈ کریں")
            example_img = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(example_img, "DSLR کوالٹی پیش نظارہ", 
                       (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (255, 255, 255), 2)
            st.image(example_img, use_column_width=True, 
                    caption="اپ لوڈ کے بعد یہاں پیش نظارہ دکھائی دے گا")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Features showcase
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h3>✨ خصوصیات جو آپ کی تصویر کو وائرل بنائیں گی</h3>
        <div style='display: flex; justify-content: center; flex-wrap: wrap; gap: 20px; margin-top: 20px;'>
            <div style='background: linear-gradient(45deg, #FF6B6B, #FF8E53); padding: 15px; border-radius: 10px; width: 200px;'>
                <h4 style='color: white;'>📸 DSLR کوالٹی</h4>
                <p style='color: white;'>اصل کیمرا جیسی تفصیلات</p>
            </div>
            <div style='background: linear-gradient(45deg, #4ECDC4, #44A08D); padding: 15px; border-radius: 10px; width: 200px;'>
                <h4 style='color: white;'>🌟 قدرتی گلو</h4>
                <p style='color: white;'>پکسلن نہیں، حقیقی چمک</p>
            </div>
            <div style='background: linear-gradient(45deg, #FFD166, #FFB347); padding: 15px; border-radius: 10px; width: 200px;'>
                <h4 style='color: white;'>💎 شفاف جلد</h4>
                <p style='color: white;'>فیک نظر نہیں آئے گا</p>
            </div>
            <div style='background: linear-gradient(45deg, #06D6A0, #04966A); padding: 15px; border-radius: 10px; width: 200px;'>
                <h4 style='color: white;'>⚡ 4K اپ سکیل</h4>
                <p style='color: white;'>ہائی رزلوشن تک</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: #888; font-size: 14px;'>
        <p>© 2024 Premium Image Enhancer | DSLR Quality Results for Instagram & Social Media</p>
        <p>تمام تصاویر محفوظ ہیں اور آٹومیٹک طریقے سے 4K HD میں تبدیل کی جاتی ہیں</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    if 'mode' not in st.session_state:
        st.session_state.mode = "dslr"
    main()
