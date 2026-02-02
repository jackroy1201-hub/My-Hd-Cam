import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# 1. Page Config
st.set_page_config(page_title="Family AI Master Studio", layout="wide")

# --- Enhanced Functions Area ---

def enhance_to_8k_advanced(img):
    """Advanced 8K enhancement with multi-stage processing"""
    h, w = img.shape[:2]
    
    # Stage 1: Initial upscaling
    upscaled = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    
    # Stage 2: Noise reduction with edge preservation
    denoised = cv2.bilateralFilter(upscaled, 9, 80, 80)
    
    # Stage 3: Smart sharpening
    # Create a sharp mask
    low_pass = cv2.GaussianBlur(denoised, (0, 0), 3.0)
    high_pass = cv2.subtract(denoised, low_pass)
    
    # Adaptive sharpening based on edge strength
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_mask = cv2.GaussianBlur(edges, (5, 5), 0) / 255.0
    edge_mask = np.stack([edge_mask] * 3, axis=2)  # Fixed axis parameter
    
    sharpened = cv2.addWeighted(denoised, 1.0, high_pass, 0.3 + 0.2 * edge_mask, 0)
    
    # Stage 4: Micro-contrast enhancement
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Blend original and enhanced luminance
    l_final = cv2.addWeighted(l, 0.6, l_enhanced, 0.4, 0)
    
    # Stage 5: Color vibrancy boost
    a = cv2.add(a, 5)
    b = cv2.add(b, 3)
    
    final = cv2.cvtColor(cv2.merge((l_final, a, b)), cv2.COLOR_LAB2BGR)
    
    # Stage 6: Final polishing
    final = cv2.bilateralFilter(final, 7, 30, 30)
    
    return final

def apply_face_wash_pro(img):
    """Professional face enhancement with skin detection"""
    # Detect skin areas for targeted enhancement
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Refine skin mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # Gaussian blur for soft edges
    skin_mask_soft = cv2.GaussianBlur(skin_mask.astype(np.float32), (21, 21), 0) / 255.0
    skin_mask_soft = np.stack([skin_mask_soft] * 3, axis=2)  # Fixed axis parameter
    
    # Apply different processing to skin and non-skin areas
    # Process skin areas
    skin_smoothed = cv2.bilateralFilter(img, 11, 70, 70)
    
    # Create natural glow
    glow = cv2.GaussianBlur(skin_smoothed, (0, 0), 2.5)
    skin_glow = cv2.addWeighted(skin_smoothed, 0.8, glow, 0.2, 0)
    
    # Enhance skin tone
    lab = cv2.cvtColor(skin_glow, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Soften highlights on skin
    l_skin = cv2.GaussianBlur(l, (0, 0), 1.5)
    l = cv2.addWeighted(l, 0.7, l_skin, 0.3, 0)
    
    # Slight red boost for healthy look
    a = cv2.add(a, 3)
    
    enhanced_lab = cv2.merge((l, a, b))
    skin_final = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Process non-skin areas (keep details)
    non_skin = cv2.bilateralFilter(img, 5, 25, 25)
    
    # Blend skin and non-skin areas
    result = skin_final * skin_mask_soft + non_skin * (1 - skin_mask_soft)
    
    # Overall enhancement
    result = result.astype(np.uint8)
    result = cv2.bilateralFilter(result, 9, 40, 40)
    
    return result.astype(np.uint8)

def apply_hdr_effect(img, strength=0.3):
    """Apply natural HDR effect"""
    # Multiple exposures simulation
    exposures = []
    for gamma in [0.8, 1.0, 1.2]:
        adjusted = np.power(img.astype(np.float32) / 255.0, gamma) * 255.0
        exposures.append(adjusted)
    
    # Merge exposures
    hdr = np.zeros_like(img, dtype=np.float32)
    for exp in exposures:
        hdr += exp
    hdr = hdr / len(exposures)
    
    # Tone mapping
    tonemapped = np.tanh(hdr / 255.0 * (1 + strength)) * 255.0
    
    # Local contrast enhancement
    tonemapped_uint8 = tonemapped.astype(np.uint8)  # Fixed: Convert to uint8 first
    lab = cv2.cvtColor(tonemapped_uint8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Blend with original
    l_final = cv2.addWeighted(l, 0.6, l_enhanced, 0.4, 0)
    
    final_lab = cv2.merge((l_final, a, b))
    return cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

def apply_cinematic_look(img):
    """Cinematic color grading"""
    # Convert to float
    img_float = img.astype(np.float32) / 255.0
    
    # Create color grading LUT
    # Teal and orange tones (cinematic look)
    b, g, r = cv2.split(img_float)
    
    # Boost orange in highlights
    r = r * 1.1
    g = g * 0.95
    
    # Add teal to shadows
    b_shadow = np.where(b < 0.3, b * 1.05, b)
    b = cv2.addWeighted(b, 0.7, b_shadow, 0.3, 0)
    
    # Merge and convert back
    result = cv2.merge([b, g, r]) * 255.0
    
    # Add film grain
    noise = np.random.normal(0, 0.005, img.shape).astype(np.float32)
    result = result + noise * 255.0
    
    # Vignette effect
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/3)
    kernel_y = cv2.getGaussianKernel(rows, rows/3)
    kernel = kernel_y * kernel_x.T
    
    mask = kernel / kernel.max()
    mask = np.stack([mask] * 3, axis=2)  # Fixed axis parameter
    
    result = result * (0.8 + 0.2 * mask)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_ai_portrait_mode(img):
    """Simulate portrait mode with background blur"""
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to create mask
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Create mask (1 for subject, 0 for background)
    mask = 255 - edges_dilated
    mask = cv2.GaussianBlur(mask, (21, 21), 0) / 255.0
    mask = np.stack([mask] * 3, axis=2)  # Fixed axis parameter
    
    # Blur background
    background_blur = cv2.GaussianBlur(img, (25, 25), 0)
    
    # Blend
    result = img * mask + background_blur * (1 - mask)
    
    # Enhance subject
    subject_enhanced = cv2.convertScaleAbs(img, alpha=1.05, beta=5)
    
    # Final blend
    final = result * mask + subject_enhanced * (1 - mask) * 0.3 + img * (1 - mask) * 0.7
    
    return np.clip(final, 0, 255).astype(np.uint8)

def apply_skin_retouch(img):
    """Professional skin retouching"""
    # Separate color channels
    b, g, r = cv2.split(img)
    
    # Reduce red channel noise (blemishes)
    r_smoothed = cv2.bilateralFilter(r, 13, 50, 50)
    
    # Enhance blue channel for brightness
    b_enhanced = cv2.add(b, 5)
    
    # Recombine
    retouched = cv2.merge([b_enhanced, g, r_smoothed])
    
    # Frequency separation
    # Low frequency (color/texture)
    low_freq = cv2.bilateralFilter(retouched, 15, 80, 80)
    
    # High frequency (details)
    high_freq = cv2.subtract(retouched, low_freq)
    
    # Smooth low frequency more
    low_freq_smooth = cv2.bilateralFilter(low_freq, 21, 100, 100)
    
    # Recombine with preserved details
    result = cv2.add(low_freq_smooth, high_freq * 0.7)
    
    # Final polish
    result = cv2.bilateralFilter(result, 9, 30, 30)
    
    return result.astype(np.uint8)

# --- UI Styling ---
st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #2C3E50; margin-bottom: 30px;'>💎 AI Family Master Studio Pro</h1>", unsafe_allow_html=True)

# File uploader with drag and drop
img_file = st.file_uploader("📤 تصویر اپلوڈ کریں (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if img_file:
    if 'original' not in st.session_state or st.session_state.get('last_file') != img_file.name:
        raw_img = Image.open(img_file).convert("RGB")
        st.session_state.original = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
        st.session_state.processed = st.session_state.original.copy()
        st.session_state.last_file = img_file.name
        st.session_state.applied_8k = False
        st.session_state.history = [st.session_state.original.copy()]

if 'processed' in st.session_state:
    # --- Progress Bar for Processing ---
    progress_bar = st.progress(0)
    
    # --- Professional Tools Section ---
    st.write("### 🎯 پرو فیسشنل ٹولز")
    pro_col1, pro_col2, pro_col3, pro_col4 = st.columns(4)
    
    with pro_col1:
        if st.button("✨ AI Skin Retouch", use_container_width=True):
            progress_bar.progress(30)
            st.session_state.processed = apply_skin_retouch(st.session_state.processed)
            st.session_state.history.append(st.session_state.processed.copy())
            progress_bar.progress(100)
            st.rerun()
    
    with pro_col2:
        if st.button("🎬 Cinematic Look", use_container_width=True):
            progress_bar.progress(30)
            st.session_state.processed = apply_cinematic_look(st.session_state.processed)
            st.session_state.history.append(st.session_state.processed.copy())
            progress_bar.progress(100)
            st.rerun()
    
    with pro_col3:
        if st.button("📸 Portrait Mode", use_container_width=True):
            progress_bar.progress(30)
            st.session_state.processed = apply_ai_portrait_mode(st.session_state.processed)
            st.session_state.history.append(st.session_state.processed.copy())
            progress_bar.progress(100)
            st.rerun()
    
    with pro_col4:
        if st.button("🌟 Natural HDR", use_container_width=True):
            progress_bar.progress(30)
            st.session_state.processed = apply_hdr_effect(st.session_state.processed, 0.3)
            st.session_state.history.append(st.session_state.processed.copy())
            progress_bar.progress(100)
            st.rerun()
    
    # --- Main Tools Section ---
    st.write("### 🛠️ مین ایڈیٹنگ ٹولز")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🧼 Face Glow Pro", use_container_width=True):
            progress_bar.progress(30)
            st.session_state.processed = apply_face_wash_pro(st.session_state.processed)
            st.session_state.history.append(st.session_state.processed.copy())
            progress_bar.progress(100)
            st.rerun()
    
    with col2:
        if st.button("🚀 8K Ultra HD Pro", use_container_width=True):
            progress_bar.progress(30)
            st.session_state.processed = enhance_to_8k_advanced(st.session_state.processed)
            st.session_state.applied_8k = True
            st.session_state.history.append(st.session_state.processed.copy())
            progress_bar.progress(100)
            st.rerun()
    
    with col3:
        if st.button("🔄 Undo Last", use_container_width=True):
            if len(st.session_state.history) > 1:
                st.session_state.history.pop()
                st.session_state.processed = st.session_state.history[-1].copy()
                st.rerun()
    
    with col4:
        if st.button("🗑️ Reset All", use_container_width=True):
            st.session_state.processed = st.session_state.original.copy()
            st.session_state.history = [st.session_state.original.copy()]
            st.session_state.applied_8k = False
            st.rerun()
    
    progress_bar.empty()
    
    # --- Advanced Adjustment Panel ---
    with st.expander("🎚️ ایڈوانسڈ ایڈجسٹمنٹس", expanded=True):
        adj1, adj2, adj3, adj4 = st.columns(4)
        
        with adj1:
            temperature = st.slider("Temperature", -30, 30, 0, 5)
        
        with adj2:
            tint = st.slider("Tint", -30, 30, 0, 5)
        
        with adj3:
            clarity = st.slider("Clarity", -50, 50, 0, 5)
        
        with adj4:
            dehaze = st.slider("Dehaze", 0, 100, 0, 5)
    
    # --- Preset Filters ---
    st.write("### 🎨 پری سیٹ فلٹرز")
    presets = st.columns(5)
    
    preset_configs = {
        "Vibrant": {"contrast": 1.1, "saturation": 1.2, "brightness": 1.05},
        "Portrait": {"contrast": 0.95, "saturation": 0.9, "brightness": 1.1},
        "Dramatic": {"contrast": 1.3, "saturation": 1.1, "brightness": 0.95},
        "Cinematic": {"contrast": 1.15, "saturation": 0.85, "brightness": 0.9},
        "Natural": {"contrast": 1.05, "saturation": 1.0, "brightness": 1.02}
    }
    
    for idx, (preset_name, config) in enumerate(preset_configs.items()):
        with presets[idx]:
            if st.button(preset_name, use_container_width=True):
                # Apply preset
                temp_img = st.session_state.processed.copy()
                
                # Adjust contrast
                lab = cv2.cvtColor(temp_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = np.clip(l.astype(np.float32) * config["contrast"], 0, 255)
                
                # Adjust saturation
                hsv = cv2.cvtColor(temp_img, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] *= config["saturation"]
                hsv[:, :, 2] *= config["brightness"]
                temp_img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
                
                st.session_state.processed = temp_img
                st.session_state.history.append(st.session_state.processed.copy())
                st.rerun()
    
    # --- Final Touch Adjustments ---
    st.write("### ✨ فائنل ٹچ")
    final1, final2, final3 = st.columns(3)
    
    with final1:
        sharpness = st.slider("شارپنس", 0.0, 2.0, 1.0, 0.05)
    
    with final2:
        noise_reduction = st.slider("نوائس ریڈکشن", 0, 100, 30, 5)
    
    with final3:
        vignette_strength = st.slider("وگنیٹ", 0.0, 1.0, 0.0, 0.05)
    
    # --- Apply Final Adjustments ---
    final_img = st.session_state.processed.copy()
    
    # Apply sharpness
    if sharpness != 1.0:
        kernel_size = int(3 + (sharpness - 1.0) * 2)
        kernel_size = max(3, min(kernel_size, 9))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(final_img, (kernel_size, kernel_size), 0)
        final_img = cv2.addWeighted(final_img, sharpness, blurred, 1 - sharpness, 0)
    
    # Apply noise reduction
    if noise_reduction > 0:
        final_img = cv2.bilateralFilter(final_img, 9, noise_reduction, noise_reduction)
    
    # Apply vignette
    if vignette_strength > 0:
        rows, cols = final_img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/2)
        kernel_y = cv2.getGaussianKernel(rows, rows/2)
        kernel = kernel_y * kernel_x.T
        
        mask = 1 - kernel * vignette_strength
        mask = np.stack([mask] * 3, axis=2)  # Fixed axis parameter
        final_img = (final_img * mask).astype(np.uint8)
    
    # Apply temperature and tint
    if temperature != 0 or tint != 0:
        img_float = final_img.astype(np.float32)
        
        # Temperature (blue-yellow balance)
        img_float[:, :, 0] = np.clip(img_float[:, :, 0] + temperature, 0, 255)
        img_float[:, :, 2] = np.clip(img_float[:, :, 2] - temperature * 0.5, 0, 255)
        
        # Tint (green-magenta balance)
        img_float[:, :, 1] = np.clip(img_float[:, :, 1] + tint, 0, 255)
        
        final_img = img_float.astype(np.uint8)
    
    # --- Display Results ---
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("**📤 اصل تصویر**")
        st.image(cv2.cvtColor(st.session_state.original, cv2.COLOR_BGR2RGB), 
                use_container_width=True, caption=f"Size: {st.session_state.original.shape[1]}x{st.session_state.original.shape[0]}")
    
    with col_right:
        st.markdown("**✨ ترمیم شدہ تصویر**")
        st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), 
                use_container_width=True, caption=f"Size: {final_img.shape[1]}x{final_img.shape[0]}")
    
    # --- Download Options ---
    st.write("### 💾 ڈاؤن لوڈ آپشنز")
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    
    with dl_col1:
        # High Quality JPEG
        _, buffer_hq = cv2.imencode(".jpg", final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        st.download_button("📥 HQ JPEG (95%)", 
                          buffer_hq.tobytes(), 
                          "family_portrait_hq.jpg", 
                          "image/jpeg",
                          use_container_width=True)
    
    with dl_col2:
        # Maximum Quality
        _, buffer_max = cv2.imencode(".jpg", final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        st.download_button("💎 Max Quality (100%)", 
                          buffer_max.tobytes(), 
                          "family_portrait_max.jpg", 
                          "image/jpeg",
                          use_container_width=True)
    
    with dl_col3:
        # PNG Format (Lossless)
        _, buffer_png = cv2.imencode(".png", final_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        st.download_button("🔷 PNG (Lossless)", 
                          buffer_png.tobytes(), 
                          "family_portrait.png", 
                          "image/png",
                          use_container_width=True)
    
    # --- Image Info ---
    st.write("### 📊 تصویر کی معلومات")
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.metric("اصل سائز", f"{st.session_state.original.shape[1]}x{st.session_state.original.shape[0]}")
    
    with info_col2:
        st.metric("ترمیم شدہ سائز", f"{final_img.shape[1]}x{final_img.shape[0]}")
    
    with info_col3:
        enhancement = "✅ 8K Enhanced" if st.session_state.get('applied_8k', False) else "⭕ Standard"
        st.metric("Enhancement", enhancement)

# --- Tips and Instructions ---
st.sidebar.title("💡 تجاویز")
st.sidebar.info("""
**بہترین نتائج کے لیے:**

1. **ترتیب:** 
   - پہلے AI Skin Retouch
   - پھر Face Glow Pro
   - آخر میں 8K Ultra HD

2. **فلٹرز:**
   - Cinematic Look: فلمی رنگ
   - Portrait Mode: بیک گراؤنڈ بلر
   - Natural HDR: قدرتی ڈیپتھ

3. **ایڈجسٹمنٹ:**
   - Sharpness: 1.2-1.3
   - Noise Reduction: 30-50
   - Temperature: ذائقہ کے مطابق
""")

st.sidebar.title("⚙️ سیٹنگز")
default_quality = st.sidebar.selectbox(
    "ڈیفالٹ کوالٹی",
    ["High (95%)", "Maximum (100%)", "Balanced (90%)"]
)

auto_enhance = st.sidebar.checkbox("Auto-enhance on upload", value=True)

if st.sidebar.button("Clear Cache"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
