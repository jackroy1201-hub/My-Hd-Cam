import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import io
import tempfile
import os
import time
from moviepy.editor import VideoFileClip

# ---------------- UI Settings ----------------
st.set_page_config(page_title="Roman Studio Pro", layout="wide", page_icon="🎨")

st.markdown("""
<style>
.main-header {
    padding: 1.5rem; text-align: center;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white; border-radius: 15px; margin-bottom: 20px;
}
.stButton>button { width: 100%; border-radius: 12px; font-weight: bold; height: 3.2em; background-color: #2a5298; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎨 Roman Studio Pro</h1><p>Vivid Colors & AI Glow Studio</p></div>', unsafe_allow_html=True)

# ---------------- Advanced Processing Functions ----------------

def apply_custom_filter(img_pil, mode, intensity=10):
    """رنگوں کو تیز کرنے اور گلو دینے کا مین فنکشن"""
    
    if mode == "🌟 Super Glow & Vivid":
        # رنگوں کو انتہائی تیز (Vivid) کرنا
        enhancer = ImageEnhance.Color(img_pil)
        img = enhancer.enhance(2.2)  # High Saturation
        
        # برائٹنس اور کنٹراسٹ بڑھانا
        img = ImageEnhance.Contrast(img).enhance(1.4)
        img = ImageEnhance.Brightness(img).enhance(1.1)
        
        # گلو ایفیکٹ (Blur Overlay)
        glow_layer = img.filter(ImageFilter.GaussianBlur(radius=intensity/2))
        img = Image.blend(img, glow_layer, alpha=0.35) # 35% Glow
        
        # شارپنس
        return ImageEnhance.Sharpness(img).enhance(1.6)

    elif mode == "🎨 Deep HDR Colors":
        img = ImageEnhance.Color(img_pil).enhance(2.5)
        img = ImageEnhance.Contrast(img).enhance(1.5)
        return ImageEnhance.Sharpness(img).enhance(2.0)

    elif mode == "💎 HD Clean":
        n = np.array(img_pil.convert("RGB"))
        den = cv2.fastNlMeansDenoisingColored(n, None, 6, 6, 7, 21)
        blur = cv2.GaussianBlur(den, (0,0), 1)
        hd = cv2.addWeighted(den, 1.1, blur, -0.1, 0)
        return Image.fromarray(hd)
    
    elif mode == "💄 Beauty Face":
        n = np.array(img_pil.convert("RGB"))
        clean = cv2.bilateralFilter(n, int(intensity), 45, 45)
        return Image.fromarray(clean)
    
    elif mode == "🎨 Oil Paint / Cartoon":
        n = np.array(img_pil.convert("RGB"))
        # Cartoon/Oil effect using Stylization
        res = cv2.stylization(n, sigma_s=60, sigma_r=0.07)
        return Image.fromarray(res)

    return img_pil

# ---------------- Sidebar & Tabs ----------------
with st.sidebar:
    st.title("⚙️ Engine Control")
    quality = st.slider("Photo Export Quality", 70, 100, 90)
    st.markdown("---")
    st.warning("Note: بڑی ویڈیوز کے لیے مقامی کمپیوٹر (Local PC) استعمال کریں تاکہ اکاؤنٹ بلاک نہ ہو۔")

tabs = st.tabs(["📸 Photo Editor", "🎥 Video Editor"])

# ================= TAB 1: PHOTO EDITOR =================
with tabs[0]:
    pic = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"], key="p_up")
    if pic:
        img_input = Image.open(pic).convert("RGB")
        col_a, col_b = st.columns(2)
        
        mode_p = st.selectbox("Select Effect", ["None", "🌟 Super Glow & Vivid", "🎨 Deep HDR Colors", "💎 HD Clean", "💄 Beauty Face", "🎨 Oil Paint / Cartoon"])
        p_intensity = st.slider("Effect Intensity", 3, 30, 10, key="p_slider")
        
        processed_img = apply_custom_filter(img_input, mode_p, p_intensity)
        
        col_a.image(img_input, caption="Original", use_container_width=True)
        col_b.image(processed_img, caption=f"Result: {mode_p}", use_container_width=True)
        
        buf = io.BytesIO()
        processed_img.save(buf, "JPEG", quality=quality)
        st.download_button("📥 Download Photo", buf.getvalue(), "Roman_Studio_Photo.jpg")

# ================= TAB 2: VIDEO EDITOR =================
with tabs[1]:
    video_file = st.file_uploader("Upload Video", ["mp4", "mov", "avi"], key="v_up")

    if video_file:
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            v_mode = st.selectbox("Select Video Effect:", ["🌟 Super Glow & Vivid", "🎨 Deep HDR Colors", "💎 HD Clean", "💄 Beauty Face", "🎨 Oil Paint / Cartoon"])
        with v_col2:
            v_intensity = st.slider("Glow / Beauty Intensity", 3, 30, 10, key="v_slider")

        if st.button("🚀 Process & Export Video"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            tfile_path = tfile.name
            tfile.close()
            
            output_final = f"Roman_Studio_Final_{int(time.time())}.mp4"
            
            with st.spinner("AI Engine رنگوں کو گلو دے رہا ہے..."):
                try:
                    cap = cv2.VideoCapture(tfile_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    temp_proc = "processing_raw.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(temp_proc, fourcc, fps, (w, h))

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    bar = st.progress(0)
                    
                    count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        # Process Frame
                        img_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        res_pil = apply_custom_filter(img_frame, v_mode, v_intensity)
                        
                        # Write Frame
                        out.write(cv2.cvtColor(np.array(res_pil), cv2.COLOR_RGB2BGR))
                        
                        count += 1
                        if count % 10 == 0: bar.progress(min(count/total_frames, 1.0))
                    
                    cap.release()
                    out.release()

                    # Re-attach Audio & Fix Memory
                    with VideoFileClip(temp_proc) as processed_clip:
                        with VideoFileClip(tfile_path) as original_clip:
                            final_video = processed_clip.set_audio(original_clip.audio)
                            final_video.write_videofile(output_final, codec="libx264", audio_codec="aac")
                    
                    st.success("ایڈیٹنگ مکمل ہوگئی!")
                    st.video(output_final)
                    
                    with open(output_final, "rb") as f:
                        st.download_button("📥 ڈاؤن لوڈ ویڈیو", f, output_final)
                
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    # Cleanup to save space
                    if os.path.exists(tfile_path): os.remove(tfile_path)
                    if os.path.exists(temp_proc): os.remove(temp_proc)

st.markdown("---")
st.markdown("<center>Roman Studio Pro 2026 | Powered by Roman Studio AI</center>", unsafe_allow_html=True)
