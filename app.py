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

def apply_custom_filter(img_pil, mode, smooth_val=10):
    # 1. Super Glow & Vivid (تیز رنگ اور چمک)
    if mode == "🌟 Super Glow & Vivid":
        # Enhance Colors
        img = ImageEnhance.Color(img_pil).enhance(1.8) # رنگ تیز کرنا
        img = ImageEnhance.Contrast(img).enhance(1.2) # کنٹراسٹ
        # Creating Glow Effect
        overlay = img.filter(ImageFilter.GaussianBlur(radius=4))
        img = Image.blend(img, overlay, alpha=0.3)
        return ImageEnhance.Brightness(img).enhance(1.1)

    # 2. Deep Colors (گہرے اور شارپ رنگ)
    elif mode == "🎨 Deep Colors":
        img = ImageEnhance.Color(img_pil).enhance(2.0)
        img = ImageEnhance.Sharpness(img).enhance(2.0)
        return img

    # 3. HD Clean
    elif mode == "💎 HD Clean":
        n = np.array(img_pil.convert("RGB"))
        den = cv2.fastNlMeansDenoisingColored(n, None, 6, 6, 7, 21)
        blur = cv2.GaussianBlur(den, (0,0), 1)
        hd = cv2.addWeighted(den, 1.1, blur, -0.1, 0)
        return Image.fromarray(hd)
    
    # 4. Beauty Face
    elif mode == "💄 Beauty Face":
        n = np.array(img_pil.convert("RGB"))
        clean = cv2.bilateralFilter(n, smooth_val, 40, 40)
        return Image.fromarray(clean)
    
    # 5. iPhone Vibe
    elif mode == "📱 iPhone Vibe":
        img = ImageEnhance.Color(img_pil).enhance(1.3)
        img = ImageEnhance.Brightness(img).enhance(1.05)
        return ImageEnhance.Sharpness(img).enhance(1.5)
    
    # 6. Sketch
    elif mode == "🎨 Sketch":
        n = np.array(img_pil.convert("RGB"))
        gray = cv2.cvtColor(n, cv2.COLOR_RGB2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, cv2.bitwise_not(blur), scale=256.0)
        return Image.fromarray(cv2.cvtColor(sketch.astype(np.uint8), cv2.COLOR_GRAY2RGB))

    return img_pil

# ---------------- Sidebar & Tabs ----------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    quality = st.slider("Export Quality", 80, 100, 95)
    st.success("Roman Studio 2026")

tabs = st.tabs(["📸 Photo Editor", "🎥 Video Editor"])

# ================= TAB 1: PHOTO EDITOR =================
with tabs[0]:
    pic = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"], key="p_up")
    if pic:
        img_input = Image.open(pic).convert("RGB")
        col_a, col_b = st.columns(2)
        
        mode_p = st.selectbox("Select Effect", ["None", "🌟 Super Glow & Vivid", "🎨 Deep Colors", "💎 HD Clean", "💄 Beauty Face", "📱 iPhone Vibe", "🎨 Sketch"])
        
        processed_img = apply_custom_filter(img_input, mode_p)
        
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
            v_mode = st.selectbox("Select Video Effect:", ["🌟 Super Glow & Vivid", "🎨 Deep Colors", "💎 HD Clean", "💄 Beauty Face", "📱 iPhone Vibe", "🎨 Sketch"])
        with v_col2:
            v_smooth = st.slider("Beauty/Glow Intensity", 3, 30, 10)

        if st.button("🚀 Process Video (Vivid Mode)"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            tfile_path = tfile.name
            tfile.close()
            
            # Unique filename for fixing cache issue
            output_final = f"Roman_Studio_Vivid_{int(time.time())}.mp4"
            
            with st.spinner("رنگوں کو تیز کیا جا رہا ہے..."):
                try:
                    cap = cv2.VideoCapture(tfile_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    w, h = int(cap.get(cv2.CAP_PROP_WIDTH)), int(cap.get(cv2.CAP_PROP_HEIGHT))
                    
                    temp_proc = "processing_video.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(temp_proc, fourcc, fps, (w, h))

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    bar = st.progress(0)
                    
                    count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        # Convert to PIL for High Quality Filters
                        img_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        res_pil = apply_custom_filter(img_frame, v_mode, v_smooth)
                        
                        # Convert back to BGR for Video Saving
                        final_frame = cv2.cvtColor(np.array(res_pil), cv2.COLOR_RGB2BGR)
                        out.write(final_frame)
                        
                        count += 1
                        if count % 10 == 0: bar.progress(min(count/total_frames, 1.0))
                    
                    cap.release()
                    out.release()

                    # Re-attach Original Audio
                    with VideoFileClip(temp_proc) as processed_clip:
                        with VideoFileClip(tfile_path) as original_clip:
                            final_video = processed_clip.set_audio(original_clip.audio)
                            final_video.write_videofile(output_final, codec="libx264", audio_codec="aac")
                    
                    st.success("ویڈیو تیار ہے!")
                    st.video(output_final)
                    
                    with open(output_final, "rb") as f:
                        st.download_button("📥 ڈاؤن لوڈ ویڈیو", f, output_final)
                
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(tfile_path): os.remove(tfile_path)
                    if os.path.exists(temp_proc): os.remove(temp_proc)

st.markdown("---")
st.markdown("<center>Roman Studio Pro 2026 | Color & Glow Engine</center>", unsafe_allow_html=True)
