import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import tempfile
import os
from moviepy.editor import VideoFileClip

# ---------------- UI Settings ----------------

st.set_page_config(page_title="Roman Studio Pro", layout="wide", page_icon="🎨")

# Roman Studio Custom Styling
st.markdown("""
<style>
.main-header {
    padding: 1.5rem;
    text-align: center;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
    border-radius: 15px;
    margin-bottom: 20px;
}
.stButton>button {
    width: 100%;
    border-radius: 12px;
    font-weight: bold;
    height: 3.2em;
    background-color: #2a5298;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎨 Roman Studio Pro</h1><p>Full HD & 4K AI Photo + Video Studio</p></div>', unsafe_allow_html=True)

# ---------------- Core Processing Functions ----------------

def to_np(img): 
    return np.array(img.convert("RGB"))

def clean_hd_logic(img_pil):
    """HD Cleaning logic for both Photo & Video"""
    n = to_np(img_pil)
    den = cv2.fastNlMeansDenoisingColored(n, None, 6, 6, 7, 21)
    blur = cv2.GaussianBlur(den, (0,0), 1)
    hd = cv2.addWeighted(den, 1.1, blur, -0.1, 0)
    return Image.fromarray(hd)

def beauty_logic(img_pil, smooth_val):
    """Skin Smoothing logic"""
    n = to_np(img_pil)
    clean = cv2.bilateralFilter(n, smooth_val, 40, 40)
    return Image.fromarray(clean)

def iphone_look_logic(img_pil):
    """iPhone Aesthetic processing"""
    img = ImageEnhance.Color(img_pil).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    return img

# ---------------- Sidebar ----------------

with st.sidebar:
    st.title("⚙️ Control Panel")
    quality = st.slider("Export HD Quality", 80, 100, 100)
    st.info("Roman Studio Pro 2026")
    st.success("AI Core: Active")

# ---------------- Main Tabs ----------------

tabs = st.tabs(["📸 Photo Editor", "🎥 Video Editor"])

# ================= TAB 1: PHOTO EDITOR =================

with tabs[0]:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pic = st.file_uploader("Upload Image", ["jpg", "png", "jpeg", "webp"], key="photo_up")
        if pic:
            img_input = Image.open(pic).convert("RGB")
            if "img" not in st.session_state:
                st.session_state.img = img_input
                st.session_state.org = img_input

    if pic:
        with col2:
            c_orig, c_edit = st.columns(2)
            c_orig.image(st.session_state.org, caption="Original", use_container_width=True)
            c_edit.image(st.session_state.img, caption="Edited HD", use_container_width=True)

        p_tabs = st.tabs(["✨ AI Magic", "💄 Beauty", "📱 Filters"])
        
        with p_tabs[0]:
            if st.button("💎 Full HD Clean Boost", key="p_hd"):
                st.session_state.img = clean_hd_logic(st.session_state.img)
                st.rerun()
        
        with p_tabs[1]:
            smooth = st.slider("Skin Smooth", 3, 20, 8, key="p_smooth_val")
            if st.button("💄 Apply Natural Beauty", key="p_beauty"):
                st.session_state.img = beauty_logic(st.session_state.img, smooth)
                st.rerun()

        with p_tabs[2]:
            if st.button("📱 iPhone Look", key="p_iphone"):
                st.session_state.img = iphone_look_logic(st.session_state.img)
                st.rerun()

        st.markdown("---")
        buf = io.BytesIO()
        st.session_state.img.save(buf, "JPEG", quality=quality)
        st.download_button("📥 Download HD Photo", buf.getvalue(), "Roman_Studio_HD.jpg")
        
        if st.button("🔄 Reset Photo"):
            st.session_state.img = st.session_state.org
            st.rerun()
    else:
        st.info("تصویر اپلوڈ کریں تاکہ ایڈیٹنگ ٹولز نظر آئیں۔")

# ================= TAB 2: VIDEO EDITOR =================

with tabs[1]:
    st.subheader("🎥 Video AI Engine")
    video_file = st.file_uploader("Upload Video", ["mp4", "mov", "avi"], key="video_up")

    if video_file:
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            v_mode = st.radio("Select Video Effect:", ["💎 HD Clean", "💄 Beauty Face", "📱 iPhone Vibe"])
        with v_col2:
            v_smooth = st.slider("Smooth Intensity (for Beauty)", 3, 25, 10)

        if st.button("🚀 Start AI Processing"):
            # Save uploaded video to temp
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            tfile_path = tfile.name
            tfile.close()
            
            with st.spinner("Processing... اس میں ویڈیو کی لمبائی کے حساب سے وقت لگ سکتا ہے"):
                try:
                    cap = cv2.VideoCapture(tfile_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    temp_proc = "temp_proc_video.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(temp_proc, fourcc, fps, (w, h))

                    progress_bar = st.progress(0)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    count = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        # Convert BGR to PIL for our logic
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(frame_rgb)

                        # Apply selected filter
                        if v_mode == "💎 HD Clean":
                            res_pil = clean_hd_logic(pil_frame)
                        elif v_mode == "💄 Beauty Face":
                            res_pil = beauty_logic(pil_frame, v_smooth)
                        else: # iPhone Vibe
                            res_pil = iphone_look_logic(pil_frame)

                        # Back to BGR for VideoWriter
                        final_frame = cv2.cvtColor(np.array(res_pil), cv2.COLOR_RGB2BGR)
                        out.write(final_frame)
                        
                        count += 1
                        if count % 10 == 0:
                            progress_bar.progress(min(count/total_frames, 1.0))
                    
                    cap.release()
                    out.release()

                    # Re-attach Audio
                    output_final = "Roman_Studio_Final.mp4"
                    with VideoFileClip(temp_proc) as video_clip:
                        with VideoFileClip(tfile_path) as original_clip:
                            final_video = video_clip.set_audio(original_clip.audio)
                            final_video.write_videofile(output_final, codec="libx264", audio_codec="aac")
                    
                    st.success("ویڈیو کامیابی سے ایڈٹ ہو گئی!")
                    st.video(output_final)
                    with open(output_final, "rb") as f:
                        st.download_button("📥 ڈاؤن لوڈ ویڈیو", f, "Roman_Studio_Video.mp4")
                
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    # Cleanup
                    if os.path.exists(tfile_path): os.remove(tfile_path)
                    if os.path.exists("temp_proc_video.mp4"): os.remove("temp_proc_video.mp4")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("<center><p style='color:gray;'>Roman Studio Pro 2026 – Secure & Private AI</p></center>", unsafe_allow_html=True)
