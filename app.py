import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
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
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎨 Roman Studio Pro</h1><p>Full HD & 4K AI Photo + Video Studio</p></div>', unsafe_allow_html=True)

# ---------------- Helper Functions ----------------

def to_np(img): 
    return np.array(img.convert("RGB"))

def clean_hd(img):
    n = to_np(img)
    den = cv2.fastNlMeansDenoisingColored(n, None, 6, 6, 7, 21)
    blur = cv2.GaussianBlur(den, (0,0), 1)
    hd = cv2.addWeighted(den, 1.1, blur, -0.1, 0)
    return Image.fromarray(hd)

# ---------------- Sidebar ----------------

with st.sidebar:
    st.title("⚙️ Control Panel")
    quality = st.slider("Export HD Quality", 80, 100, 100)
    st.success("All Systems Active")
    st.info("Roman Studio Pro 2026")

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

        p_tabs = st.tabs(["✨ AI Magic", "💄 Beauty", "🎬 Social/Pro"])
        
        with p_tabs[0]:
            if st.button("💎 Full HD Clean Boost"):
                st.session_state.img = clean_hd(st.session_state.img)
                st.rerun()
        
        with p_tabs[1]:
            smooth = st.slider("Skin Smooth", 3, 20, 8)
            if st.button("💄 Apply Natural Beauty"):
                n = to_np(st.session_state.img)
                clean = cv2.bilateralFilter(n, smooth, 40, 40)
                st.session_state.img = Image.fromarray(clean)
                st.rerun()

        with p_tabs[2]:
            if st.button("📱 iPhone Look"):
                img = st.session_state.img
                img = ImageEnhance.Color(img).enhance(1.1)
                img = ImageEnhance.Sharpness(img).enhance(1.3)
                st.session_state.img = img
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
    st.subheader("📱 TikTok Full HD + Beauty")
    video_file = st.file_uploader("Upload Video", ["mp4", "mov", "avi"], key="video_up")

    if video_file:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        tfile.close()
        
        if st.button("✨ Start AI Processing (Keep Voice)"):
            with st.spinner("Processing Video... اس میں تھوڑا وقت لگ سکتا ہے"):
                try:
                    cap = cv2.VideoCapture(tfile.name)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    temp_proc = "temp_output.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(temp_proc, fourcc, fps, (w, h))

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        # Apply subtle beauty filter
                        proc_frame = cv2.bilateralFilter(frame, 5, 30, 30)
                        out.write(proc_frame)
                    
                    cap.release()
                    out.release()

                    # Merge Audio back using MoviePy
                    video_clip = VideoFileClip(temp_proc)
                    original_clip = VideoFileClip(tfile.name)
                    final_clip = video_clip.set_audio(original_clip.audio)
                    
                    output_name = "Roman_Studio_Video_HD.mp4"
                    final_clip.write_videofile(output_name, codec="libx264", audio_codec="aac")
                    
                    st.success("پروسیسنگ مکمل ہوگئی!")
                    st.video(output_name)
                    with open(output_name, "rb") as f:
                        st.download_button("📥 ڈاؤن لوڈ ویڈیو", f, "Roman_HD_Video.mp4")
                
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(tfile.name): os.remove(tfile.name)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("<center><p style='color:gray;'>Roman Studio Pro 2026 – All AI Systems Active</p></center>", unsafe_allow_html=True)
