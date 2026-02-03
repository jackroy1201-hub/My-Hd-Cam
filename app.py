import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
 militant io, tempfile, os
from moviepy.editor import VideoFileClip

# ---------------- UI Settings ----------------

st.set_page_config(page_title="Roman Studio Pro", layout="wide", page_icon="🎨")

# Custom CSS for Roman Studio Look
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

# ہم نے ٹیبز کو باہر نکال لیا ہے تاکہ ویڈیو آپشن ہمیشہ نظر آئے
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
            c1, c2 = st.columns(2)
            c1.image(st.session_state.org, caption="Original", use_container_width=True)
            c2.image(st.session_state.img, caption="Edited HD", use_container_width=True)

        # Photo Tools inside sub-tabs
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

        # Download Photo
        buf = io.BytesIO()
        st.session_state.img.save(buf, "JPEG", quality=quality)
        st.download_button("📥 Download HD Photo", buf.getvalue(), "Roman_Studio_HD.jpg")
        if st.button("🔄 Reset Photo"):
            st.session_state.img = st.session_state.org
            st.rerun()
    else:
        st.info("Please upload a photo to see editing options.")

# ================= TAB 2: VIDEO EDITOR =================

with tabs[1]:
    st.subheader("📱 TikTok Full HD + Beauty + Voice Safe")
    video_file = st.file_uploader("Upload Video", ["mp4", "mov", "avi"], key="video_up")

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        v_col1, v_col2 = st.columns(2)
        
        with v_col1:
            if st.button("✨ TikTok Beauty HD (Keep Voice)"):
                with st.spinner("Processing Video... Please wait"):
                    cap = cv2.VideoCapture(tfile.name)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    temp_v = "temp_proc.mp4"
                    out = cv2.VideoWriter(temp_v, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        # Applying Beauty Filter to each frame
                        frame = cv2.bilateralFilter(frame, 5, 30, 30)
                        out.write(frame)
                    
                    cap.release()
                    out.release()

                    # Merge Audio
                    orig_clip = VideoFileClip(tfile.name)
                    processed_clip = VideoFileClip(temp_v)
                    final_clip = processed_clip.set_audio(orig_clip.audio)
                    final_path = "Roman_Studio_TikTok_HD.mp4"
                    final_clip.write_videofile(final_path, codec="libx264", audio_codec="aac")
                    
                    st.success("Video Ready!")
                    st.video(final_path)
                    with open(final_path, "rb") as f:
                        st.download_button("📥 Download Processed Video", f, "Roman_Studio_Video.mp4")

        with v_col2:
            st.write("Video Details:")
            st.write(f"Format: {video_file.type}")
            st.write("Ready for AI Enhancement")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("<center><p style='color:gray;'>Roman Studio Pro 2026 – All AI Systems Active</p></center>", unsafe_allow_html=True)
