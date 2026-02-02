import streamlit as st  
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
import time
from moviepy.editor import VideoFileClip, ImageSequenceClip, CompositeVideoClip, TextClip, ColorClip
from moviepy.editor import concatenate_videoclips

# 1. Page Config
st.set_page_config(page_title="Roman Studio Pro", layout="wide")

# --- تمام پرانے پروفیشنل فیچرز (Functions) ---

def enhance_to_8k_advanced(img):
    """تصویر کو 8K کوالٹی میں بدلنے اور صاف کرنے کے لیے"""
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    denoised = cv2.bilateralFilter(upscaled, 9, 80, 80)
    low_pass = cv2.GaussianBlur(denoised, (0, 0), 3.0)
    high_pass = cv2.subtract(denoised, low_pass)
    sharpened = cv2.addWeighted(denoised, 1.0, high_pass, 0.5, 0)
    return sharpened

def apply_face_wash_pro(img):
    """چہرے کی جلد کو صاف اور گلو دینے کے لیے (Face Glow)"""
    skin_smoothed = cv2.bilateralFilter(img, 11, 70, 70)
    return cv2.addWeighted(img, 0.3, skin_smoothed, 0.7, 0)

def apply_hair_color_change(img, color_type="brown", intensity=0.7):
    """بالوں کا رنگ تبدیل کرنے کا فیچر"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    hair_mask = cv2.inRange(l, 20, 100)
    if color_type == "brown":
        a = cv2.add(a, 10); b = cv2.add(b, 20)
    elif color_type == "blonde":
        l = cv2.multiply(l, 1.2); b = cv2.add(b, 30)
    elif color_type == "black":
        l = cv2.multiply(l, 0.8)
    result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(img, 1-intensity, result, intensity, 0)

def apply_cinematic_look(img):
    """ویڈیو کو سنیماٹک لک دینے کے لیے"""
    img_float = img.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_float)
    r *= 1.1; b *= 0.9 
    result = cv2.merge([b, g, r]) * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_hdr_effect(img):
    """ایچ ڈی آر ایفیکٹ (HDR Effect)"""
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

# --- ویڈیو پروسیسنگ کے مددگار فنکشنز ---

def extract_frames(video_path, frame_rate=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // frame_rate)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
    cap.release()
    return frames, fps

def process_video_frames(frames, effects_list):
    processed = []
    prog = st.progress(0)
    for i, frame in enumerate(frames):
        temp_frame = frame.copy()
        for fx in effects_list:
            if fx['type'] == "8k": temp_frame = enhance_to_8k_advanced(temp_frame)
            if fx['type'] == "face": temp_frame = apply_face_wash_pro(temp_frame)
            if fx['type'] == "cine": temp_frame = apply_cinematic_look(temp_frame)
            if fx['type'] == "hdr": temp_frame = apply_hdr_effect(temp_frame)
        processed.append(temp_frame)
        prog.progress((i + 1) / len(frames))
    return processed

# --- مین انٹرفیس (Main UI) ---
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🎬 Roman Studio Pro</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📹 ویڈیو ایڈیٹر", "🖼️ فوٹو ایڈیٹر", "⚙️ سیٹنگز"])

with tab1:
    st.header("ویڈیو ایڈیٹنگ (Roman Studio)")
    video_file = st.file_uploader("ویڈیو اپ لوڈ کریں", type=['mp4', 'mov', 'avi'])
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        if 'video_frames' not in st.session_state:
            with st.spinner("ویڈیو لوڈ ہو رہی ہے..."):
                st.session_state.video_frames, st.session_state.video_fps = extract_frames(tfile.name)
                st.session_state.original_frames = st.session_state.video_frames.copy()

        st.subheader("ایفیکٹس منتخب کریں")
        col1, col2, col3 = st.columns(3)
        
        if col1.button("🚀 8K + Face Glow"):
            st.session_state.video_frames = process_video_frames(st.session_state.original_frames, [{'type': '8k'}, {'type': 'face'}])
            st.success("ایفیکٹ لگ گیا!")
            
        if col2.button("🎬 Cinematic Look"):
            st.session_state.video_frames = process_video_frames(st.session_state.original_frames, [{'type': 'cine'}, {'type': 'hdr'}])
            st.success("سنیماٹک ایفیکٹ مکمل!")

        if col3.button("🔄 ری سیٹ ویڈیو"):
            st.session_state.video_frames = st.session_state.original_frames.copy()
            st.rerun()

        st.image(st.session_state.video_frames[len(st.session_state.video_frames)//2], caption="پیش نظارہ (Preview)", use_container_width=True)

        if st.button("📥 ویڈیو ڈاؤن لوڈ کریں"):
            out_p = "roman_studio_video.mp4"
            clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in st.session_state.video_frames], fps=st.session_state.video_fps)
            clip.write_videofile(out_p, codec='libx264')
            with open(out_p, "rb") as f:
                st.download_button("فائل ڈاؤن لوڈ کریں", f, file_name="Roman_Studio_Video.mp4")

with tab2:
    st.header("فوٹو ایڈیٹنگ (Roman Studio)")
    img_file = st.file_uploader("تصویر اپ لوڈ کریں", type=['jpg', 'jpeg', 'png'])
    
    if img_file:
        raw_img = Image.open(img_file)
        img_np = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(raw_img, caption="Original Photo")
            
        with col_img2:
            st.subheader("ایڈیٹنگ ٹولز")
            if st.button("✨ Auto Enhance (8K + Face Glow)"):
                res = enhance_to_8k_advanced(img_np)
                res = apply_face_wash_pro(res)
                st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), caption="Roman Studio Enhanced")
                
                is_success, buffer = cv2.imencode(".jpg", res)
                st.download_button("تصویر محفوظ کریں", io.BytesIO(buffer), file_name="Roman_Enhanced.jpg")
            
            if st.button("🤎 Brown Hair Effect"):
                res = apply_hair_color_change(img_np, "brown")
                st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), caption="Hair Color Changed")

with tab3:
    st.header("سیٹنگز اور سیکیورٹی")
    st.write("Roman Studio آپ کے ڈیٹا کی حفاظت کو یقینی بناتا ہے۔")
    if st.button("ڈیٹا صاف کریں (Reset App)"):
        st.session_state.clear()
        st.rerun()
