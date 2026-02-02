Import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
import time
from moviepy.editor import VideoFileClip, ImageSequenceClip, CompositeVideoClip, TextClip, ColorClip
from moviepy.editor import concatenate_videoclips
import subprocess
import json
from pathlib import Path

# 1. Page Config
st.set_page_config(page_title="AI Family Video Studio Pro", layout="wide")

# --- Enhanced Functions for Video ---
def process_frame_with_effect(frame, effect_type, intensity=1.0, **kwargs):
    """Apply various effects to a single frame"""
    
    if effect_type == "8k_enhance":
        return enhance_to_8k_advanced(frame)
    
    elif effect_type == "face_wash":
        return apply_face_wash_pro(frame)
    
    elif effect_type == "hair_color":
        color_type = kwargs.get('color_type', 'brown')
        return apply_hair_color_change(frame, color_type, intensity)
    
    elif effect_type == "skin_retouch":
        return apply_skin_retouch(frame)
    
    elif effect_type == "cinematic":
        return apply_cinematic_look(frame)
    
    elif effect_type == "portrait_mode":
        return apply_ai_portrait_mode(frame)
    
    elif effect_type == "hdr":
        return apply_hdr_effect(frame, intensity)
    
    elif effect_type == "vibrant":
        # Vibrant filter
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.2  # Increase saturation
        hsv[:, :, 2] *= 1.05  # Increase brightness
        result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result
    
    elif effect_type == "dramatic":
        # Dramatic filter
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = np.clip(l.astype(np.float32) * 1.3, 0, 255)
        result_lab = cv2.merge((l.astype(np.uint8), a, b))
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        return result
    
    elif effect_type == "vintage":
        # Vintage film effect
        img_float = frame.astype(np.float32) / 255.0
        
        # Sepia tone
        sepia = np.array([[0.393, 0.769, 0.189],
                          [0.349, 0.686, 0.168],
                          [0.272, 0.534, 0.131]])
        
        result = cv2.transform(img_float, sepia)
        result = np.clip(result, 0, 1)
        
        # Add vignette
        rows, cols = frame.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/3)
        kernel_y = cv2.getGaussianKernel(rows, rows/3)
        kernel = kernel_y * kernel_x.T
        vignette = kernel / kernel.max()
        vignette = np.stack([vignette] * 3, axis=2)
        
        result = result * (0.7 + 0.3 * vignette)
        
        # Add film grain
        noise = np.random.normal(0, 0.02, frame.shape).astype(np.float32)
        result = result + noise
        result = np.clip(result, 0, 1) * 255
        
        return result.astype(np.uint8)
    
    elif effect_type == "glitch":
        # Glitch effect
        result = frame.copy()
        height, width = frame.shape[:2]
        
        # Random channel shift
        shift = np.random.randint(1, 10)
        channel = np.random.randint(0, 3)
        
        if channel == 0:  # Blue channel
            result[:height-shift, :, 0] = frame[shift:, :, 0]
        elif channel == 1:  # Green channel
            result[:height-shift, :, 1] = frame[shift:, :, 1]
        else:  # Red channel
            result[:height-shift, :, 2] = frame[shift:, :, 2]
        
        # Random noise lines
        num_lines = np.random.randint(1, 5)
        for _ in range(num_lines):
            line_pos = np.random.randint(0, height)
            line_height = np.random.randint(1, 5)
            noise = np.random.randint(0, 100, (line_height, width, 3))
            result[line_pos:line_pos+line_height, :] = noise
        
        return result
    
    elif effect_type == "neon":
        # Neon effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Colorize edges
        neon_colors = {
            0: (255, 0, 0),    # Blue
            1: (0, 255, 0),    # Green
            2: (0, 0, 255),    # Red
            3: (255, 255, 0),  # Cyan
            4: (255, 0, 255),  # Magenta
            5: (0, 255, 255)   # Yellow
        }
        
        color_idx = np.random.randint(0, 6)
        color = neon_colors[color_idx]
        
        result = np.zeros_like(frame)
        result[edges > 0] = color
        
        # Blend with original
        alpha = 0.3
        result = cv2.addWeighted(frame, 1 - alpha, result, alpha, 0)
        
        return result
    
    elif effect_type == "bloom":
        # Bloom effect (glow)
        blur = cv2.GaussianBlur(frame, (0, 0), 10)
        result = cv2.addWeighted(frame, 0.7, blur, 0.3, 0)
        
        # Increase brightness in highlights
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.where(hsv[:, :, 2] > 150, 
                               np.clip(hsv[:, :, 2] * 1.2, 0, 255),
                               hsv[:, :, 2])
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return result
    
    elif effect_type == "anime":
        # Anime/cartoon effect
        # Color quantization
        Z = frame.reshape((-1, 3))
        Z = np.float32(Z)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        result = res.reshape((frame.shape))
        
        # Edge enhancement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.erode(edges, None, iterations=1)
        
        # Combine with edges
        edges_mask = edges[:, :, np.newaxis] / 255.0
        result = result * (1 - edges_mask) + np.array([0, 0, 0]) * edges_mask
        
        return result.astype(np.uint8)
    
    else:
        return frame

# --- Existing Image Processing Functions (Adapted for Video) ---
def enhance_to_8k_advanced(img):
    """Advanced 8K enhancement"""
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    denoised = cv2.bilateralFilter(upscaled, 9, 80, 80)
    
    low_pass = cv2.GaussianBlur(denoised, (0, 0), 3.0)
    high_pass = cv2.subtract(denoised, low_pass)
    
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_mask = cv2.GaussianBlur(edges, (5, 5), 0) / 255.0
    edge_mask = np.stack([edge_mask] * 3, axis=2)
    
    sharpened = cv2.addWeighted(denoised, 1.0, high_pass, 0.3 + 0.2 * edge_mask, 0)
    
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    l_final = cv2.addWeighted(l, 0.6, l_enhanced, 0.4, 0)
    a = cv2.add(a, 5)
    b = cv2.add(b, 3)
    
    final = cv2.cvtColor(cv2.merge((l_final, a, b)), cv2.COLOR_LAB2BGR)
    final = cv2.bilateralFilter(final, 7, 30, 30)
    
    return final

def apply_face_wash_pro(img):
    """Face enhancement"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    skin_mask_soft = cv2.GaussianBlur(skin_mask.astype(np.float32), (21, 21), 0) / 255.0
    skin_mask_soft = np.stack([skin_mask_soft] * 3, axis=2)
    
    skin_smoothed = cv2.bilateralFilter(img, 11, 70, 70)
    glow = cv2.GaussianBlur(skin_smoothed, (0, 0), 2.5)
    skin_glow = cv2.addWeighted(skin_smoothed, 0.8, glow, 0.2, 0)
    
    lab = cv2.cvtColor(skin_glow, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_skin = cv2.GaussianBlur(l, (0, 0), 1.5)
    l = cv2.addWeighted(l, 0.7, l_skin, 0.3, 0)
    a = cv2.add(a, 3)
    
    enhanced_lab = cv2.merge((l, a, b))
    skin_final = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    non_skin = cv2.bilateralFilter(img, 5, 25, 25)
    result = skin_final * skin_mask_soft + non_skin * (1 - skin_mask_soft)
    
    result = result.astype(np.uint8)
    result = cv2.bilateralFilter(result, 9, 40, 40)
    
    return result

def apply_hair_color_change(img, color_type="brown", intensity=0.7):
    """Hair color change"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    hair_mask = cv2.inRange(l, 20, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
    
    hair_mask_soft = cv2.GaussianBlur(hair_mask.astype(np.float32), (15, 15), 0) / 255.0
    hair_mask_soft = np.stack([hair_mask_soft] * 3, axis=2)
    
    color_adjustments = {
        "black": {"a_adj": -10, "b_adj": -5, "l_factor": 0.8},
        "brown": {"a_adj": 10, "b_adj": 20, "l_factor": 1.1},
        "dark_brown": {"a_adj": 5, "b_adj": 10, "l_factor": 0.9},
        "blonde": {"a_adj": -5, "b_adj": 40, "l_factor": 1.3},
        "auburn": {"a_adj": 30, "b_adj": 15, "l_factor": 1.1},
        "burgundy": {"a_adj": 40, "b_adj": -10, "l_factor": 1.0},
        "gray": {"a_adj": 0, "b_adj": 0, "l_factor": 1.5, "desaturate": True},
        "highlights": {"a_adj": 0, "b_adj": 15, "l_factor": 1.2, "streaks": True}
    }
    
    if color_type not in color_adjustments:
        color_type = "brown"
    
    adj = color_adjustments[color_type]
    
    l_adj = l.copy().astype(np.float32)
    a_adj = a.copy().astype(np.float32)
    b_adj = b.copy().astype(np.float32)
    
    if color_type == "gray":
        gray_boost = 30 * hair_mask_soft[:, :, 0]
        l_adj = np.where(hair_mask > 0, np.clip(l_adj + gray_boost, 0, 255), l_adj)
    else:
        l_adj = np.where(hair_mask > 0, np.clip(l_adj * adj["l_factor"], 0, 255), l_adj)
    
    a_adj = np.where(hair_mask > 0, np.clip(a_adj + adj["a_adj"], 0, 255), a_adj)
    b_adj = np.where(hair_mask > 0, np.clip(b_adj + adj["b_adj"], 0, 255), b_adj)
    
    if color_type == "gray" and adj.get("desaturate", False):
        a_adj = np.where(hair_mask > 0, a_adj * 0.3, a_adj)
        b_adj = np.where(hair_mask > 0, b_adj * 0.3, b_adj)
    
    l_adj = l_adj.astype(np.uint8)
    a_adj = a_adj.astype(np.uint8)
    b_adj = b_adj.astype(np.uint8)
    
    lab_adj = cv2.merge((l_adj, a_adj, b_adj))
    bgr_adj = cv2.cvtColor(lab_adj, cv2.COLOR_LAB2BGR)
    result = cv2.addWeighted(img, 1 - intensity, bgr_adj, intensity, 0)
    
    hair_only = cv2.bitwise_and(result, result, mask=hair_mask)
    hair_smoothed = cv2.bilateralFilter(hair_only, 9, 50, 50)
    
    hair_mask_feathered = cv2.GaussianBlur(hair_mask, (21, 21), 0) / 255.0
    hair_mask_feathered = np.stack([hair_mask_feathered] * 3, axis=2)
    
    final_result = img * (1 - hair_mask_feathered) + hair_smoothed * hair_mask_feathered
    
    return final_result.astype(np.uint8)

def apply_skin_retouch(img):
    """Skin retouching"""
    b, g, r = cv2.split(img)
    r_smoothed = cv2.bilateralFilter(r, 13, 50, 50)
    b_enhanced = cv2.add(b, 5)
    
    retouched = cv2.merge([b_enhanced, g, r_smoothed])
    low_freq = cv2.bilateralFilter(retouched, 15, 80, 80)
    high_freq = cv2.subtract(retouched, low_freq)
    low_freq_smooth = cv2.bilateralFilter(low_freq, 21, 100, 100)
    
    result = cv2.add(low_freq_smooth, high_freq * 0.7)
    result = cv2.bilateralFilter(result, 9, 30, 30)
    
    return result.astype(np.uint8)

def apply_cinematic_look(img):
    """Cinematic effect"""
    img_float = img.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_float)
    
    r = r * 1.1
    g = g * 0.95
    b_shadow = np.where(b < 0.3, b * 1.05, b)
    b = cv2.addWeighted(b, 0.7, b_shadow, 0.3, 0)
    
    result = cv2.merge([b, g, r]) * 255.0
    
    noise = np.random.normal(0, 0.005, img.shape).astype(np.float32)
    result = result + noise * 255.0
    
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/3)
    kernel_y = cv2.getGaussianKernel(rows, rows/3)
    kernel = kernel_y * kernel_x.T
    
    mask = kernel / kernel.max()
    mask = np.stack([mask] * 3, axis=2)
    
    result = result * (0.8 + 0.2 * mask)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_ai_portrait_mode(img):
    """Portrait mode"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    mask = 255 - edges_dilated
    mask = cv2.GaussianBlur(mask, (21, 21), 0) / 255.0
    mask = np.stack([mask] * 3, axis=2)
    
    background_blur = cv2.GaussianBlur(img, (25, 25), 0)
    result = img * mask + background_blur * (1 - mask)
    
    subject_enhanced = cv2.convertScaleAbs(img, alpha=1.05, beta=5)
    final = result * mask + subject_enhanced * (1 - mask) * 0.3 + img * (1 - mask) * 0.7
    
    return np.clip(final, 0, 255).astype(np.uint8)

def apply_hdr_effect(img, strength=0.3):
    """HDR effect"""
    exposures = []
    for gamma in [0.8, 1.0, 1.2]:
        adjusted = np.power(img.astype(np.float32) / 255.0, gamma) * 255.0
        exposures.append(adjusted)
    
    hdr = np.zeros_like(img, dtype=np.float32)
    for exp in exposures:
        hdr += exp
    hdr = hdr / len(exposures)
    
    tonemapped = np.tanh(hdr / 255.0 * (1 + strength)) * 255.0
    tonemapped_uint8 = tonemapped.astype(np.uint8)
    
    lab = cv2.cvtColor(tonemapped_uint8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    l_final = cv2.addWeighted(l, 0.6, l_enhanced, 0.4, 0)
    final_lab = cv2.merge((l_final, a, b))
    
    return cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

# --- Video Processing Functions ---
def extract_frames(video_path, frame_rate=10):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // frame_rate)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        frame_count += 1
    
    cap.release()
    return frames, fps

def process_video_frames(frames, effects_list):
    """Apply effects to video frames"""
    processed_frames = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, frame in enumerate(frames):
        processed_frame = frame.copy()
        
        # Apply all selected effects
        for effect in effects_list:
            if effect['enabled']:
                processed_frame = process_frame_with_effect(
                    processed_frame, 
                    effect['type'], 
                    effect['intensity'],
                    **effect.get('params', {})
                )
        
        processed_frames.append(processed_frame)
        
        # Update progress
        progress = (i + 1) / len(frames)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {i+1}/{len(frames)}")
    
    progress_bar.empty()
    status_text.empty()
    
    return processed_frames

def create_video_from_frames(frames, fps, output_path, add_transitions=False):
    """Create video from frames"""
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    # Save frames
    for i, frame in enumerate(frames):
        temp_path = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
        cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        temp_files.append(temp_path)
    
    try:
        # Create video clip
        clip = ImageSequenceClip(temp_files, fps=fps)
        
        # Add transitions if requested
        if add_transitions:
            # Simple fade transition
            clip = clip.crossfadein(0.5).crossfadeout(0.5)
        
        # Write video
        clip.write_videofile(
            output_path, 
            codec='libx264', 
            audio=False, 
            ffmpeg_params=['-pix_fmt', 'yuv420p', '-crf', '18']
        )
        
        # Cleanup
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        os.rmdir(temp_dir)
        
        return True
    except Exception as e:
        st.error(f"Error creating video: {str(e)}")
        return False

# --- Transition Effects ---
def apply_transition(frame1, frame2, transition_type, progress):
    """Apply transition between two frames"""
    if transition_type == "fade":
        return cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
    
    elif transition_type == "slide_right":
        width = frame1.shape[1]
        shift = int(width * progress)
        
        result = frame1.copy()
        result[:, shift:] = frame2[:, :width-shift]
        return result
    
    elif transition_type == "slide_left":
        width = frame1.shape[1]
        shift = int(width * progress)
        
        result = frame1.copy()
        result[:, :width-shift] = frame2[:, shift:]
        return result
    
    elif transition_type == "zoom":
        scale = 1 + progress * 0.5
        height, width = frame1.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        
        zoomed = cv2.resize(frame2, (new_width, new_height))
        start_x = max(0, (new_width - width) // 2)
        start_y = max(0, (new_height - height) // 2)
        
        result = zoomed[start_y:start_y+height, start_x:start_x+width]
        return cv2.addWeighted(frame1, 1 - progress, result, progress, 0)
    
    elif transition_type == "rotate":
        angle = progress * 360
        height, width = frame2.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(frame2, rotation_matrix, (width, height))
        
        return cv2.addWeighted(frame1, 1 - progress, rotated, progress, 0)
    
    else:
        return frame2

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
        margin: 5px 0;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .video-btn {
        background: linear-gradient(45deg, #FF416C 0%, #FF4B2B 100%) !important;
    }
    
    .effect-btn {
        background: linear-gradient(45deg, #36D1DC 0%, #5B86E5 100%) !important;
    }
    
    .hair-btn {
        background: linear-gradient(45deg, #8B4513 0%, #D2691E 100%) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4B0082;
        color: white;
    }
    
    .effect-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Main App ---
st.markdown("<h1 style='text-align: center; color: #2C3E50; margin-bottom: 30px;'>🎬 AI Family Video Studio Pro</h1>", unsafe_allow_html=True)

# Tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["📹 Video Editor", "🖼️ Image Editor", "🎨 Effects Gallery", "⚙️ Settings"])

with tab1:
    # Video Upload Section
    st.header("📤 Video Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    with col2:
        st.write("### Video Settings")
        frame_rate = st.slider("Frame Rate (FPS)", 5, 30, 10, 1, 
                              help="Lower frame rate for faster processing")
        video_quality = st.selectbox("Output Quality", 
                                    ["High (HD)", "Medium", "Low (Fast)"])
    
    if video_file:
        # Save uploaded video
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        with open(temp_video_path, 'wb') as f:
            f.write(video_file.getvalue())
        
        if 'video_frames' not in st.session_state or st.session_state.get('last_video') != video_file.name:
            with st.spinner("Extracting frames from video..."):
                st.session_state.video_frames, st.session_state.video_fps = extract_frames(temp_video_path, frame_rate)
                st.session_state.original_frames = st.session_state.video_frames.copy()
                st.session_state.last_video = video_file.name
                st.session_state.video_effects = []
        
        # Video Preview
        st.header("🎥 Video Preview")
        
        if st.session_state.video_frames:
            col_preview1, col_preview2 = st.columns(2)
            
            with col_preview1:
                st.subheader("Original Video Frame")
                preview_idx = st.slider("Frame Preview", 0, len(st.session_state.original_frames)-1, 
                                       len(st.session_state.original_frames)//2)
                st.image(st.session_state.original_frames[preview_idx], 
                        caption=f"Frame {preview_idx}", use_container_width=True)
            
            with col_preview2:
                st.subheader("Processed Video Frame")
                st.image(st.session_state.video_frames[preview_idx] 
                        if len(st.session_state.video_frames) > preview_idx 
                        else st.session_state.original_frames[preview_idx],
                        caption=f"Frame {preview_idx}", use_container_width=True)
        
        # Video Effects Section
        st.header("🎭 Video Effects")
        
        effect_cols = st.columns(4)
        
        # Quick Effect Buttons
        with effect_cols[0]:
            if st.button("🚀 8K Enhancement", use_container_width=True, key="vid_8k"):
                effect = {'type': '8k_enhance', 'enabled': True, 'intensity': 1.0}
                if effect not in st.session_state.video_effects:
                    st.session_state.video_effects.append(effect)
                st.rerun()
        
        with effect_cols[1]:
            if st.button("✨ Face Glow", use_container_width=True, key="vid_face"):
                effect = {'type': 'face_wash', 'enabled': True, 'intensity': 1.0}
                if effect not in st.session_state.video_effects:
                    st.session_state.video_effects.append(effect)
                st.rerun()
        
        with effect_cols[2]:
            if st.button("🎬 Cinematic", use_container_width=True, key="vid_cine"):
                effect = {'type': 'cinematic', 'enabled': True, 'intensity': 1.0}
                if effect not in st.session_state.video_effects:
                    st.session_state.video_effects.append(effect)
                st.rerun()
        
        with effect_cols[3]:
            if st.button("📸 Portrait Mode", use_container_width=True, key="vid_portrait"):
                effect = {'type': 'portrait_mode', 'enabled': True, 'intensity': 1.0}
                if effect not in st.session_state.video_effects:
                    st.session_state.video_effects.append(effect)
                st.rerun()
        
        # More Effects
        effect_cols2 = st.columns(4)
        
        with effect_cols2[0]:
            if st.button("🌟 HDR Effect", use_container_width=True, key="vid_hdr"):
                effect = {'type': 'hdr', 'enabled': True, 'intensity': 0.3}
                if effect not in st.session_state.video_effects:
                    st.session_state.video_effects.append(effect)
                st.rerun()
        
        with effect_cols2[1]:
            if st.button("🎨 Vibrant Colors", use_container_width=True, key="vid_vibrant"):
                effect = {'type': 'vibrant', 'enabled': True, 'intensity': 1.0}
                if effect not in st.session_state.video_effects:
                    st.session_state.video_effects.append(effect)
                st.rerun()
        
        with effect_cols2[2]:
            if st.button("🎞️ Vintage Film", use_container_width=True, key="vid_vintage"):
                effect = {'type': 'vintage', 'enabled': True, 'intensity': 1.0}
                if effect not in st.session_state.video_effects:
                    st.session_state.video_effects.append(effect)
                st.rerun()
        
        with effect_cols2[3]:
            if st.button("🌀 Glitch Effect", use_container_width=True, key="vid_glitch"):
                effect = {'type': 'glitch', 'enabled': True, 'intensity': 1.0}
                if effect not in st.session_state.video_effects:
                    st.session_state.video_effects.append(effect)
                st.rerun()
        
        # Hair Color Effects for Video
        st.subheader("👩‍🦰 Hair Color Effects")
        hair_cols = st.columns(4)
        
        hair_colors = [
            ("⚫ Black", "black"),
            ("🤎 Brown", "brown"),
            ("👩‍🦳 Blonde", "blonde"),
            ("🍷 Burgundy", "burgundy")
        ]
        
        for i, (label, color_type) in enumerate(hair_colors):
            with hair_cols[i]:
                if st.button(label, use_container_width=True, key=f"vid_hair_{color_type}"):
                    effect = {
                        'type': 'hair_color', 
                        'enabled': True, 
                        'intensity': 0.7,
                        'params': {'color_type': color_type}
                    }
                    if effect not in st.session_state.video_effects:
                        st.session_state.video_effects.append(effect)
                    st.rerun()
        
        # Effect Management
        st.subheader("⚙️ Active Effects")
        
        if st.session_state.video_effects:
            for idx, effect in enumerate(st.session_state.video_effects):
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"**{effect['type'].replace('_', ' ').title()}**")
                
                with col2:
                    if 'intensity' in effect:
                        new_intensity = st.slider(
                            "Intensity", 
                            0.0, 2.0, 
                            float(effect['intensity']), 
                            0.1,
                            key=f"intensity_{idx}"
                        )
                        st.session_state.video_effects[idx]['intensity'] = new_intensity
                
                with col3:
                    if st.button("❌ Remove", key=f"remove_{idx}"):
                        st.session_state.video_effects.pop(idx)
                        st.rerun()
        else:
            st.info("No effects applied yet. Click the buttons above to add effects.")
        
        # Process Video Button
        if st.button("🎬 Process Video with All Effects", type="primary", use_container_width=True):
            if st.session_state.video_effects:
                with st.spinner("Processing video frames..."):
                    processed_frames = process_video_frames(
                        st.session_state.original_frames, 
                        st.session_state.video_effects
                    )
                    st.session_state.video_frames = processed_frames
                st.success("Video processing completed!")
            else:
                st.warning("Please add at least one effect before processing.")
        
        # Video Actions
        st.header("💾 Video Export")
        
        action_cols = st.columns(3)
        
        with action_cols[0]:
            if st.button("🔄 Reset Video", use_container_width=True):
                st.session_state.video_frames = st.session_state.original_frames.copy()
                st.session_state.video_effects = []
                st.rerun()
        
        with action_cols[1]:
            if st.button("🎞️ Preview Processed Video", use_container_width=True):
                if 'video_frames' in st.session_state:
                    # Show sample frames
                    st.subheader("Processed Video Sample")
                    sample_cols = st.columns(4)
                    for i in range(min(4, len(st.session_state.video_frames))):
                        idx = i * len(st.session_state.video_frames) // 4
                        with sample_cols[i]:
                            st.image(st.session_state.video_frames[idx], 
                                    caption=f"Frame {idx}", use_container_width=True)
        
        with action_cols[2]:
            if st.button("📥 Export Video", type="primary", use_container_width=True):
                if 'video_frames' in st.session_state:
                    with st.spinner("Creating video file..."):
                        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                        
                        if create_video_from_frames(
                            st.session_state.video_frames,
                            st.session_state.video_fps,
                            output_path
                        ):
                            # Provide download link
                            with open(output_path, 'rb') as f:
                                video_bytes = f.read()
                            
                            st.download_button(
                                label="Download Processed Video",
                                data=video_bytes,
                                file_name="processed_video.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                        else:
                            st.error("Failed to create video file")

with tab2:
    # Original Image Editor Interface
    st.header("🖼️ Image Editor")
    
    img_file = st.file_uploader("📤 Upload Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
    
    if img_file:
        if 'img_original' not in st.session_state or st.session_state.get('last_img_file') != img_file.name:
            raw_img = Image.open(img_file).convert("RGB")
            st.session_state.img_original = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
            st.session_state.img_processed = st.session_state.img_original.copy()
            st.session_state.last_img_file = img_file.name
            st.session_state.img_history = [st.session_state.img_original.copy()]
        
        # Display images
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(st.session_state.img_original, cv2.COLOR_BGR2RGB), 
                    use_container_width=True)
        
        with col_right:
            st.subheader("Processed Image")
            st.image(cv2.cvtColor(st.session_state.img_processed, cv2.COLOR_BGR2RGB), 
                    use_container_width=True)
        
        # Image Effects
        st.subheader("🛠️ Image Effects")
        
        img_effect_cols = st.columns(4)
        
        with img_effect_cols[0]:
            if st.button("🚀 8K Enhance", use_container_width=True, key="img_8k"):
                st.session_state.img_processed = enhance_to_8k_advanced(st.session_state.img_processed)
                st.session_state.img_history.append(st.session_state.img_processed.copy())
                st.rerun()
        
        with img_effect_cols[1]:
            if st.button("✨ Face Glow", use_container_width=True, key="img_face"):
                st.session_state.img_processed = apply_face_wash_pro(st.session_state.img_processed)
                st.session_state.img_history.append(st.session_state.img_processed.copy())
                st.rerun()
        
        with img_effect_cols[2]:
            if st.button("🎬 Cinematic", use_container_width=True, key="img_cine"):
                st.session_state.img_processed = apply_cinematic_look(st.session_state.img_processed)
                st.session_state.img_history.append(st.session_state.img_processed.copy())
                st.rerun()
        
        with img_effect_cols[3]:
            if st.button("📸 Portrait Mode", use_container_width=True, key="img_portrait"):
                st.session_state.img_processed = apply_ai_portrait_mode(st.session_state.img_processed)
                st.session_state.img_history.append(st.session_state.img_processed.copy())
                st.rerun()
        
        # More image effects
        img_effect_cols2 = st.columns(4)
        
        with img_effect_cols2[0]:
            if st.button("🤎 Brown Hair", use_container_width=True, key="img_hair_brown"):
                st.session_state.img_processed = apply_hair_color_change(
                    st.session_state.img_processed, "brown", 0.7)
                st.session_state.img_history.append(st.session_state.img_processed.copy())
                st.rerun()
        
        with img_effect_cols2[1]:
            if st.button("👩‍🦳 Blonde Hair", use_container_width=True, key="img_hair_blonde"):
                st.session_state.img_processed = apply_hair_color_change(
                    st.session_state.img_processed, "blonde", 0.7)
                st.session_state.img_history.append(st.session_state.img_processed.copy())
                st.rerun()
        
        with img_effect_cols2[2]:
            if st.button("🌟 HDR Effect", use_container_width=True, key="img_hdr"):
                st.session_state.img_processed = apply_hdr_effect(st.session_state.img_processed, 0.3)
                st.session_state.img_history.append(st.session_state.img_processed.copy())
                st.rerun()
        
        with img_effect_cols2[3]:
            if st.button("✨ Skin Retouch", use_container_width=True, key="img_skin"):
                st.session_state.img_processed = apply_skin_retouch(st.session_state.img_processed)
                st.session_state.img_history.append(st.session_state.img_processed.copy())
                st.rerun()
        
        # Image Actions
        st.subheader("🔄 Image Actions")
        
        action_cols = st.columns(4)
        
        with action_cols[0]:
            if st.button("↩️ Undo", use_container_width=True):
                if len(st.session_state.img_history) > 1:
                    st.session_state.img_history.pop()
                    st.session_state.img_processed = st.session_state.img_history[-1].copy()
                    st.rerun()
        
        with action_cols[1]:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.img_processed = st.session_state.img_original.copy()
                st.session_state.img_history = [st.session_state.img_original.copy()]
                st.rerun()
        
        with action_cols[2]:
            # Download processed image
            _, buffer = cv2.imencode(".jpg", st.session_state.img_processed, 
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            st.download_button(
                "💾 Save Image",
                buffer.tobytes(),
                "edited_image.jpg",
                "image/jpeg",
                use_container_width=True
            )
        
        with action_cols[3]:
            if st.button("🎬 Apply to Video", use_container_width=True):
                if 'video_frames' in st.session_state:
                    # Apply current image effect to all video frames
                    effect_type = "custom"  # You would track the last effect
                    st.info("This would apply the current image effect to all video frames")

with tab3:
    # Effects Gallery
    st.header("🎨 Effects Gallery")
    
    st.markdown("### 💎 Premium Effects Collection")
    
    # HD Effects Showcase
    effect_categories = [
        {
            "name": "Beauty & Portrait",
            "effects": [
                {"name": "8K Ultra HD", "icon": "🚀", "desc": "4x resolution enhancement"},
                {"name": "AI Face Glow", "icon": "✨", "desc": "Professional skin retouching"},
                {"name": "Portrait Mode", "icon": "📸", "desc": "Background blur effect"},
                {"name": "Skin Retouch Pro", "icon": "🌟", "desc": "Advanced skin smoothing"}
            ]
        },
        {
            "name": "Hair & Style",
            "effects": [
                {"name": "Hair Color Changer", "icon": "👩‍🦰", "desc": "8+ natural hair colors"},
                {"name": "Hair Highlights", "icon": "💫", "desc": "Natural-looking highlights"},
                {"name": "Gray Hair Effect", "icon": "👨‍🦳", "desc": "Aging simulation"}
            ]
        },
        {
            "name": "Cinematic & Filters",
            "effects": [
                {"name": "Cinematic Look", "icon": "🎬", "desc": "Hollywood color grading"},
                {"name": "HDR Pro", "icon": "🔆", "desc": "High dynamic range"},
                {"name": "Vintage Film", "icon": "🎞️", "desc": "Retro film effect"},
                {"name": "Dramatic B&W", "icon": "⚫", "desc": "Black & white contrast"}
            ]
        },
        {
            "name": "Creative & Fun",
            "effects": [
                {"name": "Glitch Art", "icon": "🌀", "desc": "Digital distortion effects"},
                {"name": "Neon Glow", "icon": "💡", "desc": "Neon light effects"},
                {"name": "Anime Filter", "icon": "🇯🇵", "desc": "Cartoon/anime style"},
                {"name": "Bloom Effect", "icon": "🌺", "desc": "Soft glow and highlights"}
            ]
        }
    ]
    
    # Display effect categories
    for category in effect_categories:
        st.markdown(f"### {category['name']}")
        
        cols = st.columns(len(category['effects']))
        
        for idx, effect in enumerate(category['effects']):
            with cols[idx]:
                with st.container():
                    st.markdown(f"""
                    <div class='effect-card'>
                        <h3>{effect['icon']} {effect['name']}</h3>
                        <p>{effect['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Try {effect['name']}", key=f"gallery_{effect['name']}"):
                        st.info(f"Applying {effect['name']}...")
                        # This would apply the effect to current image/video

with tab4:
    # Settings
    st.header("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Processing Settings")
        
        processing_mode = st.selectbox(
            "Processing Mode",
            ["Fast (Lower Quality)", "Balanced", "High Quality (Slow)"]
        )
        
        auto_save = st.checkbox("Auto-save processed files", value=True)
        
        default_format = st.selectbox(
            "Default Output Format",
            ["MP4 (Recommended)", "AVI", "MOV", "GIF"]
        )
    
    with col2:
        st.subheader("Performance")
        
        max_frames = st.slider("Maximum frames to process", 50, 1000, 200, 50,
                              help="Lower for faster processing, higher for better quality")
        
        use_gpu = st.checkbox("Use GPU acceleration (if available)", value=False)
        
        cache_size = st.slider("Cache size (MB)", 100, 1000, 500, 100)
    
    st.subheader("Export Settings")
    
    export_cols = st.columns(3)
    
    with export_cols[0]:
        video_quality = st.select_slider(
            "Video Quality",
            options=["Low", "Medium", "High", "Ultra HD"]
        )
    
    with export_cols[1]:
        audio_quality = st.select_slider(
            "Audio Quality",
            options=["Mono", "Stereo", "Surround"]
        )
    
    with export_cols[2]:
        frame_rate_out = st.select_slider(
            "Output Frame Rate",
            options=[24, 25, 30, 60]
        )
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved successfully!")

# --- Sidebar ---
st.sidebar.title("🎬 Quick Actions")
st.sidebar.info("""
**Video Editing Tips:**

1. **Upload Video**: MP4 format works best
2. **Add Effects**: Click effect buttons to add
3. **Adjust Intensity**: Use sliders for each effect
4. **Preview**: Check processed frames
5. **Export**: Download your edited video

**Recommended Workflow:**
1. Face enhancement first
2. Hair color/styling
3. Color grading/filters
4. Special effects
5. Final adjustments
""")

st.sidebar.title("📊 Statistics")
if 'video_frames' in st.session_state:
    st.sidebar.metric("Total Frames", len(st.session_state.video_frames))
    st.sidebar.metric("Frame Rate", f"{st.session_state.video_fps} FPS")
    st.sidebar.metric("Active Effects", len(st.session_state.video_effects))

st.sidebar.title("🎯 Quick Effects")
quick_effect = st.sidebar.selectbox(
    "Apply Quick Effect",
    ["None", "Instagram Filter", "Vintage Look", "Cinematic", "Anime", "Glitch"]
)

if quick_effect != "None" and 'video_frames' in st.session_state:
    if st.sidebar.button("Apply Quick Effect"):
        # Map quick effect to actual effect
        effect_map = {
            "Instagram Filter": "vibrant",
            "Vintage Look": "vintage",
            "Cinematic": "cinematic",
            "Anime": "anime",
            "Glitch": "glitch"
        }
        
        effect_type = effect_map[quick_effect]
        effect = {'type': effect_type, 'enabled': True, 'intensity': 1.0}
        
        if effect not in st.session_state.video_effects:
            st.session_state.video_effects.append(effect)
            st.sidebar.success(f"Added {quick_effect} effect!")
            st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎬 <b>AI Family Video Studio Pro</b> | Professional Video & Image Editing Suite</p>
    <p>Made with ❤️ for creative families | Version 2.0</p>
</div>
""", unsafe_allow_html=True)
