import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
import mediapipe as mp
import io

# --- 1. پیج سیٹ اپ اور سیکیورٹی ---
st.set_page_config(page_title="Family Secure HD Camera", layout="centered")

if 'user_db' not in st.session_state:
    st.session_state['user_db'] = {"Admin": "12@24", "Family": "4590$"}

@st.cache_resource
def load_models():
    mp_selfie = mp.solutions.selfie_segmentation
    return mp_selfie.SelfieSegmentation(model_selection=1)

selfie_seg = load_models()

if 'auth' not in st.session_state: st.session_state['auth'] = False
if 'page' not in st.session_state: st.session_state['page'] = "Home"

# --- 2. لاگ ان سسٹم ---
def login():
    st.markdown("<h2 style='text-align: center;'>🔐 Secure Family Access</h2>", unsafe_allow_html=True)
    u_name = st.text_input("Username")
    u_pwd = st.text_input("Password", type="password")
    if st.button("Unlock Access", use_container_width=True):
        db = st.session_state['user_db']
        if u_name in db and u_pwd == db[u_name]:
            st.session_state['auth'] = True
            st.session_state['current_user'] = u_name
            st.rerun()
        else: st.error("غلط معلومات!")

# --- 3. مین ایپ ---
if not st.session_state['auth']:
    login()
else:
    c_t1, c_t2 = st.columns([0.8, 0.2])
    with c_t1:
        if st.button("🏠 Home"): st.session_state['page'] = "Home"; st.rerun()
    with c_t2:
        if st.button("🚪 Logout"): st.session_state['auth'] = False; st.rerun()

    if st.session_state['page'] == "Home":
        st.markdown(f"<h3 style='text-align: center;'>Welcome, {st.session_state['current_user']}</h3>", unsafe_allow_html=True)
        if st.button("📸 iPhone HD Pro Camera", use_container_width=True):
            st.session_state['page'] = "Camera"; st.rerun()

    elif st.session_state['page'] == "Camera":
        st.header("📸 iPhone HD Pro Camera")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            bright = st.slider("Brightness", 0.5, 2.0, 1.1)
            zoom_val = st.slider("Zoom", 1.0, 3.0, 1.0)
        with col_c2:
            mode = st.radio("Style:", ["Natural HD", "Portrait Blur"])

        img_file = st.camera_input("تصویر کھینچیں")
        if img_file:
            img = Image.open(img_file)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            h, w, _ = frame.shape
            if zoom_val > 1.0:
                new_h, new_w = int(h / zoom_val), int(w / zoom_val)
                start_h, start_w = (h - new_h) // 2, (w - new_w) // 2
                frame = cv2.resize(frame[start_h:start_h+new_h, start_w:start_w+new_w], (w, h), interpolation=cv2.INTER_CUBIC)
            if bright != 1.0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV); v = hsv[:,:,2]
                v = cv2.add(v, int((bright - 1.0) * 80)); hsv[:,:,2] = np.clip(v, 0, 255)
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            processed = cv2.bilateralFilter(frame, 9, 75, 75)
            if mode == "Portrait Blur":
                res = selfie_seg.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                condition = np.stack((res.segmentation_mask,) * 3, axis=-1) > 0.5
                processed = np.where(condition, processed, cv2.GaussianBlur(processed, (45, 45), 0))
            
            final_img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO(); final_img.save(buf, format="JPEG", subsampling=0, quality=100)
            st.image(final_img, caption="HD Result Ready")
            if st.download_button("📥 موبائل میں سیو کریں اور سرور صاف کریں", data=buf.getvalue(), file_name=f"HD_{int(time.time())}.jpg", mime="image/jpeg", use_container_width=True):
                st.success("ڈاؤن لوڈ مکمل! سرور صاف کیا جا رہا ہے..."); time.sleep(1); st.rerun()
