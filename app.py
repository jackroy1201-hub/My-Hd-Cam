import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import io, tempfile, os
from moviepy.editor import VideoFileClip

# ---------------- UI ----------------

st.set_page_config("Roman Studio Pro", layout="wide", page_icon="🎨")

st.markdown("""
<style>
.main-header{padding:1.5rem;text-align:center;
background:linear-gradient(135deg,#1e3c72,#2a5298);
color:white;border-radius:15px;margin-bottom:20px}
.stButton>button{width:100%;border-radius:12px;font-weight:bold;height:3.2em}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎨 Roman Studio Pro</h1><p>Full HD & 4K AI Photo + Video Studio</p></div>', unsafe_allow_html=True)

# ---------------- Utils ----------------

def to_np(img): 
    return np.array(img.convert("RGB"))

def natural_sharp(img):
    n = to_np(img)
    blur = cv2.GaussianBlur(n,(0,0),1.1)
    sharp = cv2.addWeighted(n,1.12,blur,-0.12,0)
    return Image.fromarray(np.clip(sharp,0,255).astype(np.uint8))

def clean_hd(img):
    n = to_np(img)
    den = cv2.fastNlMeansDenoisingColored(n,None,6,6,7,21)
    blur = cv2.GaussianBlur(den,(0,0),1)
    hd = cv2.addWeighted(den,1.1,blur,-0.1,0)
    return Image.fromarray(hd)

def natural_hdr(img):
    n = to_np(img)
    det = cv2.detailEnhance(n, sigma_s=6, sigma_r=0.08)
    blend = cv2.addWeighted(n,0.6,det,0.4,0)
    return Image.fromarray(blend)

# ---------------- Sidebar ----------------

with st.sidebar:
    st.title("⚙️ Control Panel")
    quality = st.slider("Export HD Quality",80,100,100)
    st.success("All Systems Active")

# ---------------- Upload ----------------

col1,col2 = st.columns([1,2])

with col1:
    pic = st.file_uploader("Upload Image",["jpg","png","jpeg","webp"])
    if pic:
        img = Image.open(pic).convert("RGB")
        if "img" not in st.session_state:
            st.session_state.img = img
            st.session_state.org = img

with col2:
    if pic:
        c1,c2 = st.columns(2)
        c1.image(st.session_state.org,caption="Original",use_container_width=True)
        c2.image(st.session_state.img,caption="Edited HD",use_container_width=True)

# ================= PHOTO EDITOR =================

if pic:

    tabs = st.tabs(["✨ AI Magic","💄 Beauty","👔 Hair","🎬 Social","🎞 Pro","🌿 Natural HD Pro","🎥 Video Editor"])

    # ---- AI MAGIC ----

    with tabs[0]:

        if st.button("💎 Full HD Clean Boost"):
            st.session_state.img = clean_hd(st.session_state.img)
            st.rerun()

        if st.button("📷 Natural DSLR Sharp"):
            st.session_state.img = natural_sharp(st.session_state.img)
            st.rerun()

        if st.button("🌟 Natural HDR"):
            st.session_state.img = natural_hdr(st.session_state.img)
            st.rerun()

    # ---- BEAUTY ----

    with tabs[1]:

        smooth = st.slider("Skin Smooth",3,20,8)
        bright = st.slider("Brightness",0.8,1.5,1.05)

        if st.button("💄 Apply Natural Beauty"):
            n = to_np(st.session_state.img)
            clean = cv2.bilateralFilter(n,smooth,40,40)
            img2 = Image.fromarray(clean)
            img2 = ImageEnhance.Brightness(img2).enhance(bright)
            st.session_state.img = img2
            st.rerun()

    # ---- HAIR ----

    with tabs[2]:

        hair_colors={
            "Black":[20,20,20],
            "Brown":[110,70,40],
            "Golden":[190,150,60]
        }

        h=st.selectbox("Hair Color",hair_colors)

        if st.button("💇 Apply Hair Color"):
            n=to_np(st.session_state.img)
            gray=cv2.cvtColor(n,cv2.COLOR_RGB2GRAY)
            mask=cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV)[1]
            mask=cv2.GaussianBlur(mask,(21,21),0)/255
            res=(n*(1-mask[:,:,None]*0.4)+np.array(hair_colors[h])*(mask[:,:,None]*0.4)).astype(np.uint8)
            st.session_state.img=Image.fromarray(res)
            st.rerun()

    # ---- SOCIAL ----

    with tabs[3]:

        if st.button("📱 iPhone Look"):
            img=st.session_state.img
            img=ImageEnhance.Color(img).enhance(1.1)
            img=ImageEnhance.Sharpness(img).enhance(1.3)
            st.session_state.img=img
            st.rerun()

        if st.button("✨ TikTok Glow"):
            n=to_np(st.session_state.img)
            glow=cv2.GaussianBlur(n,(25,25),0)
            st.session_state.img=Image.fromarray(cv2.addWeighted(n,0.8,glow,0.2,0))
            st.rerun()

    # ---- PRO ----

    with tabs[4]:

        if st.button("🖤 Classic B&W"):
            st.session_state.img=ImageOps.grayscale(st.session_state.img)
            st.rerun()

        if st.button("🌅 Golden Hour"):
            st.session_state.img=ImageEnhance.Color(st.session_state.img).enhance(1.4)
            st.rerun()

    # ---- NATURAL HD ----

    with tabs[5]:

        if st.button("🌿 Clean Camera Look"):
            n=to_np(st.session_state.img)
            soft=cv2.bilateralFilter(n,5,35,35)
            st.session_state.img=Image.fromarray(soft)
            st.rerun()

        if st.button("✨ Natural Glow"):
            img=st.session_state.img
            img=ImageEnhance.Brightness(img).enhance(1.03)
            img=ImageEnhance.Sharpness(img).enhance(1.05)
            st.session_state.img=img
            st.rerun()

    # ================= VIDEO EDITOR =================

    with tabs[6]:

        st.subheader("📱 TikTok Full HD + Beauty + Voice Safe")

        video = st.file_uploader("Upload Video",["mp4","mov","avi"])

        if video:

            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(video.read())

            # ---- TikTok HD Beauty ----

            if st.button("✨ TikTok Beauty HD (Keep Voice)"):

                cap=cv2.VideoCapture(temp.name)
                fps=cap.get(cv2.CAP_PROP_FPS)

                temp_video="temp_video.mp4"
                final="TikTok_Beauty_HD.mp4"

                out=cv2.VideoWriter(
                    temp_video,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (1080,1920)
                )

                while True:
                    ret,frame=cap.read()
                    if not ret: break

                    frame=cv2.resize(frame,(1080,1920))
                    smooth=cv2.bilateralFilter(frame,7,40,40)
                    glow=cv2.GaussianBlur(smooth,(25,25),0)
                    beauty=cv2.addWeighted(smooth,0.85,glow,0.15,0)
                    beauty=cv2.convertScaleAbs(beauty,1.05,6)

                    out.write(beauty)

                cap.release()
                out.release()

                orig=VideoFileClip(temp.name)
                new=VideoFileClip(temp_video)

                final_clip=new.set_audio(orig.audio)
                final_clip.write_videofile(final,codec="libx264",audio_codec="aac",verbose=False,logger=None)

                st.download_button("📥 Download TikTok Beauty HD",open(final,"rb"),"TikTok_Beauty_HD.mp4")

            # ---- 4K DSLR ----

            if st.button("🎬 Convert to 4K DSLR HD"):

                cap=cv2.VideoCapture(temp.name)
                fps=cap.get(cv2.CAP_PROP_FPS)

                out_path="Roman_4K_DSLR.mp4"

                out=cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (3840,2160)
                )

                while True:
                    ret,frame=cap.read()
                    if not ret: break

                    frame=cv2.resize(frame,(3840,2160),interpolation=cv2.INTER_CUBIC)
                    frame=cv2.detailEnhance(frame,5,0.07)

                    blur=cv2.GaussianBlur(frame,(0,0),1)
                    frame=cv2.addWeighted(frame,1.08,blur,-0.08,0)

                    out.write(frame)

                cap.release()
                out.release()

                st.download_button("📥 Download 4K DSLR Video",open(out_path,"rb"),"Roman_4K_DSLR.mp4")

    # ---------------- Download Image ----------------

    st.markdown("---")
    b1,b2=st.columns(2)

    with b1:
        buf=io.BytesIO()
        st.session_state.img.save(buf,"JPEG",quality=quality,subsampling=0)
        st.download_button("📥 Download HD Photo",buf.getvalue(),"Roman_HD.jpg")

    with b2:
        if st.button("🔄 Reset Original"):
            st.session_state.img=st.session_state.org
            st.rerun()

st.markdown("<center><p style='color:gray;'>Roman Studio Pro 2026 – All AI Systems Active</p></center>",unsafe_allow_html=True)
