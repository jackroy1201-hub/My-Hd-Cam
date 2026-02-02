import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import io, time

st.set_page_config(page_title="Roman HD Studio Pro FINAL", layout="centered")

st.markdown("""
<style>
.stButton>button{
width:100%;height:3.6em;border-radius:14px;
font-weight:bold;border:1px solid #ddd;margin-bottom:6px;
}
.stButton>button:hover{background:#fff3e0;border-color:#ff9800}
.auto button{
background:linear-gradient(135deg,#FFD700,#FF8C00);
font-size:1.1em;border:none;
}
</style>
""", unsafe_allow_html=True)

st.title("📸 Roman HD Studio Pro FINAL")

pic = st.file_uploader("Upload Photo",type=["jpg","png","jpeg"])

def progress(txt):
    bar=st.progress(0)
    for i in range(100):
        time.sleep(0.004)
        bar.progress(i+1)
    st.success(txt)

if pic:
    original = Image.open(pic).convert("RGB")

    if "img" not in st.session_state:
        st.session_state.img = original

    st.markdown('<div class="auto">',unsafe_allow_html=True)
    if st.button("🤖 AI AUTO HD BEAUTY"):
        progress("AI HD Done!")

        img=np.array(original)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        smooth=cv2.bilateralFilter(img,15,90,90)
        glow=cv2.GaussianBlur(smooth,(0,0),3)
        hd=cv2.addWeighted(smooth,1.6,glow,-0.6,0)
        hd=cv2.detailEnhance(hd,15,0.15)

        final=Image.fromarray(cv2.cvtColor(hd,cv2.COLOR_BGR2RGB))
        final=ImageEnhance.Sharpness(final).enhance(1.8)
        final=ImageEnhance.Color(final).enhance(1.25)
        final=ImageEnhance.Brightness(final).enhance(1.05)

        st.session_state.img=final
    st.markdown('</div>',unsafe_allow_html=True)

    st.write("### 🎨 Filters")

    c1,c2=st.columns(2)

    with c1:
        if st.button("🌈 Vivid HD"):
            progress("Vivid Done")
            img=ImageEnhance.Color(original).enhance(2)
            img=ImageEnhance.Contrast(img).enhance(1.3)
            st.session_state.img=img

        if st.button("🎵 TikTok Glow"):
            progress("Glow Done")
            arr=np.array(original)
            blur=cv2.GaussianBlur(arr,(9,9),0)
            soft=cv2.addWeighted(arr,1.3,blur,-0.3,0)
            st.session_state.img=Image.fromarray(soft)

        if st.button("🌟 Model Look"):
            progress("Model Done")
            img=ImageEnhance.Sharpness(original).enhance(1.7)
            st.session_state.img=ImageEnhance.Color(img).enhance(1.3)

        if st.button("💄 Beauty Smooth"):
            progress("Beauty Done")
            arr=np.array(original)
            smooth=cv2.bilateralFilter(arr,20,100,100)
            st.session_state.img=Image.fromarray(smooth)

    with c2:
        if st.button("👻 Snapchat Clean"):
            progress("Clean Done")
            img=ImageOps.autocontrast(original)
            st.session_state.img=ImageEnhance.Sharpness(img).enhance(2.3)

        if st.button("🍏 iPhone HD"):
            progress("iPhone Done")
            img=ImageEnhance.Sharpness(original).enhance(2.6)
            st.session_state.img=ImageEnhance.Contrast(img).enhance(1.15)

        if st.button("🎭 Cinematic"):
            progress("Cinematic Done")
            st.session_state.img=ImageEnhance.Contrast(original).enhance(1.9)

        if st.button("💇 Hair Day Shine"):
            progress("Hair Shine Done")
            img=ImageEnhance.Sharpness(original).enhance(2)
            img=ImageEnhance.Brightness(img).enhance(1.05)
            st.session_state.img=img

    st.write("### 💇 Hair Colour Studio")

    color=st.selectbox("Select Hair Color",
        ["None","Brown","Golden","Red","Blue Black","Purple","Blonde"])

    if color!="None":
        arr=np.array(original)
        hsv=cv2.cvtColor(arr,cv2.COLOR_RGB2HSV)

        mask=cv2.inRange(hsv,(0,20,20),(180,255,120))
        mask=cv2.GaussianBlur(mask,(25,25),0)

        shades={
            "Brown":(42,42,165),
            "Golden":(0,215,255),
            "Red":(60,20,220),
            "Blue Black":(20,20,60),
            "Purple":(140,0,140),
            "Blonde":(180,220,255)
        }

        col=np.full(arr.shape,shades[color],dtype=np.uint8)
        alpha=mask/255

        recolor=arr*(1-alpha[...,None])+col*alpha[...,None]
        recolor=recolor.astype(np.uint8)

        st.session_state.img=Image.fromarray(recolor)

    st.write("### 📸 Portrait Mode Pro")

    depth=st.slider("Depth Strength",0,40,0)

    if depth>0:
        arr=np.array(original)
        h,w,_=arr.shape
        blur=cv2.GaussianBlur(arr,(depth*2+1,depth*2+1),0)

        mask=np.zeros((h,w),np.uint8)
        cv2.ellipse(mask,(w//2,h//2),(w//3,h//1.8),0,0,360,255,-1)
        mask=cv2.GaussianBlur(mask,(201,201),0)/255

        out=arr*mask[...,None]+blur*(1-mask[...,None])
        out=out.astype(np.uint8)

        st.session_state.img=Image.fromarray(out)

    st.image(st.session_state.img,use_container_width=True)

    buf=io.BytesIO()
    st.session_state.img.save(buf,"JPEG",quality=100,subsampling=0)

    st.download_button("📥 Save FULL HD",buf.getvalue(),
                       "Roman_Studio_Final.jpg","image/jpeg")

    if st.button("🔄 Reset"):
        st.session_state.img=original
        st.rerun()

else:
    st.info("Upload image to start editing")
