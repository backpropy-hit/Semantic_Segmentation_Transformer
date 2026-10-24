import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from transformers import UperNetForSemanticSegmentation

MODEL_ID = "hit100issue/semantic-segmentation-vit"   # replaces MODEL_DIR = "best_model"
CLASS_NAMES = ["other", "wall", "road", "vegetation", "vehicle", "roof", "water"]
PALETTE = np.array([(80, 80, 80), (230, 160, 60), (190, 190, 200), (50, 160, 60),
                    (220, 50, 50), (150, 60, 40), (50, 110, 220)], dtype=np.uint8)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Drone Image Segmentation", layout="wide")


@st.cache_resource(show_spinner="Loading Swin model...")
def load_model() -> UperNetForSemanticSegmentation:
    return UperNetForSemanticSegmentation.from_pretrained(MODEL_ID).to(DEVICE).eval()


@torch.no_grad()
def segment(model, image: Image.Image, long_side: int, flip_tta: bool):
    w, h = image.size
    scale = min(1.0, long_side / max(w, h))
    image = image.resize((round(w * scale), round(h * scale)), Image.Resampling.BILINEAR)
    x = TF.normalize(TF.to_tensor(image), MEAN, STD).unsqueeze(0).to(DEVICE)
    with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
        probs = model(pixel_values=x).logits.float().softmax(dim=1)
        if flip_tta:
            flipped = model(pixel_values=x.flip(-1)).logits.float().softmax(dim=1)
            probs = (probs + flipped.flip(-1)) / 2
    return image, probs.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


st.title("Drone Image Semantic Segmentation")
st.caption("Swin-T + UperNet, fine-tuned on the VDD dataset (7 classes).")

with st.sidebar:
    st.header("Settings")
    long_side = st.select_slider(
        "Processing resolution (long side)", options=[512, 768, 1024], value=1024,
        help="The model was trained at 1024. Lower is faster on CPU but less accurate.",
    )
    opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.5, 0.05)
    flip_tta = st.checkbox("Flip test-time augmentation", value=False, help="Slightly better, about 2x slower.")
    st.markdown("**Legend**")
    for name, (r, g, b) in zip(CLASS_NAMES, PALETTE):
        st.markdown(
            f"<span style='background:rgb({r},{g},{b});color:white;padding:2px 10px;border-radius:4px'>{name}</span>",
            unsafe_allow_html=True,
        )

upload = st.file_uploader("Upload a drone image", type=["jpg", "jpeg", "png"])

if upload is not None:
    image = ImageOps.exif_transpose(Image.open(upload)).convert("RGB")  # fixes phone-photo rotation
    with st.spinner("Segmenting..."):
        resized, mask = segment(load_model(), image, long_side, flip_tta)

    color_mask = PALETTE[mask]
    overlay = ((1 - opacity) * np.asarray(resized) + opacity * color_mask).astype(np.uint8)

    col1, col2, col3 = st.columns(3)
    col1.image(resized, caption="Input", use_container_width=True)
    col2.image(overlay, caption="Overlay", use_container_width=True)
    col3.image(color_mask, caption="Segmentation mask", use_container_width=True)

    st.subheader("Pixel share per class (%)")
    share = 100 * np.bincount(mask.ravel(), minlength=len(CLASS_NAMES)) / mask.size
    st.bar_chart(pd.Series(share, index=CLASS_NAMES))
else:
    st.info("Upload an image to see the segmentation.")