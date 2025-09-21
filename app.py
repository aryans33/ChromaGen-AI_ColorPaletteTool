import streamlit as st
from dotenv import load_dotenv
import os
from io import BytesIO
from PIL import Image, ImageEnhance

# Local imports
from image_palette import extract_palette_from_image, create_palette_image, palette_payload_from_colors
from text_palette import (
    text_to_palette_structured_payload,
    text_to_palette_options_structured,
)

# Load environment variables
load_dotenv()

# ---- Streamlit UI ----
st.set_page_config(page_title="ChromaGen – AI Color Palette Generator", layout="wide")
st.title("ChromaGen – AI Color Palette Generator")

mode = st.sidebar.radio("Mode", ["Image to Palette", "Text to Palette"])

# ------------------ IMAGE TO PALETTE ------------------
if mode == "Image to Palette":
    st.header("📷 Generate Palette from Image")

    n_colors_img = st.sidebar.slider("Number of colors", min_value=3, max_value=12, value=8, step=1)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    def _augment_variants(img: Image.Image) -> list[BytesIO]:
        # produce few lightweight variants for diverse options
        variants = []
        ops = [
            ("base", img),
            ("color_up", ImageEnhance.Color(img).enhance(1.15)),
            ("contrast_up", ImageEnhance.Contrast(img).enhance(1.1)),
            ("sharp_up", ImageEnhance.Sharpness(img).enhance(1.1)),
        ]
        for name, im in ops:
            buf = BytesIO()
            im.save(buf, format="PNG")
            buf.seek(0)
            variants.append(buf)
        return variants

    if uploaded_file:
        try:
            # Single palette preview
            result = extract_palette_from_image(uploaded_file, n_colors=n_colors_img)
            colors = result["colors"]
            st.image(result["image"], caption="Uploaded Image", width="stretch")
            st.image(create_palette_image(colors, width=800, height=120), caption="Generated Palette", width="stretch")

            # NEW: structured view (primary, secondary, accent) with copy blocks
            payload_img = palette_payload_from_colors(colors)
            structured_img = payload_img["palette"]
            copy_blocks_img = payload_img["copy"]

            st.subheader("Primary")
            st.image(create_palette_image(structured_img["primary"], width=800, height=80), caption=None, width="stretch")
            st.code(copy_blocks_img["primary"]["hex"], language="text")
            st.code(copy_blocks_img["primary"]["rgb"], language="text")
            st.code(copy_blocks_img["primary"]["hsl"], language="text")

            st.subheader("Secondary")
            st.image(create_palette_image(structured_img["secondary"], width=800, height=80), caption=None, width="stretch")
            st.code(copy_blocks_img["secondary"]["hex"], language="text")
            st.code(copy_blocks_img["secondary"]["rgb"], language="text")
            st.code(copy_blocks_img["secondary"]["hsl"], language="text")

            st.subheader("Accent")
            st.image(create_palette_image(structured_img["accent"], width=800, height=80), caption=None, width="stretch")
            st.code(copy_blocks_img["accent"]["hex"], language="text")
            st.code(copy_blocks_img["accent"]["rgb"], language="text")
            st.code(copy_blocks_img["accent"]["hsl"], language="text")

            with st.expander("Structured JSON"):
                st.code(payload_img["json"], language="json")

            # Multiple options
            st.subheader("More options")
            img_obj = Image.open(uploaded_file).convert("RGB")
            variants = _augment_variants(img_obj)
            option_palettes = []
            cols = st.columns(2)
            for i, var in enumerate(variants[:4]):
                opt = extract_palette_from_image(var, n_colors=n_colors_img)
                option_palettes.append(opt["colors"])
                with cols[i % 2]:
                    st.image(create_palette_image(opt["colors"], width=800, height=120), caption=f"Option {i+1}", width="stretch")
                    if st.button(f"Use Option {i+1}", key=f"use_img_opt_{i+1}"):
                        st.session_state["selected_palette_image"] = opt["colors"]

            if "selected_palette_image" in st.session_state:
                st.subheader("✅ Selected Palette (Image)")
                sel_img = st.session_state["selected_palette_image"]
                st.image(create_palette_image(sel_img, width=800, height=120), caption="Your selection", width="stretch")

                # NEW: structured view for selected option
                payload_sel = palette_payload_from_colors(sel_img)
                structured_sel = payload_sel["palette"]
                copy_sel = payload_sel["copy"]

                st.markdown("Primary")
                st.image(create_palette_image(structured_sel["primary"], width=800, height=80), caption=None, width="stretch")
                st.code(copy_sel["primary"]["hex"], language="text")
                st.code(copy_sel["primary"]["rgb"], language="text")
                st.code(copy_sel["primary"]["hsl"], language="text")

                st.markdown("Secondary")
                st.image(create_palette_image(structured_sel["secondary"], width=800, height=80), caption=None, width="stretch")
                st.code(copy_sel["secondary"]["hex"], language="text")
                st.code(copy_sel["secondary"]["rgb"], language="text")
                st.code(copy_sel["secondary"]["hsl"], language="text")

                st.markdown("Accent")
                st.image(create_palette_image(structured_sel["accent"], width=800, height=80), caption=None, width="stretch")
                st.code(copy_sel["accent"]["hex"], language="text")
                st.code(copy_sel["accent"]["rgb"], language="text")
                st.code(copy_sel["accent"]["hsl"], language="text")

                with st.expander("Structured JSON (Selected)"):
                    st.code(payload_sel["json"], language="json")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ------------------ TEXT TO PALETTE ------------------
elif mode == "Text to Palette":
    st.header("📝 Generate Palette from Text Prompt")

    prompt = st.text_input("Describe your palette (e.g., 'barbie', 'sunset pastel', 'forest earthy')")
    generate = st.button("Generate Palette", type="primary")

    if generate:
        if prompt.strip():
            try:
                # Structured single result (same style as image mode)
                payload = text_to_palette_structured_payload(prompt, n_colors=6)
                structured = payload["palette"]
                copy_blocks = payload["copy"]
                flat = structured["primary"] + structured["secondary"] + structured["accent"]

                st.image(create_palette_image(flat, width=800, height=120), caption="Generated Palette", width="stretch")

                st.subheader("Primary")
                st.image(create_palette_image(structured["primary"], width=800, height=80), caption=None, width="stretch")
                st.code(copy_blocks["primary"]["hex"], language="text")
                st.code(copy_blocks["primary"]["rgb"], language="text")
                st.code(copy_blocks["primary"]["hsl"], language="text")

                st.subheader("Secondary")
                st.image(create_palette_image(structured["secondary"], width=800, height=80), caption=None, width="stretch")
                st.code(copy_blocks["secondary"]["hex"], language="text")
                st.code(copy_blocks["secondary"]["rgb"], language="text")
                st.code(copy_blocks["secondary"]["hsl"], language="text")

                st.subheader("Accent")
                st.image(create_palette_image(structured["accent"], width=800, height=80), caption=None, width="stretch")
                st.code(copy_blocks["accent"]["hex"], language="text")
                st.code(copy_blocks["accent"]["rgb"], language="text")
                st.code(copy_blocks["accent"]["hsl"], language="text")

                with st.expander("Structured JSON"):
                    st.code(payload["json"], language="json")

                # More options (same pattern as image mode)
                st.subheader("More options")
                opts = text_to_palette_options_structured(prompt, n_colors=6, n_options=4)
                cols = st.columns(2)
                for i, sp in enumerate(opts):
                    flat_opt = sp["primary"] + sp["secondary"] + sp["accent"]
                    with cols[i % 2]:
                        st.image(create_palette_image(flat_opt, width=800, height=120), caption=f"Option {i+1}", width="stretch")
                        if st.button(f"Use Option {i+1}", key=f"use_txt_opt_{i+1}"):
                            st.session_state["selected_palette_text"] = sp

                if "selected_palette_text" in st.session_state:
                    sel = st.session_state["selected_palette_text"]
                    st.subheader("✅ Selected Palette (Text)")
                    flat_sel = sel["primary"] + sel["secondary"] + sel["accent"]
                    st.image(create_palette_image(flat_sel, width=800, height=120), caption="Your selection", width="stretch")
            except Exception as e:
                st.error(f"Error generating palette: {str(e)}")
        else:
            st.warning("Please enter a text description.")