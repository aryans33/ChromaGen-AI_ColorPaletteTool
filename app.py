import streamlit as st
from dotenv import load_dotenv
import os

# Local imports
from image_palette import extract_palette_from_image, create_palette_image
from text_palette import text_to_palette

# Load environment variables
load_dotenv()

# ---- Streamlit UI ----
st.set_page_config(page_title="ChromaGen – AI Color Palette Generator", layout="wide")
st.title("ChromaGen – AI Color Palette Generator")

mode = st.sidebar.radio("Mode", ["Image to Palette", "Text to Palette (GAN)"])

# ------------------ IMAGE TO PALETTE ------------------
if mode == "Image to Palette":
    st.header("📷 Generate Palette from Image")

    n_colors_img = st.sidebar.slider("Number of colors", min_value=3, max_value=12, value=8, step=1)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            result = extract_palette_from_image(uploaded_file, n_colors=n_colors_img)
            colors = result["colors"]

            # replace deprecated use_column_width
            st.image(result["image"], caption="Uploaded Image", width="stretch")
            st.image(create_palette_image(colors, width=800, height=120), caption="Generated Palette", width="stretch")

            st.subheader("🎨 Extracted Colors")
            for c in colors:
                st.write(f"*HEX:* {c['hex']} | *RGB:* {c['rgb']} | *HSL:* {c['hsl']}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ------------------ TEXT TO PALETTE ------------------
elif mode == "Text to Palette (GAN)":
    st.header("📝 Generate Palette from Text Prompt")

    # No extra sliders. Deterministic and retrieval-augmented under the hood.
    prompt = st.text_input("Describe your palette (e.g., 'barbie', 'sunset pastel', 'forest earthy')")

    if st.button("Generate Palette", type="primary"):
        if prompt.strip():
            try:
                palette = text_to_palette(prompt, n_colors=6)
                st.image(create_palette_image(palette, width=800, height=120), caption="Generated Palette", width="stretch")

                st.subheader("🎨 Generated Colors")
                for c in palette:
                    st.write(f"*HEX:* {c['hex']} | *RGB:* {c['rgb']} | *HSL:* {c['hsl']}")
            except Exception as e:
                st.error(f"Error generating palette: {str(e)}")
        else:
            st.warning("Please enter a text description.")