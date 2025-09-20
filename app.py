import streamlit as st
from image_palette import extract_palette_from_image, create_palette_image

# Streamlit app title
st.title("🎨 Image Color Palette Extractor")

st.markdown(
    "Upload an image and extract its dominant colors (in HEX, RGB, HSL)."
)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Number of colors to extract
        n_colors = st.slider("Number of colors", min_value=3, max_value=15, value=8)

        # Extract palette
        result = extract_palette_from_image(uploaded_file, n_colors=n_colors)
        colors = result["colors"]
        image = result["image"]

        # Show uploaded image
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

        # Show palette image
        st.subheader("Extracted Color Palette")
        palette_img = create_palette_image(colors)
        st.image(palette_img, use_container_width=False)

        # Show color details
        st.subheader("Color Details")
        for i, color in enumerate(colors, start=1):
            hex_color = color["hex"]
            rgb_color = color["rgb"]
            hsl_color = color["hsl"]

            st.markdown(
                f"""
                <div style="display:flex; align-items:center; margin-bottom:8px;">
                    <div style="width:40px; height:40px; background:{hex_color}; border:1px solid #ccc; margin-right:10px;"></div>
                    <div>
                        <b>Color {i}</b><br>
                        HEX: {hex_color}<br>
                        RGB: {rgb_color}<br>
                        HSL: {hsl_color}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("👆 Upload an image to get started.")
