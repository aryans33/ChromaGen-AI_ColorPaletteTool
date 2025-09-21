import streamlit as st
from dotenv import load_dotenv
import os
from io import BytesIO
from PIL import Image, ImageEnhance
import base64
import streamlit.components.v1 as components  # NEW: render full HTML

# Local imports
from image_palette import extract_palette_from_image, create_palette_image, palette_payload_from_colors
from text_palette import (
    text_to_palette_structured_payload,
    text_to_palette_options_structured,
    build_accessibility_report,
    simulate_palette_colors,
    # NEW imports
    export_css_vars,
    export_scss_vars,
    export_tailwind_config,
    export_palette_json,
    export_react_theme,
    export_aco,
    export_ase,
    flatten_structured,
)

# Load environment variables
load_dotenv()

# Add readable labels for simulation modes
CB_LABELS = {
    "protanopia": "Protanopia (red-blindness)",
    "deuteranopia": "Deuteranopia (green-blindness)",
    "tritanopia": "Tritanopia (blue-blindness)",
}

# ---- Streamlit UI ----
st.set_page_config(page_title="ChromaGen – AI Color Palette Generator", layout="wide")
st.title("ChromaGen – AI Color Palette Generator")

# NEW: Lightweight REST-like API (?api=1&prompt=...&n=6)
_qp = st.query_params
api_flag = _qp.get("api", "0")
if isinstance(api_flag, list):
	api_flag = api_flag[0]
if api_flag == "1":
	prompt = _qp.get("prompt", "")
	if isinstance(prompt, list):
		prompt = prompt[0]
	n_val = _qp.get("n", "6")
	if isinstance(n_val, list):
		n_val = n_val[0]
	try:
		n = int(n_val)
	except Exception:
		n = 6
	try:
		payload = text_to_palette_structured_payload(prompt, n_colors=n)
		st.json(payload["palette"])
	except Exception as e:
		st.json({"error": str(e)})
	st.stop()

def _light_css() -> str:
	return """
	<style>
	/* App + main containers */
	html, body, [data-testid="stAppViewContainer"], .stApp, .block-container {
		background-color: #ffffff !important;
		color: #111111 !important;
	}
	/* Header (top bar) */
	[data-testid="stHeader"],
	[data-testid="stHeader"] > div {
		background: #ffffff !important;
		border-bottom: 1px solid #e5e7eb !important;
	}
	/* Sidebar */
	[data-testid="stSidebar"] > div {
		background-color: #ffffff !important;
		border-right: 1px solid #e5e7eb !important;
	}
	h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown {
		color: #111111 !important;
	}
	/* Inputs */
	input, textarea, select,
	.stTextInput input, .stTextArea textarea,
	.stSelectbox div[role="combobox"],
	.stNumberInput input {
		background-color: #ffffff !important;
		color: #111111 !important;
		border-color: #e5e7eb !important;
	}

	/* File uploader dropzone + button */
	[data-testid="stFileUploaderDropzone"] {
		background-color: #ffffff !important;
		border: 1px dashed #d1d5db !important;
		color: #111111 !important;
	}
	[data-testid="stFileUploaderDropzone"] * { color: #111111 !important; }

	/* Browse files button (light) */
	[data-testid="stFileUploader"] button {
		background-color: #2563eb !important;
		color: #ffffff !important;
		border: 1px solid #1d4ed8 !important;
		border-radius: 6px !important;
	}
	/* Ensure nested elements are white too */
	[data-testid="stFileUploader"] button * { color: #ffffff !important; }
	[data-testid="stFileUploader"] button:hover { background-color: #1d4ed8 !important; }
	[data-testid="stFileUploader"] button:active { background-color: #1e40af !important; }

	/* Light mode buttons (fix "Use Option" and all blue buttons) */
	.stButton > button {
		background-color: #2563eb !important;
		color: #ffffff !important;
		border: 1px solid #1d4ed8 !important;
		border-radius: 8px !important;
	}
	/* Ensure nested spans/icons stay white */
	.stButton > button * { color: #ffffff !important; }
	.stButton > button:hover { background-color: #1d4ed8 !important; }
	.stButton > button:active { background-color: #1e40af !important; }

	/* CODE BLOCKS (force light backgrounds) */
	[data-testid="stCodeBlock"],
	[data-testid="stCodeBlock"] > div { background-color: #ffffff !important; }
	[data-testid="stCodeBlock"] pre, pre, code, .stMarkdown code {
		background-color: #f7f7f7 !important;
		color: #111111 !important;
		border: 1px solid #e5e7eb !important;
		border-radius: 8px !important;
	}
	[data-testid="stCodeBlock"] pre { padding: 12px 14px !important; }

	/* Buttons: ensure download buttons look like primary buttons */
	.stDownloadButton > button,
	[data-testid="stDownloadButton"] button {
		background-color: #2563eb !important;
		color: #ffffff !important;
		border: 1px solid #1d4ed8 !important;
		border-radius: 8px !important;
		width: 100% !important;            /* NEW: align width */
		justify-content: center !important;/* NEW: center text */
	}
	.stDownloadButton > button * { color:#ffffff !important; }
	[data-testid="stDownloadButton"] button:hover { background-color:#1d4ed8 !important; }
	[data-testid="stDownloadButton"] button:active { background-color:#1e40af !important; }

	/* Small badges for status (contrast) */
	.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; margin-right:6px; }
	.badge-pass { background: #DCFCE7; color:#166534; border:1px solid #86EFAC; }
	.badge-fail { background: #FEE2E2; color:#991B1B; border:1px solid #FCA5A5; }
	.preview-card { padding:12px; border:1px solid rgba(0,0,0,0.12); border-radius:12px; margin-bottom:12px; }
	.preview-bar { height:10px; border-radius:999px; margin-bottom:10px; opacity:0.9; }
	</style>
	"""

def _dark_css() -> str:
	return """
	<style>
	:root { --bg:#0e1117; --text:#e6e6e6; --card:#161b22; --code:#0f172a; --border:#30363d; --primary:#3b82f6; }
	html, body, [data-testid="stAppViewContainer"], .stApp, .block-container {
		background-color: var(--bg) !important; color: var(--text) !important;
	}
	[data-testid="stHeader"], [data-testid="stHeader"] > div {
		background: var(--bg) !important; border-bottom: 1px solid var(--border) !important;
	}
	[data-testid="stSidebar"] > div {
		background-color: var(--card) !important; border-right: 1px solid var(--border) !important;
	}
	h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown { color: var(--text) !important; }
	.stButton > button {
		background: var(--primary) !important;
		color: #ffffff !important;
		border: 1px solid transparent !important;
		border-radius: 6px !important;
	}
	/* Ensure nested spans/icons stay white */
	.stButton > button * { color: #ffffff !important; }

	/* File uploader button (dark) */
	[data-testid="stFileUploader"] button {
		background-color: var(--primary) !important;
		color: #ffffff !important;
		border: 1px solid transparent !important;
		border-radius: 6px !important;
	}
	/* Ensure nested elements are white too */
	[data-testid="stFileUploader"] button * { color: #ffffff !important; }
	[data-testid="stFileUploader"] button:hover { filter: brightness(0.9); }

	/* CODE BLOCKS (dark) */
	[data-testid="stCodeBlock"],
	[data-testid="stCodeBlock"] > div {
		background-color: var(--bg) !important;
	}
	[data-testid="stCodeBlock"] pre, pre, code, .stMarkdown code {
		background-color: var(--code) !important;
		color: var(--text) !important;
		border: 1px solid var(--border) !important;
		border-radius: 8px !important;
	}
	[data-testid="stCodeBlock"] pre {
		padding: 12px 14px !important;
	}

	/* Buttons: ensure download buttons look like primary buttons in dark mode */
	.stDownloadButton > button,
	[data-testid="stDownloadButton"] button {
		background: var(--primary) !important;
		color: #ffffff !important;
		border: 1px solid transparent !important;
		border-radius: 8px !important;
		width: 100% !important;            /* NEW: align width */
		justify-content: center !important;/* NEW: center text */
	}
	.stDownloadButton > button * { color:#ffffff !important; }
	[data-testid="stDownloadButton"] button:hover { filter: brightness(0.9); }

	/* Small badges for status (contrast) */
	.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; margin-right:6px; }
	.badge-pass { background: rgba(34,197,94,0.15); color:#86EFAC; border:1px solid rgba(34,197,94,0.35); }
	.badge-fail { background: rgba(239,68,68,0.15); color:#FCA5A5; border:1px solid rgba(239,68,68,0.35); }
	.preview-card { padding:12px; border:1px solid var(--border); border-radius:12px; margin-bottom:12px; }
	.preview-bar { height:10px; border-radius:999px; margin-bottom:10px; opacity:0.9; }
	</style>
	"""

# Use a placeholder so CSS is fully replaced on toggle
_theme_css_slot = st.empty()
dark_mode = st.sidebar.toggle("Dark mode", value=st.session_state.get("dark_mode", False))
st.session_state["dark_mode"] = dark_mode
_theme_css_slot.markdown(_dark_css() if dark_mode else _light_css(), unsafe_allow_html=True)

mode = st.sidebar.radio("Mode", ["Image to Palette", "Text to Palette"])

def _rgba_str(rgb, alpha=0.12):
	(r, g, b) = rgb
	return f"rgba({int(r)}, {int(g)}, {int(b)}, {alpha})"

def _option_bg_from_palette(palette_colors):
	# Use the lightest swatch for background (soft tint)
	def lum(c):
		r, g, b = c["rgb"]
		return 0.2126*r + 0.7152*g + 0.0722*b
	lightest = max(palette_colors, key=lum)
	return _rgba_str(lightest["rgb"], 0.12)

def _palette_to_base64_image(colors, width=800, height=120):
	img = create_palette_image(colors, width=width, height=height)
	buf = BytesIO()
	img.save(buf, format="PNG")
	return base64.b64encode(buf.getvalue()).decode("utf-8")

def render_option_card(palette_colors, caption: str, use_key: str):
	bg = _option_bg_from_palette(palette_colors)
	img_b64 = _palette_to_base64_image(palette_colors, width=800, height=120)
	st.markdown(
		f"""
		<div style="
			background:{bg};
			border: 1px solid rgba(0,0,0,0.08);
			border-radius:16px;
			padding:16px 16px 8px 16px;
			margin-bottom:12px;">
			<img src="data:image/png;base64,{img_b64}" style="width:100%;border-radius:12px;" />
			<div style="text-align:center;opacity:0.8;margin-top:8px;">{caption}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)
	return st.button(f"Use {caption}", key=use_key)

# ---------------------- NEW: Modular UI helpers ----------------------

def _hex_only(colors):
	return [c["hex"].upper() for c in colors]

def _contrast_data_for_hex(hex_bg: str):
	rep = build_accessibility_report([{"hex": hex_bg, "rgb": (0,0,0), "hsl": (0,0,0)}])
	info = rep["per_color"][0]
	return {"best": info["best"], "white": info["white"], "black": info["black"]}

def render_contrast_previews(flat_colors, section_key: str):
	# Controls
	force = st.selectbox("Text color", ["Auto (best)", "Black", "White"], key=f"force_text_{section_key}")
	cols_per_row = 3
	cols = st.columns(cols_per_row if flat_colors else 1)
	for idx, c in enumerate(flat_colors):
		hex_bg = c["hex"].upper()
		data = _contrast_data_for_hex(hex_bg)
		best = data["best"]["text"]
		text_choice = {"Auto (best)": best, "Black": "black", "White": "white"}[force]
		fg = "#FFFFFF" if text_choice == "white" else "#000000"
		# Ratios for both normal and large
		ratio = data[text_choice]["ratio"]
		aa_ok = ratio >= 4.5
		aa_l_ok = ratio >= 3.0
		aaa_ok = ratio >= 7.0

		with cols[idx % cols_per_row]:
			st.markdown(
				f"<div class='preview-card' style='background:{hex_bg};'>"
				f"<div class='preview-bar' style='background:{fg};opacity:0.25;'></div>"
				f"<div style='color:{fg};font-size:32px;font-weight:800;line-height:1.2;'>H1 32px</div>"
				f"<div style='color:{fg};font-size:20px;opacity:0.95;'>H2 24px equivalent</div>"
				f"<div style='color:{fg};font-size:16px;margin:8px 0;'>Body 16px – readable paragraph</div>"
				f"<button style='padding:8px 12px;border:0;border-radius:8px;background:{fg};color:{hex_bg};'>Button</button>"
				f"<div style='margin-top:10px;'>"
				f"<span class='badge {'badge-pass' if aa_ok else 'badge-fail'}'>AA normal {ratio:.2f}</span>"
				f"<span class='badge {'badge-pass' if aa_l_ok else 'badge-fail'}'>AA large {ratio:.2f}</span>"
				f"<span class='badge {'badge-pass' if aaa_ok else 'badge-fail'}'>AAA {ratio:.2f}</span>"
				f"</div>"
				f"</div>",
				unsafe_allow_html=True,
			)

def render_dyslexia_previews(flat_colors, section_key: str):
	# Controls for better readability testing
	colc = st.columns(4)
	with colc[0]:
		font_family = st.radio("Font", ["OpenDyslexic", "System default"], horizontal=True, key=f"dys_font_{section_key}")
	with colc[1]:
		base_size = st.slider("Font size (px)", 14, 22, 16, 1, key=f"dys_size_{section_key}")
	with colc[2]:
		line_h = st.slider("Line height", 1.2, 2.0, 1.6, 0.1, key=f"dys_lh_{section_key}")
	with colc[3]:
		letter_sp = st.slider("Letter spacing (em)", 0.0, 0.1, 0.03, 0.01, key=f"dys_ls_{section_key}")

	# Load OpenDyslexic font via CSS
	st.markdown("""
	<style>
	@font-face {
		font-family: 'OpenDyslexic';
		src: url('https://cdn.jsdelivr.net/gh/antijingoist/opendyslexic/phase6/otf/OpenDyslexic3-Regular.otf') format('opentype');
		font-display: swap;
	}
	.dys-block {
		padding:16px; border-radius:12px; margin-bottom:10px;
		border:1px solid rgba(0,0,0,0.08);
	}
	.dys-h { font-weight:700; margin:0 0 6px 0; }
	.dys-p { margin:0; opacity:0.95; }
	</style>
	""", unsafe_allow_html=True)

	# Sample content
	heading = "A quick preview heading"
	para = "The quick brown fox jumps over the lazy dog. 1234567890. Sphinx of black quartz, judge my vow."

	cols_per_row = 3
	cols = st.columns(cols_per_row if flat_colors else 1)
	for i, c in enumerate(flat_colors):
		bg = c["hex"].upper()
		# pick readable foreground
		choice = _contrast_data_for_hex(bg)["best"]["text"]
		fg = "#FFFFFF" if choice == "white" else "#000000"

		ff = "'OpenDyslexic', system-ui, -apple-system, Segoe UI, Roboto, sans-serif" if font_family == "OpenDyslexic" else "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
		style = f"background:{bg};color:{fg};font-family:{ff};font-size:{base_size}px;line-height:{line_h};letter-spacing:{letter_sp}em;"
		with cols[i % cols_per_row]:
			st.markdown(
				f"<div class='dys-block' style='{style}'>"
				f"<div class='dys-h'>{heading}</div>"
				f"<div class='dys-p'>{para}</div>"
				f"</div>",
				unsafe_allow_html=True
			)

def render_export_tools(structured, section_key: str):
	flat = flatten_structured(structured)

	# Row 1
	r1c1, r1c2, r1c3 = st.columns(3)
	with r1c1:
		css_txt = export_css_vars(flat, prefix="chroma")
		st.download_button("Download CSS vars", data=css_txt, file_name="palette.css", mime="text/css", key=f"dl_css_{section_key}")
	with r1c2:
		scss_txt = export_scss_vars(flat, prefix="chroma")
		st.download_button("Download SCSS vars", data=scss_txt, file_name="palette.scss", mime="text/x-scss", key=f"dl_scss_{section_key}")
	with r1c3:
		tw = export_tailwind_config(structured)
		st.download_button("Download Tailwind config", data=tw, file_name="tailwind.config.js", mime="application/javascript", key=f"dl_tw_{section_key}")

	# Row 2
	r2c1, r2c2, r2c3 = st.columns(3)
	with r2c1:
		react_json = export_react_theme(structured)
		st.download_button("Download React theme JSON", data=react_json, file_name="theme.json", mime="application/json", key=f"dl_react_{section_key}")
	with r2c2:
		js = export_palette_json(structured)
		st.download_button("Download JSON", data=js, file_name="palette.json", mime="application/json", key=f"dl_json_{section_key}")
	with r2c3:
		# Put ACO/ASE side-by-side but still aligned in the last cell
		cc1, cc2 = st.columns(2)
		with cc1:
			aco_bytes = export_aco(flat, "ChromaGen")
			st.download_button("Adobe ACO", data=aco_bytes, file_name="palette.aco", mime="application/octet-stream", key=f"dl_aco_{section_key}")
		with cc2:
			ase_bytes = export_ase(flat, "ChromaGen")
			st.download_button("Adobe ASE", data=ase_bytes, file_name="palette.ase", mime="application/octet-stream", key=f"dl_ase_{section_key}")

def render_moodboard(flat_colors, section_key: str):
	hexes = _hex_only(flat_colors)
	if not hexes:
		return
	# Choose colors from palette
	bg_grad = f"linear-gradient(135deg, {hexes[0]}, {hexes[min(1,len(hexes)-1)]})"
	card_bg = hexes[min(2, len(hexes)-1)]
	# Compute readable text for the accent card
	card_fg_choice = _contrast_data_for_hex(card_bg)["best"]["text"]
	card_fg = "#FFFFFF" if card_fg_choice == "white" else "#000000"
	# Secondary card background adapted to theme for visibility
	dark_mode = st.session_state.get("dark_mode", False)
	sec_bg = "#111111" if dark_mode else "#ffffff"
	sec_fg = "#ffffff" if dark_mode else "#111111"
	btn_bg = hexes[min(3, len(hexes)-1)]
	btn_fg_choice = _contrast_data_for_hex(btn_bg)["best"]["text"]
	btn_fg = "#FFFFFF" if btn_fg_choice == "white" else "#000000"
	title_fg_choice = _contrast_data_for_hex(hexes[0])["best"]["text"]
	title_fg = "#FFFFFF" if title_fg_choice == "white" else "#000000"
	html = f"""
	<div style="border:1px solid rgba(0,0,0,0.1);border-radius:14px;overflow:hidden;">
	  <div style="padding:30px;color:{title_fg};background:{bg_grad};">
	    <div style="font-size:28px;font-weight:700;">Moodboard Hero</div>
	    <div style="opacity:0.9;margin-top:6px;">Gradient hero composed from your first two swatches.</div>
	    <button style="margin-top:12px;padding:10px 14px;border-radius:8px;background:{btn_bg};color:{btn_fg};border:none;">Primary Action</button>
	  </div>
	  <div style="display:flex;gap:12px;padding:16px;background:rgba(0,0,0,0.02);">
	    <div style="flex:1;padding:14px;border-radius:10px;background:{card_bg};color:{card_fg};">
	      <div style="font-weight:700;">Accent Card</div>
	      <div style="opacity:0.9;">Card content on accent surface</div>
	    </div>
	    <div style="flex:1;padding:14px;border-radius:10px;background:{sec_bg};color:{sec_fg};border:1px solid rgba(0,0,0,0.08);">
	      <div style="font-weight:700;">Secondary Card</div>
	      <div style="opacity:0.9;">Neutral surface with proper contrast</div>
	    </div>
	  </div>
	</div>
	"""
	# st.markdown(html, unsafe_allow_html=True)
	components.html(html, height=340, scrolling=False)  # NEW: render inside iframe

def render_live_preview(flat_colors, section_key: str):
	hexes = _hex_only(flat_colors)
	if not hexes:
		st.info("Generate a palette to preview.")
		return
	# Controls for live mapping
	n = len(hexes)
	colc = st.columns(4)
	with colc[0]:
		bg_idx = st.selectbox("Hero background", [f"{i+1}: {hexes[i]}" for i in range(n)], index=0, key=f"lp_bg_{section_key}")
	with colc[1]:
		accent_idx = st.selectbox("Accent color", [f"{i+1}: {hexes[i]}" for i in range(n)], index=min(1, n-1), key=f"lp_ac_{section_key}")
	with colc[2]:
		btn_idx = st.selectbox("Button color", [f"{i+1}: {hexes[i]}" for i in range(n)], index=min(2, n-1), key=f"lp_btn_{section_key}")
	with colc[3]:
		text_mode = st.selectbox("Text color", ["Auto (best)", "Black", "White"], index=0, key=f"lp_txt_{section_key}")

	def _parse_choice(choice: str) -> int:
		try:
			return max(0, int(choice.split(":")[0]) - 1)
		except Exception:
			return 0

	bg = hexes[_parse_choice(bg_idx)]
	accent = hexes[_parse_choice(accent_idx)]
	btn = hexes[_parse_choice(btn_idx)]

	# Determine hero text color with explicit mapping
	_map = {"Auto (best)": None, "Black": "#000000", "White": "#FFFFFF"}
	chosen = _map.get(text_mode)
	if chosen is None:
		hero_choice = _contrast_data_for_hex(bg)["best"]["text"]
		hero_fg = "#FFFFFF" if hero_choice == "white" else "#000000"
	else:
		hero_fg = chosen

	# Accent foreground for chips/links; ensure readable on accent
	acc_fg_choice = _contrast_data_for_hex(accent)["best"]["text"]
	acc_fg = "#FFFFFF" if acc_fg_choice == "white" else "#000000"

	# Button text color follows selection unless Auto, then best for button bg
	if chosen is None:
		btn_choice = _contrast_data_for_hex(btn)["best"]["text"]
		btn_fg = "#FFFFFF" if btn_choice == "white" else "#000000"
	else:
		btn_fg = chosen

	# Use !important to override dark-mode CSS rules for headings/paragraphs
	html = f"""
	<div class="lp" style="border:1px solid rgba(0,0,0,0.1);border-radius:14px;overflow:hidden;">
		<section style="padding:28px;background:{bg};color:{hero_fg} !important;">
			<!-- Accent bar makes accent choice visible -->
			<div style="height:6px;background:{accent};border-radius:999px;margin-bottom:12px;"></div>

			<h2 style="margin:0 0 8px 0;color:{hero_fg} !important;">Hero Section</h2>
			<p style="margin:0 0 12px 0;opacity:0.92;color:{hero_fg} !important;">Sample landing page preview using your palette mapping.</p>

			<!-- Accent pill + link show accent clearly -->
			<span style="display:inline-block;background:{accent};color:{acc_fg} !important;padding:4px 10px;border-radius:999px;font-size:12px;margin-right:10px;">New</span>
			<a href="#" style="color:{accent} !important;text-decoration:underline;margin-right:14px;">Learn more</a>

			<div style="margin-top:12px;">
				<button style="padding:10px 14px;border-radius:8px;background:{btn};color:{btn_fg} !important;border:none;margin-right:8px;">Primary</button>
				<button style="padding:10px 14px;border-radius:8px;background:transparent;color:{btn} !important;border:1px solid {btn};">Secondary</button>
			</div>
		</section>

		<section style="padding:16px;background:#fff;">
			<div style="display:flex;gap:12px;">
				<div style="flex:1;border:1px solid rgba(0,0,0,0.08);border-left:4px solid {accent};padding:12px;border-radius:8px;">
					<div style="font-weight:700;color:#111111 !important;">Card Title</div>
					<div style="opacity:0.9;color:#111111 !important;">Card copy with neutral contrast</div>
				</div>
				<div style="flex:1;border:1px solid rgba(0,0,0,0.08);border-left:4px solid {accent};padding:12px;border-radius:8px;">
					<div style="font-weight:700;color:#111111 !important;">Card Title</div>
					<div style="opacity:0.9;color:#111111 !important;">Another card using different accent</div>
				</div>
			</div>
		</section>
	</div>
	"""
	# st.markdown(html, unsafe_allow_html=True)
	components.html(html, height=520, scrolling=False)  # NEW: render inside iframe

def render_extras(structured, section_key: str):
	flat = flatten_structured(structured)
	# Tabs (removed Localization & Harmony)
	t1, t2, t3, t4, t5 = st.tabs(["Contrast", "Dyslexia", "Exports", "Storyboard", "Live Preview"])
	with t1:
		st.subheader("Dynamic Contrast Previews")
		render_contrast_previews(flat, section_key)
	with t2:
		st.subheader("Dyslexia-friendly Previews")
		render_dyslexia_previews(flat, section_key)
	with t3:
		st.subheader("Exports")
		render_export_tools(structured, section_key)
	with t4:
		st.subheader("Palette Storyboard")
		render_moodboard(flat, section_key)
	with t5:
		st.subheader("Live Preview")
		render_live_preview(flat, section_key)

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

            # NEW: Accessibility (Image)
            st.subheader("Accessibility")
            flat_img = structured_img["primary"] + structured_img["secondary"] + structured_img["accent"]
            report = build_accessibility_report(flat_img)
            # concise lines per color
            lines = []
            for r in report["per_color"]:
                lines.append(
                    f"{r['hex']} | best text: {r['best']['text']} (ratio {r['best']['ratio']:.2f}) | "
                    f"AA: {'pass' if r['best']['AA'] else 'fail'} | "
                    f"AAA: {'pass' if r['best']['AAA'] else 'fail'}"
                )
            st.code("\n".join(lines), language="text")

            st.markdown("Color‑blindness simulation")
            st.caption(
                "- Protanopia: reduced sensitivity to red (L-cone)\n"
                "- Deuteranopia: reduced sensitivity to green (M-cone)\n"
                "- Tritanopia: reduced sensitivity to blue (S-cone)"
            )
            sim_cols = st.columns(3)
            for col, mode in zip(sim_cols, ["protanopia", "deuteranopia", "tritanopia"]):
                pal_sim = simulate_palette_colors(flat_img, mode)
                with col:
                    st.image(
                        create_palette_image(pal_sim, width=800, height=80),
                        caption=CB_LABELS.get(mode, mode.capitalize()),
                        width="stretch",
                    )

            # NEW: Extras (Image current)
            render_extras(structured_img, "image_current")

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
                    # Use option background card
                    if render_option_card(opt["colors"], f"Option {i+1}", use_key=f"use_img_opt_{i+1}"):
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

                # NEW: Accessibility for selected (Image)
                st.subheader("Accessibility (Selected)")
                flat_sel_img = structured_sel["primary"] + structured_sel["secondary"] + structured_sel["accent"]
                rep_sel = build_accessibility_report(flat_sel_img)
                lines_sel = []
                for r in rep_sel["per_color"]:
                    lines_sel.append(
                        f"{r['hex']} | best text: {r['best']['text']} (ratio {r['best']['ratio']:.2f}) | "
                        f"AA: {'pass' if r['best']['AA'] else 'fail'} | "
                        f"AAA: {'pass' if r['best']['AAA'] else 'fail'}"
                    )
                st.code("\n".join(lines_sel), language="text")

                st.markdown("Color‑blindness simulation (Selected)")
                st.caption(
                    "- Protanopia: reduced sensitivity to red (L-cone)\n"
                    "- Deuteranopia: reduced sensitivity to green (M-cone)\n"
                    "- Tritanopia: reduced sensitivity to blue (S-cone)"
                )
                sim_cols2 = st.columns(3)
                for col, mode in zip(sim_cols2, ["protanopia", "deuteranopia", "tritanopia"]):
                    pal_sim = simulate_palette_colors(flat_sel_img, mode)
                    with col:
                        st.image(
                            create_palette_image(pal_sim, width=800, height=80),
                            caption=CB_LABELS.get(mode, mode.capitalize()),
                            width="stretch",
                        )

                # NEW: Extras (Image)
                render_extras(structured_sel, "image_selected")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ------------------ TEXT TO PALETTE ------------------
elif mode == "Text to Palette":
    st.header("📝 Generate Palette from Text Prompt")

    prompt = st.text_input("Describe your palette ")
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

                # NEW: Accessibility (Text)
                st.subheader("Accessibility")
                flat_txt = structured["primary"] + structured["secondary"] + structured["accent"]
                report_txt = build_accessibility_report(flat_txt)
                lines_txt = []
                for r in report_txt["per_color"]:
                    lines_txt.append(
                        f"{r['hex']} | best text: {r['best']['text']} (ratio {r['best']['ratio']:.2f}) | "
                        f"AA: {'pass' if r['best']['AA'] else 'fail'} | "
                        f"AAA: {'pass' if r['best']['AAA'] else 'fail'}"
                    )
                st.code("\n".join(lines_txt), language="text")

                st.markdown("Color‑blindness simulation")
                st.caption(
                    "- Protanopia: reduced sensitivity to red (L-cone)\n"
                    "- Deuteranopia: reduced sensitivity to green (M-cone)\n"
                    "- Tritanopia: reduced sensitivity to blue (S-cone)"
                )
                sim_cols_t = st.columns(3)
                for col, mode in zip(sim_cols_t, ["protanopia", "deuteranopia", "tritanopia"]):
                    pal_sim = simulate_palette_colors(flat_txt, mode)
                    with col:
                        st.image(
                            create_palette_image(pal_sim, width=800, height=80),
                            caption=CB_LABELS.get(mode, mode.capitalize()),
                            width="stretch",
                        )

                # NEW: Extras (Text current)
                render_extras(structured, "text_current")

                # More options (same pattern as image mode)
                opts = text_to_palette_options_structured(prompt, n_colors=6, n_options=4)
                cols = st.columns(2)
                for i, sp in enumerate(opts):
                    flat_opt = sp["primary"] + sp["secondary"] + sp["accent"]
                    with cols[i % 2]:
                        if render_option_card(flat_opt, f"Option {i+1}", use_key=f"use_txt_opt_{i+1}"):
                            st.session_state["selected_palette_text"] = sp

                if "selected_palette_text" in st.session_state:
                    sel = st.session_state["selected_palette_text"]
                    st.subheader("✅ Selected Palette (Text)")
                    flat_sel = sel["primary"] + sel["secondary"] + sel["accent"]
                    st.image(create_palette_image(flat_sel, width=800, height=120), caption="Your selection", width="stretch")

                    # NEW: Accessibility for selected (Text)
                    st.subheader("Accessibility (Selected)")
                    flat_sel_txt = sel["primary"] + sel["secondary"] + sel["accent"]
                    rep_sel_t = build_accessibility_report(flat_sel_txt)
                    lines_sel_t = []
                    for r in rep_sel_t["per_color"]:
                        lines_sel_t.append(
                            f"{r['hex']} | best text: {r['best']['text']} (ratio {r['best']['ratio']:.2f}) | "
                            f"AA: {'pass' if r['best']['AA'] else 'fail'} | "
                            f"AAA: {'pass' if r['best']['AAA'] else 'fail'}"
                        )
                    st.code("\n".join(lines_sel_t), language="text")

                    st.markdown("Color‑blindness simulation (Selected)")
                    st.caption(
                        "- Protanopia: reduced sensitivity to red (L-cone)\n"
                        "- Deuteranopia: reduced sensitivity to green (M-cone)\n"
                        "- Tritanopia: reduced sensitivity to blue (S-cone)"
                    )
                    sim_cols_ts = st.columns(3)
                    for col, mode in zip(sim_cols_ts, ["protanopia", "deuteranopia", "tritanopia"]):
                        pal_sim = simulate_palette_colors(flat_sel_txt, mode)
                        with col:
                            st.image(
                                create_palette_image(pal_sim, width=800, height=80),
                                caption=CB_LABELS.get(mode, mode.capitalize()),
                                width="stretch",
                            )
            except Exception as e:
                st.error(f"Error generating palette: {str(e)}")
        else:
            st.warning("Please enter a text description.")