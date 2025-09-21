import streamlit as st
from dotenv import load_dotenv
import os
from io import BytesIO
from PIL import Image, ImageEnhance
import base64
import streamlit.components.v1 as components
from datetime import datetime
import requests  # keep for optional Google Fonts API
import re  # NEW: regex for CSS parsing
from urllib.parse import urljoin  # NEW: resolve linked CSS URLs

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
    # NEW imports for adaptive theme
    rgb_to_hex,
    hsl_to_rgb,
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

# ---------------------- NEW: History helpers ----------------------
def _safe_rerun():
	"""Compatibility rerun: prefer st.rerun(), fallback to st.experimental_rerun()."""
	try:
		st.rerun()
	except Exception:
		try:
			st.experimental_rerun()
		except Exception:
			pass

def _init_history():
	if "history" not in st.session_state:
		st.session_state["history"] = []  # list of entries
	if "history_loaded_palette" not in st.session_state:
		st.session_state["history_loaded_palette"] = None
	if "history_sigs" not in st.session_state:
		st.session_state["history_sigs"] = set()
	if "history_preview_sigs" not in st.session_state:
		st.session_state["history_preview_sigs"] = set()

def _palette_signature(structured_or_flat) -> str:
	# Accept structured {primary,secondary,accent} or flat list
	try:
		def _hexes(arr): return [c["hex"].upper() for c in arr]
		if isinstance(structured_or_flat, dict):
			flat = structured_or_flat.get("primary", []) + structured_or_flat.get("secondary", []) + structured_or_flat.get("accent", [])
		else:
			flat = structured_or_flat
		return "|".join(_hexes(flat))
	except Exception:
		return ""

def _push_history(entry: dict):
	_init_history()
	entry["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	st.session_state["history"].insert(0, entry)
	# keep last 25
	if len(st.session_state["history"]) > 25:
		del st.session_state["history"][25:]

def _push_history_palette(label: str, structured: dict):
	_init_history()
	sig = _palette_signature(structured)
	if not sig or sig in st.session_state["history_sigs"]:
		return
	flat = structured.get("primary", []) + structured.get("secondary", []) + structured.get("accent", [])
	img_b64 = _palette_to_base64_image(flat, width=600, height=80)
	st.session_state["history_sigs"].add(sig)
	_push_history({"type": "palette", "label": label, "structured": structured, "thumb_b64": img_b64, "sig": sig})

def _push_history_live_preview(label: str, mapping: dict):
	_init_history()
	# Build a unique signature from mapping values (bg|accent|btn|text)
	sig = f"{mapping.get('bg','')}|{mapping.get('accent','')}|{mapping.get('btn','')}|{mapping.get('text','')}"
	if sig in st.session_state["history_preview_sigs"]:
		return
	st.session_state["history_preview_sigs"].add(sig)
	_push_history({"type": "preview", "label": label, "mapping": mapping, "sig": sig})

def _render_history_sidebar():
	_init_history()
	with st.sidebar.expander(f"History ({len(st.session_state['history'])})", expanded=False):
		if not st.session_state["history"]:
			st.caption("No history yet. Generate a palette or save a live preview.")
		for i, e in enumerate(st.session_state["history"][:12]):
			if e.get("type") == "palette":
				c1, c2 = st.columns([3, 1])
				with c1:
					st.markdown(f"**{e.get('label','Palette')}** · {e.get('ts','')}")
					st.markdown(
						f"<img src='data:image/png;base64,{e['thumb_b64']}' style='width:100%;border-radius:8px;border:1px solid rgba(0,0,0,0.08);' />",
						unsafe_allow_html=True,
					)
				with c2:
					if st.button("Load", key=f"hist_load_{i}"):
						st.session_state["history_loaded_palette"] = e["structured"]
						st.toast("Loaded palette from history", icon="✅")
						_safe_rerun()
					if st.button("Delete", type="secondary", key=f"hist_del_{i}"):
						# remove and cleanup sig set
						if e.get("sig"):
							st.session_state["history_sigs"].discard(e["sig"])
						del st.session_state["history"][i]
						_safe_rerun()
				st.divider()
			elif e.get("type") == "preview":
				st.markdown(f"**{e.get('label','Preview')}** · {e.get('ts','')}")
				m = e.get("mapping", {})
				def chip(h): 
					return f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;border:1px solid rgba(0,0,0,0.15);background:{h};margin-right:6px;vertical-align:middle;'></span><span style='font-family:monospace;margin-right:12px'>{h}</span>"
				row = (chip(m.get('bg','')) + chip(m.get('accent','')) + chip(m.get('btn','')) + (chip(m.get('text')) if m.get('text') and m.get('text') != 'Auto' else "<span style='opacity:0.7;'>Text: Auto</span>"))
				st.markdown(row, unsafe_allow_html=True)
				if st.button("Delete", type="secondary", key=f"hist_del_prev_{i}"):
					st.session_state["history_preview_sigs"].discard(e.get("sig",""))
					del st.session_state["history"][i]
					_safe_rerun()
				st.divider()
		# Footer actions
		ca, cb = st.columns(2)
		with ca:
			if st.button("Clear history", key="hist_clear_all"):
				st.session_state["history"] = []
				st.session_state["history_sigs"] = set()
				st.session_state["history_preview_sigs"] = set()
				st.session_state["history_loaded_palette"] = None
				_safe_rerun()
		with cb:
			if st.session_state.get("history_loaded_palette"):
				if st.button("Unload palette", key="hist_unload"):
					st.session_state["history_loaded_palette"] = None
					_safe_rerun()

# ---------------------- NEW: Collaboration prototype — votes ----------------------
def _render_vote_widget(structured: dict, context_key: str):
	"""
	Simple local-only voting widget (prototype).
	Stores counts in st.session_state['votes'] and the user's vote in st.session_state['my_vote'].
	"""
	sig = _palette_signature(structured)
	if not sig:
		return
	votes = st.session_state.setdefault("votes", {})
	my = st.session_state.setdefault("my_vote", {})
	counts = votes.get(sig, {"up": 0, "down": 0})
	curr = my.get(sig, 0)  # 1=up, -1=down, 0=none

	c1, c2, c3 = st.columns([1, 1, 3])
	with c1:
		if st.button(f"👍 Upvote ({counts['up']})", key=f"up_{context_key}_{abs(hash(sig))}"):
			if curr == 1:
				counts["up"] = max(0, counts["up"] - 1)
				curr = 0
			else:
				if curr == -1:
					counts["down"] = max(0, counts["down"] - 1)
				counts["up"] += 1
				curr = 1
	with c2:
		if st.button(f"👎 Downvote ({counts['down']})", key=f"down_{context_key}_{abs(hash(sig))}"):
			if curr == -1:
				counts["down"] = max(0, counts["down"] - 1)
				curr = 0
			else:
				if curr == 1:
					counts["up"] = max(0, counts["up"] - 1)
				counts["down"] += 1
				curr = -1
	with c3:
		score = counts["up"] - counts["down"]
		status = "None" if curr == 0 else ("Up" if curr == 1 else "Down")
		st.markdown(
			f"<div style='margin-top:6px;'>Score: <b>{'+' if score>0 else ''}{score}</b> · Your vote: <i>{status}</i></div>",
			unsafe_allow_html=True
		)
	# persist
	votes[sig] = counts
	my[sig] = curr

# Render History panel now (before mode switch)
_render_history_sidebar()

mode = st.sidebar.radio("Mode", ["Image to Palette", "Text to Palette", "Website to Palette"])

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
	# Local contrast calculator (AA/AAA against any chosen text color hex)
	rep = build_accessibility_report([{"hex": hex_bg, "rgb": (0,0,0), "hsl": (0,0,0)}])
	info = rep["per_color"][0]
	return {"best": info["best"], "white": info["white"], "black": info["black"]}

# NEW: small color swatch for inline use in dropdowns/tables
def _color_chip(hexv: str) -> str:
	return (
		f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;"
		f"border:1px solid rgba(0,0,0,0.15);background:{hexv};vertical-align:middle;margin-right:6px;'></span>"
	)

# NEW: local contrast helpers (for arbitrary text hex)
def _srgb_to_linear_val(c: float) -> float:
	return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _rel_luminance_hex(hexv: str) -> float:
	h = hexv.strip().lstrip("#")
	r = int(h[0:2], 16) / 255.0
	g = int(h[2:4], 16) / 255.0
	b = int(h[4:6], 16) / 255.0
	R = _srgb_to_linear_val(r)
	G = _srgb_to_linear_val(g)
	B = _srgb_to_linear_val(b)
	return 0.2126 * R + 0.7152 * G + 0.0722 * B

def _contrast_ratio_hex(fg_hex: str, bg_hex: str) -> float:
	L1 = _rel_luminance_hex(fg_hex)
	L2 = _rel_luminance_hex(bg_hex)
	L_light, L_dark = (L1, L2) if L1 >= L2 else (L2, L1)
	return (L_light + 0.05) / (L_dark + 0.05)

# NEW: color conversion helpers for Website to Palette
def _hex_to_rgb(hexv: str) -> tuple[int, int, int]:
	h = hexv.strip().lstrip("#")
	if len(h) == 3:
		h = "".join([c * 2 for c in h])
	return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def _rgb_to_hsl(r: int, g: int, b: int) -> tuple[int, int, int]:
	rp, gp, bp = r / 255.0, g / 255.0, b / 255.0
	cmax, cmin = max(rp, gp, bp), min(rp, gp, bp)
	delta = cmax - cmin
	if delta == 0:
		h = 0
	elif cmax == rp:
		h = (60 * ((gp - bp) / delta) + 360) % 360
	elif cmax == gp:
		h = (60 * ((bp - rp) / delta) + 120) % 360
	else:
		h = (60 * ((rp - gp) / delta) + 240) % 360
	l = (cmax + cmin) / 2
	s = 0 if delta == 0 else delta / (1 - abs(2 * l - 1))
	return (int(round(h)), int(round(s * 100)), int(round(l * 100)))

def render_contrast_previews(flat_colors, section_key: str):
	# Controls: include all palette colors
	hexes = _hex_only(flat_colors)
	text_options = ["Auto (best)", "Black (#000000)", "White (#FFFFFF)"] + [f"{i+1}: {hexes[i]}" for i in range(len(hexes))]
	force = st.selectbox("Text color", text_options, key=f"force_text_{section_key}")

	# Resolve chosen text hex (None for Auto)
	if force.startswith("Auto"):
		_sel_hex = None
	elif force.startswith("Black"):
		_sel_hex = "#000000"
	elif force.startswith("White"):
		_sel_hex = "#FFFFFF"
	elif ":" in force:
		try:
			_sel_hex = hexes[int(force.split(":")[0]) - 1]
		except Exception:
			_sel_hex = None
	else:
		_sel_hex = None

	cols_per_row = 3
	cols = st.columns(cols_per_row if flat_colors else 1)
	for idx, c in enumerate(flat_colors):
		hex_bg = c["hex"].upper()
		data = _contrast_data_for_hex(hex_bg)
		# Determine actual text color to use on this swatch
		if _sel_hex is None:
			# Auto: use best (white/black)
			text_choice = data["best"]["text"]  # "white" | "black"
			text_hex = "#FFFFFF" if text_choice == "white" else "#000000"
			ratio = data[text_choice]["ratio"]
		else:
			text_hex = _sel_hex
			# If chosen is pure white/black, reuse computed ratios; else compute
			if text_hex == "#FFFFFF":
				ratio = data["white"]["ratio"]
			elif text_hex == "#000000":
				ratio = data["black"]["ratio"]
			else:
				ratio = round(_contrast_ratio_hex(text_hex, hex_bg), 2)

		aa_ok = ratio >= 4.5
		aa_l_ok = ratio >= 3.0
		aaa_ok = ratio >= 7.0

		with cols[idx % cols_per_row]:
			st.markdown(
				f"<div class='preview-card' style='background:{hex_bg};'>"
				f"<div class='preview-bar' style='background:{text_hex};opacity:0.25;'></div>"
				f"<div style='color:{text_hex};font-size:32px;font-weight:800;line-height:1.2;'>H1 32px</div>"
				f"<div style='color:{text_hex};font-size:20px;opacity:0.95;'>H2 24px equivalent</div>"
				f"<div style='color:{text_hex};font-size:16px;margin:8px 0;'>Body 16px – readable paragraph</div>"
				f"<button style='padding:8px 12px;border:0;border-radius:8px;background:{text_hex};color:{hex_bg};'>Button</button>"
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
		# Stack ACO/ASE vertically for better alignment
		aco_bytes = export_aco(flat, "ChromaGen")
		st.download_button("Adobe ACO", data=aco_bytes, file_name="palette.aco", mime="application/octet-stream", key=f"dl_aco_{section_key}")
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

def render_accessibility_table(colors: list[dict], key_prefix: str):
	# Build from report, include palette as text candidates
	rep = build_accessibility_report(colors, text_candidates=colors)
	rows_html = []
	# Header (added palette columns)
	header = """
	<table style="width:100%;border-collapse:collapse;">
	  <thead>
	    <tr>
	      <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(0,0,0,0.1);">Color</th>
	      <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(0,0,0,0.1);">HEX</th>
	      <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(0,0,0,0.1);">Best Text (B/W)</th>
	      <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(0,0,0,0.1);">Ratio</th>
	      <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(0,0,0,0.1);">AA</th>
	      <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(0,0,0,0.1);">AAA</th>
	      <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(0,0,0,0.1);">Palette Text</th>
	      <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(0,0,0,0.1);">P Ratio</th>
	      <th style="text-align:left;padding:8px;border-bottom:1px solid rgba(0,0,0,0.1);">Remark</th>
	    </tr>
	  </thead>
	  <tbody>
	"""
	for r in rep["per_color"]:
		hexv = r["hex"]
		best = r["best"]
		ratio = f"{best['ratio']:.2f}"
		aa = "Pass" if best["AA"] else "Fail"
		aaa = "Pass" if best["AAA"] else "Fail"
		msg = r.get("message", "")
		pb = r.get("palette_best")
		# small swatches
		def sw(h): return f"<span style='display:inline-block;width:16px;height:16px;border-radius:3px;border:1px solid rgba(0,0,0,0.12);background:{h};margin-right:8px;vertical-align:middle;'></span>"
		palette_cell = "-"
		palette_ratio = "-"
		if pb:
			palette_cell = sw(pb["hex"]) + f"<span style='font-family:monospace'>{pb['hex']}</span>"
			palette_ratio = f"{pb['ratio']:.2f}"

		rows_html.append(
			f"<tr>"
			f"<td style='padding:8px;border-bottom:1px solid rgba(0,0,0,0.06);'>{sw(hexv)}</td>"
			f"<td style='padding:8px;border-bottom:1px solid rgba(0,0,0,0.06);font-family:monospace;'>{hexv}</td>"
			f"<td style='padding:8px;border-bottom:1px solid rgba(0,0,0,0.06);'>{best['text'].capitalize()}</td>"
			f"<td style='padding:8px;border-bottom:1px solid rgba(0,0,0,0.06);'>{ratio}</td>"
			f"<td style='padding:8px;border-bottom:1px solid rgba(0,0,0,0.06);'>{aa}</td>"
			f"<td style='padding:8px;border-bottom:1px solid rgba(0,0,0,0.06);'>{aaa}</td>"
			f"<td style='padding:8px;border-bottom:1px solid rgba(0,0,0,0.06);'>{palette_cell}</td>"
			f"<td style='padding:8px;border-bottom:1px solid rgba(0,0,0,0.06);'>{palette_ratio}</td>"
			f"<td style='padding:8px;border-bottom:1px solid rgba(0,0,0,0.06);'>{msg}</td>"
			f"</tr>"
		)
	html = header + "\n".join(rows_html) + "</tbody></table>"
	st.markdown(html, unsafe_allow_html=True)
	st.caption(f"Summary: AA pass {rep['summary']['AA_pass']} · AAA pass {rep['summary']['AAA_pass']}")

def render_live_preview(flat_colors, section_key: str):
	hexes = _hex_only(flat_colors)
	if not hexes:
		st.info("Generate a palette to preview.")
		return
	# Controls for live mapping
	n = len(hexes)
	colc = st.columns(4)

	# Hero background selector + chip
	with colc[0]:
		bg_row = st.columns([4, 2])
		with bg_row[0]:
			bg_idx = st.selectbox("Hero background", [f"{i+1}: {hexes[i]}" for i in range(n)], index=0, key=f"lp_bg_{section_key}")
		# parse and show chip next to dropdown
		_bg = hexes[int(bg_idx.split(':')[0]) - 1] if ':' in bg_idx else hexes[0]
		with bg_row[1]:
			st.markdown(_color_chip(_bg) + f"<span style='font-family:monospace'>{_bg}</span>", unsafe_allow_html=True)

	# Accent selector + chip
	with colc[1]:
		ac_row = st.columns([4, 2])
		with ac_row[0]:
			accent_idx = st.selectbox("Accent color", [f"{i+1}: {hexes[i]}" for i in range(n)], index=min(1, n-1), key=f"lp_ac_{section_key}")
		_ac = hexes[int(accent_idx.split(':')[0]) - 1] if ':' in accent_idx else hexes[min(1, n-1)]
		with ac_row[1]:
			st.markdown(_color_chip(_ac) + f"<span style='font-family:monospace'>{_ac}</span>", unsafe_allow_html=True)

	# Button selector + chip
	with colc[2]:
		btn_row = st.columns([4, 2])
		with btn_row[0]:
			btn_idx = st.selectbox("Button color", [f"{i+1}: {hexes[i]}" for i in range(n)], index=min(2, n-1), key=f"lp_btn_{section_key}")
		_bt = hexes[int(btn_idx.split(':')[0]) - 1] if ':' in btn_idx else hexes[min(2, n-1)]
		with btn_row[1]:
			st.markdown(_color_chip(_bt) + f"<span style='font-family:monospace'>{_bt}</span>", unsafe_allow_html=True)

	# Text color selector: add palette colors + chip
	with colc[3]:
		txt_row = st.columns([4, 2])
		with txt_row[0]:
			text_options = ["Auto (best)", "Black (#000000)", "White (#FFFFFF)"] + [f"{i+1}: {hexes[i]}" for i in range(n)]
			text_mode = st.selectbox("Text color", text_options, index=0, key=f"lp_txt_{section_key}")
		# Resolve selected text color for chip display (None for Auto)
		if text_mode.startswith("Auto"):
			_txt_hex = None
		elif text_mode.startswith("Black"):
			_txt_hex = "#000000"
		elif text_mode.startswith("White"):
			_txt_hex = "#FFFFFF"
		elif ":" in text_mode:
			# "k: #HEX"
			try:
				_txt_hex = hexes[int(text_mode.split(":")[0]) - 1]
			except Exception:
				_txt_hex = None
		else:
			_txt_hex = None
		with txt_row[1]:
			if _txt_hex:
				st.markdown(_color_chip(_txt_hex) + f"<span style='font-family:monospace'>{_txt_hex}</span>", unsafe_allow_html=True)
			else:
				st.markdown("<span style='opacity:0.8;'>Auto</span>", unsafe_allow_html=True)

	def _parse_choice(choice: str) -> int:
		try:
			return max(0, int(choice.split(":")[0]) - 1)
		except Exception:
			return 0

	bg = hexes[_parse_choice(bg_idx)]
	accent = hexes[_parse_choice(accent_idx)]
	btn = hexes[_parse_choice(btn_idx)]

	# Determine hero text color with explicit mapping (now supports palette colors)
	if _txt_hex is None:
		hero_choice = _contrast_data_for_hex(bg)["best"]["text"]
		hero_fg = "#FFFFFF" if hero_choice == "white" else "#000000"
	else:
		hero_fg = _txt_hex

	# Accent foreground for chips/links; ensure readable on accent
	acc_fg_choice = _contrast_data_for_hex(accent)["best"]["text"]
	acc_fg = "#FFFFFF" if acc_fg_choice == "white" else "#000000"

	# Button text color follows text selection unless Auto, then best for button bg
	if _txt_hex is None:
		btn_choice = _contrast_data_for_hex(btn)["best"]["text"]
		btn_fg = "#FFFFFF" if btn_choice == "white" else "#000000"
	else:
		btn_fg = _txt_hex

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

# ---------------------- NEW: Adaptive Theme (Light + Dark) ----------------------
def _clamp(v, lo, hi):
	return lo if v < lo else hi if v > hi else v

def _adapt_color(c: dict, mode: str) -> dict:
	h, s, l = c["hsl"]
	if mode == "dark":
		if l >= 50:
			l2 = _clamp(100 - l + 10, 15, 65)
		else:
			l2 = _clamp(l - 10, 15, 65)
		s2 = _clamp(int(s * 1.08), 15, 90)
	else:
		l2 = _clamp(l, 30, 90)
		s2 = _clamp(s, 15, 90)
	r2 = hsl_to_rgb((h, s2, l2))
	return {"hex": rgb_to_hex(r2).upper(), "rgb": r2, "hsl": (h, s2, l2)}

def generate_adaptive_theme(structured: dict) -> dict:
	def _apply(mode: str):
		return {
			"primary": [_adapt_color(c, mode) for c in structured.get("primary", [])],
			"secondary": [_adapt_color(c, mode) for c in structured.get("secondary", [])],
			"accent": [_adapt_color(c, mode) for c in structured.get("accent", [])],
		}
	return {"light": _apply("light"), "dark": _apply("dark")}

def render_adaptive_theme(structured: dict, section_key: str):
	flat_light = flatten_structured(generate_adaptive_theme(structured)["light"])
	flat_dark = flatten_structured(generate_adaptive_theme(structured)["dark"])
	c1, c2 = st.columns(2)
	with c1:
		st.markdown("#### Light theme")
		st.image(create_palette_image(flat_light, width=800, height=80), caption=None, width="stretch")
		rep_l = build_accessibility_report(flat_light, text_candidates=flat_light)
		st.caption(f"AA pass {rep_l['summary']['AA_pass']} • AAA pass {rep_l['summary']['AAA_pass']}")
	with c2:
		st.markdown("#### Dark theme")
		st.image(create_palette_image(flat_dark, width=800, height=80), caption=None, width="stretch")
		rep_d = build_accessibility_report(flat_dark, text_candidates=flat_dark)
		st.caption(f"AA pass {rep_d['summary']['AA_pass']} • AAA pass {rep_d['summary']['AAA_pass']}")
	with st.expander("Show accessibility details"):
		st.markdown("Light")
		render_accessibility_table(flat_light, f"theme_light_{section_key}")
		st.markdown("Dark")
		render_accessibility_table(flat_dark, f"theme_dark_{section_key}")

# ---------------------- NEW: Typography helpers ----------------------
def _palette_stats(flat_colors):
	if not flat_colors:
		return {"avg_s": 0, "avg_l": 50, "hue_range": 0}
	S = [c["hsl"][1] for c in flat_colors if "hsl" in c]
	L = [c["hsl"][2] for c in flat_colors if "hsl" in c]
	H = [c["hsl"][0] for c in flat_colors if "hsl" in c]
	return {
		"avg_s": sum(S) / max(1, len(S)),
		"avg_l": sum(L) / max(1, len(L)),
		"hue_range": (max(H) - min(H)) if H else 0
	}

def _infer_vibe(flat_colors):
	stats = _palette_stats(flat_colors)
	avg_s, avg_l, hue_range = stats["avg_s"], stats["avg_l"], stats["hue_range"]
	if avg_s >= 55 and hue_range >= 80:
		return "playful"
	if avg_l <= 40:
		return "bold"
	if avg_l >= 70 and avg_s <= 35:
		return "calm"
	return "modern"

_CURATED_FONTS = {
	"modern": [
		{"heading": "Poppins", "body": "Inter", "mono": "Space Grotesk"},
		{"heading": "Outfit", "body": "Source Sans 3", "mono": "IBM Plex Mono"},
	],
	"calm": [
		{"heading": "Newsreader", "body": "Source Sans 3", "mono": "IBM Plex Mono"},
		{"heading": "Merriweather", "body": "Nunito Sans", "mono": "Fira Code"},
	],
	"bold": [
		{"heading": "Oswald", "body": "DM Sans", "mono": "JetBrains Mono"},
		{"heading": "Work Sans", "body": "Manrope", "mono": "Roboto Mono"},
	],
	"playful": [
		{"heading": "Fredoka", "body": "Nunito", "mono": "Fira Code"},
		{"heading": "Baloo 2", "body": "Quicksand", "mono": "DM Mono"},
	],
}

def _gf_css_link(families):
	q = "&".join([f"family={f.replace(' ', '+')}" for f in families])
	return f"https://fonts.googleapis.com/css2?{q}&display=swap"

def _families_for_pack(pack):
	h = f"{pack['heading']}:wght@600;800"
	b = f"{pack['body']}:wght@400;500;600"
	m = f"{pack['mono']}:wght@400;600"
	return [h, b, m]

def _maybe_swap_with_api(pack):
	api_key = os.getenv("GOOGLE_FONTS_API_KEY", "").strip()
	if not api_key:
		return pack
	try:
		resp = requests.get("https://www.googleapis.com/webfonts/v1/webfonts", params={"key": api_key, "sort": "popularity"}, timeout=5)
		if resp.ok:
			for fam in resp.json().get("items", [])[:40]:
				name = fam.get("family", "")
				if name and name not in (pack["heading"], pack["body"], pack["mono"]):
					return {**pack, "body": name}
	except Exception:
		pass
	return pack

def render_typography_suggestions(structured, section_key: str):
	flat = flatten_structured(structured)
	vibe = _infer_vibe(flat)
	idx_key = f"typo_idx_{section_key}"
	if idx_key not in st.session_state:
		st.session_state[idx_key] = 0
	alternates = _CURATED_FONTS.get(vibe, _CURATED_FONTS["modern"])
	pack_base = alternates[st.session_state[idx_key] % len(alternates)]
	pack = _maybe_swap_with_api(dict(pack_base))
	families = _families_for_pack(pack)
	css_href = _gf_css_link(families)
	html = f"""
	<!doctype html><html><head>
	  <link rel="preconnect" href="https://fonts.googleapis.com">
	  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
	  <link href="{css_href}" rel="stylesheet">
	  <style>
	    body {{ margin:0; padding:22px; background:#fff; color:#111; }}
	    .h1 {{ font-family:'{pack['heading']}', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; font-size:32px; font-weight:800; margin:0 0 8px 0; }}
	    .h2 {{ font-family:'{pack['heading']}', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; font-size:20px; font-weight:700; margin:0 0 10px 0; opacity:.95; }}
	    .body {{ font-family:'{pack['body']}', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; font-size:16px; line-height:1.65; }}
	    .mono {{ font-family:'{pack['mono']}', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:13px; background:#f7f7f7; padding:8px 10px; border-radius:8px; border:1px solid rgba(0,0,0,0.08); margin-top:12px; display:inline-block; }}
	  </style>
	</head><body>
	  <div class="h1">Heading in {pack['heading']}</div>
	  <div class="h2">Subheading in {pack['heading']} (600/700)</div>
	  <div class="body">Body in {pack['body']} · The quick brown fox jumps over the lazy dog — 12345.</div>
	  <div class="mono">code.sample = true // Monospace: {pack['mono']}</div>
	</body></html>
	"""
	st.markdown(f"Detected vibe: {vibe.capitalize()} · Heading: {pack['heading']} · Body: {pack['body']} · Mono: {pack['mono']}")
	components.html(html, height=230, scrolling=False)
	st.code(f'<link href="{css_href}" rel="stylesheet" />', language="html")
	st.code(
		f""":root {{
  --font-heading: '{pack['heading']}', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  --font-body: '{pack['body']}', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  --font-mono: '{pack['mono']}', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
.h1 {{ font-family: var(--font-heading); }}
.body {{ font-family: var(--font-body); }}
code, pre {{ font-family: var(--font-mono); }}""",
		language="css",
	)
	c1, _ = st.columns(2)
	with c1:
		if st.button("Next suggestion", key=f"typo_next_{section_key}"):
			st.session_state[idx_key] += 1
			st.rerun()

# ---------------------- NEW: Compose Extras tabs ----------------------
def render_extras(structured, section_key: str):
	flat = flatten_structured(structured)
	t1, t2, t3, t4, t5, t6, t7 = st.tabs(["Contrast", "Dyslexia", "Exports", "Storyboard", "Live Preview", "Typography", "Adaptive Theme"])
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
	with t6:
		st.subheader("Typography")
		render_typography_suggestions(structured, section_key)
	with t7:
		st.subheader("Adaptive Theme (Light + Dark)")
		render_adaptive_theme(structured, section_key)

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
            # Moved voting below after structured palette is ready

            # NEW: structured view (primary, secondary, accent) with copy blocks
            payload_img = palette_payload_from_colors(colors)
            structured_img = payload_img["palette"]
            copy_blocks_img = payload_img["copy"]

            # NEW: voting for current image result (now that structured_img exists)
            _render_vote_widget(structured_img, "img_current")
            # NEW: save to history
            _push_history_palette("Image", structured_img)

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

            # NEW: Accessibility (Image) table
            st.subheader("Accessibility")
            flat_img = structured_img["primary"] + structured_img["secondary"] + structured_img["accent"]
            render_accessibility_table(flat_img, "img_current")

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
                # Moved voting below after structured_sel is ready

                # NEW: structured view for selected option
                payload_sel = palette_payload_from_colors(sel_img)
                structured_sel = payload_sel["palette"]
                copy_sel = payload_sel["copy"]

                # NEW: voting for selected image palette (now that structured_sel exists)
                _render_vote_widget(structured_sel, "img_selected")
                # NEW: save selected to history
                _push_history_palette("Image – Selected", structured_sel)

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

                # NEW: Accessibility (Selected) table
                st.subheader("Accessibility (Selected)")
                flat_sel_img = structured_sel["primary"] + structured_sel["secondary"] + structured_sel["accent"]
                render_accessibility_table(flat_sel_img, "img_selected")

                st.markdown("Color‑blindness simulation (Selected)")
                st.caption(
                    "- Protanopia: reduced sensitivity to red (L-cone)\n"
                    "- Deuteranopia: reduced sensitivity to green (M-cone)\n"
                    "- Tritanopia: reduced sensitivity to blue (S-cone)"
                )
                sim_cols2 = st.columns(3)
                for col, mode in zip(sim_cols2, ["protanopia", "deuteranopia", "tritanopia"]):
                    pal_sim = simulate_palette_colors(flat_sel_img, mode)  # FIX: variable name
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
                # NEW: voting for current text result
                _render_vote_widget(structured, "txt_current")
                # NEW: save to history (short prompt in label)
                def _short(s, n=32): 
                    s = (s or "").strip()
                    return s if len(s) <= n else s[:n-1] + "…"
                _push_history_palette(f"Text: {_short(prompt, 32)}", structured)

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

                # NEW: Accessibility (Text) table
                st.subheader("Accessibility")
                flat_txt = structured["primary"] + structured["secondary"] + structured["accent"]
                render_accessibility_table(flat_txt, "txt_current")

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
                    # NEW: voting for selected text palette
                    _render_vote_widget(sel, "txt_selected")
                    # NEW: save selected to history
                    _push_history_palette("Text – Selected", sel)

                    st.markdown("Primary")
                    st.image(create_palette_image(sel["primary"], width=800, height=80), caption=None, width="stretch")
                    st.code(copy_blocks["primary"]["hex"], language="text")
                    st.code(copy_blocks["primary"]["rgb"], language="text")
                    st.code(copy_blocks["primary"]["hsl"], language="text")

                    st.markdown("Secondary")
                    st.image(create_palette_image(sel["secondary"], width=800, height=80), caption=None, width="stretch")
                    st.code(copy_blocks["secondary"]["hex"], language="text")
                    st.code(copy_blocks["secondary"]["rgb"], language="text")
                    st.code(copy_blocks["secondary"]["hsl"], language="text")

                    st.markdown("Accent")
                    st.image(create_palette_image(sel["accent"], width=800, height=80), caption=None, width="stretch")
                    st.code(copy_blocks["accent"]["hex"], language="text")
                    st.code(copy_blocks["accent"]["rgb"], language="text")
                    st.code(copy_blocks["accent"]["hsl"], language="text")

                    with st.expander("Structured JSON (Selected)"):
                        st.code(payload["json"], language="json")

                    # NEW: Accessibility (Selected) table
                    st.subheader("Accessibility (Selected)")
                    flat_sel_img = sel["primary"] + sel["secondary"] + sel["accent"]
                    render_accessibility_table(flat_sel_img, "txt_selected")

                    st.markdown("Color‑blindness simulation (Selected)")
                    st.caption(
                        "- Protanopia: reduced sensitivity to red (L-cone)\n"
                        "- Deuteranopia: reduced sensitivity to green (M-cone)\n"
                        "- Tritanopia: reduced sensitivity to blue (S-cone)"
                    )
                    sim_cols2 = st.columns(3)
                    for col, mode in zip(sim_cols2, ["protanopia", "deuteranopia", "tritanopia"]):
                        pal_sim = simulate_palette_colors(flat_sel_img, mode)  # FIX: variable name
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

# ------------------ WEBSITE TO PALETTE ------------------
if mode == "Website to Palette":
	st.header("🌐 Extract Palette from Website URL")
	url = st.text_input("Enter website URL", placeholder="https://example.com")
	colw1, colw2 = st.columns([2,1])

	with colw1:
		include_css = st.checkbox("Include linked stylesheets", value=True)
	with colw2:
		max_colors = st.slider("Max colors", 3, 12, 8, 1)

	# SAFETY: local minimal fallback if global extractor isn't defined
	def _extract_from_url_fallback(u: str, include_linked_css=True, css_limit=3) -> list[str]:
		try:
			resp = requests.get(u, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
			resp.raise_for_status()
			html = resp.text or ""
		except Exception:
			return []
		def _scan_all(text: str) -> set[str]:
			out = set()
			# hex
			for m in re.finditer(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b", text):
				h = m.group(0)
				if len(h) == 4:
					h = "#" + "".join([c*2 for c in h[1:]])
				out.add(h.upper())
			# rgb/rgba
			for m in re.finditer(r"rgba?\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})", text):
				r, g, b = (max(0, min(255, int(m.group(i)))) for i in (1,2,3))
				out.add(f"#{r:02X}{g:02X}{b:02X}")
			# hsl/hsla
			for m in re.finditer(r"hsla?\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})%\s*,\s*([0-9]{1,3})%", text):
				try:
					h, s, l = int(m.group(1)), int(m.group(2)), int(m.group(3))
					r, g, b = hsl_to_rgb((h, s, l))
					out.add(f"#{r:02X}{g:02X}{b:02X}")
				except Exception:
					continue
			return out
		found = _scan_all(html)
		if include_linked_css:
			for m in re.finditer(r'<link[^>]+rel=["\']?stylesheet["\']?[^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE):
				if css_limit <= 0:
					break
				href = urljoin(u, m.group(1))
				try:
					css = requests.get(href, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
					if css.ok:
						found |= _scan_all(css.text or "")
						css_limit -= 1
				except Exception:
					continue
		# simple near-duplicate filter
		unique = []
		for hx in found:
			rgb = (int(hx[1:3],16), int(hx[3:5],16), int(hx[5:7],16))
			if any(abs(rgb[0]-int(s[1:3],16))+abs(rgb[1]-int(s[3:5],16))+abs(rgb[2]-int(s[5:7],16)) < 40 for s in unique):
				continue
			unique.append(hx)
		return unique

	if st.button("Extract", type="primary"):
		if not url.strip():
			st.warning("Please enter a URL.")
		else:
			try:
				# Use global extractor if present, else fallback
				hexes = (globals().get("extract_palette_from_url")(url.strip(), include_linked_css=include_css, css_limit=3)
				         if "extract_palette_from_url" in globals()
				         else _extract_from_url_fallback(url.strip(), include_linked_css=include_css, css_limit=3))
				if not hexes:
					st.info("No colors detected. Try enabling linked stylesheets or a different page.")
					st.stop()
				# Build color dicts (hex, rgb, hsl)
				flat_colors = []
				for hx in hexes[:max_colors]:
					r, g, b = _hex_to_rgb(hx)
					h, s, l = _rgb_to_hsl(r, g, b)
					flat_colors.append({"hex": hx, "rgb": (r, g, b), "hsl": (h, s, l)})

				st.image(create_palette_image(flat_colors, width=800, height=120), caption="Extracted Palette", width="stretch")
				payload_web = palette_payload_from_colors(flat_colors)
				structured_web = payload_web["palette"]
				copy_web = payload_web["copy"]

				_render_vote_widget(structured_web, "web_current")
				def _short(s, n=42):
					s = (s or "").strip()
					return s if len(s) <= n else s[:n-1] + "…"
				_push_history_palette(f"Website: {_short(url, 42)}", structured_web)

				st.subheader("Primary")
				st.image(create_palette_image(structured_web["primary"], width=800, height=80), caption=None, width="stretch")
				st.code(copy_web["primary"]["hex"], language="text")
				st.code(copy_web["primary"]["rgb"], language="text")
				st.code(copy_web["primary"]["hsl"], language="text")

				st.subheader("Secondary")
				st.image(create_palette_image(structured_web["secondary"], width=800, height=80), caption=None, width="stretch")
				st.code(copy_web["secondary"]["hex"], language="text")
				st.code(copy_web["secondary"]["rgb"], language="text")
				st.code(copy_web["secondary"]["hsl"], language="text")

				st.subheader("Accent")
				st.image(create_palette_image(structured_web["accent"], width=800, height=80), caption=None, width="stretch")
				st.code(copy_web["accent"]["hex"], language="text")
				st.code(copy_web["accent"]["rgb"], language="text")
				st.code(copy_web["accent"]["hsl"], language="text")

				with st.expander("Structured JSON"):
					st.code(payload_web["json"], language="json")

				st.subheader("Accessibility")
				flat_all = structured_web["primary"] + structured_web["secondary"] + structured_web["accent"]
				render_accessibility_table(flat_all, "web_current")

				render_extras(structured_web, "web_current")
			except Exception as e:
				st.error(f"Error extracting colors: {str(e)}")