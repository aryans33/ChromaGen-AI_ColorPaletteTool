import os
import json
import re
from typing import List, Tuple

import google.generativeai as genai

# --- Helpers ---
def rgb_to_hex(rgb_tuple: Tuple[int, int, int]) -> str:
    r, g, b = rgb_tuple
    return '#{:02X}{:02X}{:02X}'.format(int(r), int(g), int(b))


def rgb_to_hsl(rgb_tuple: Tuple[int, int, int]) -> Tuple[int, int, int]:
    import colorsys
    r, g, b = [x / 255.0 for x in rgb_tuple]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (int(h * 360), int(s * 100), int(l * 100))


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


HEX_RE = re.compile(r"#?[0-9a-fA-F]{6}")


def _configure_gemini() -> None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Add it to .env or environment.")
    genai.configure(api_key=api_key)


def _parse_palette_text(text: str) -> List[str]:
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("palette", "colors", "hex"):
                if key in data and isinstance(data[key], list):
                    vals = data[key]
                    return _normalize_hex_list(vals)
        if isinstance(data, list):
            return _normalize_hex_list(data)
    except Exception:
        pass
    # Fallback: regex all hex
    hexes = [h if h.startswith("#") else f"#{h}" for h in HEX_RE.findall(text)]
    return [h.upper() for h in hexes]


def _normalize_hex_list(vals: List) -> List[str]:
    out = []
    for v in vals:
        if isinstance(v, str):
            m = HEX_RE.search(v)
            if m:
                h = m.group(0)
                out.append(h if h.startswith("#") else f"#{h}")
        elif isinstance(v, dict):
            # common schema: {"hex":"#AABBCC"} or {"value":"#..."}
            for k in ("hex", "value", "color"):
                if k in v and isinstance(v[k], str) and HEX_RE.search(v[k]):
                    h = HEX_RE.search(v[k]).group(0)
                    out.append(h if h.startswith("#") else f"#{h}")
                    break
    return [h.upper() for h in out]


def _pad_or_trim(hexes: List[str], n: int) -> List[str]:
    if len(hexes) >= n:
        return hexes[:n]
    # Pad by cycling existing values
    out = list(hexes)
    i = 0
    while len(out) < n and hexes:
        out.append(hexes[i % len(hexes)])
        i += 1
    # If still empty, provide neutral defaults
    if not out:
        out = ["#777777"] * n
    return out[:n]


def _gemini_generate_palette(prompt: str, n_colors: int) -> List[str]:
    _configure_gemini()
    system_instruction = (
        "You are a color palette generator. Output exactly n HEX colors as JSON. "
        "Prefer diverse yet harmonious colors matching the prompt. No prose."
    )
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_instruction,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=256,
        ),
    )
    user = (
        "Given the prompt, return a JSON object with a key 'palette' that is an array of exactly "
        f"{n_colors} HEX colors (format '#RRGGBB'). No other keys or text.\n"
        f"Prompt: {prompt}\n"
        f"Return JSON like: {{\"palette\": [\"#AABBCC\", \"#DDEEFF\", ...]}}"
    )
    resp = model.generate_content(user)
    text = (getattr(resp, "text", "") or "").strip()
    hexes = _parse_palette_text(text)
    return _pad_or_trim(hexes, n_colors)


def text_to_palette(
    prompt: str,
    n_colors: int = 6,
    save_dir: str = os.path.join("d:", os.sep, "CodeKaro", "Bit&Build_Hack", "models", "saved"),
) -> List[dict]:
    """
    Generate a color palette using Gemini only (no GAN).
    Returns list of {hex, rgb, hsl}.
    """
    n_colors = max(3, min(12, int(n_colors)))
    hexes = _gemini_generate_palette(prompt, n_colors)
    palette = []
    for h in hexes:
        r, g, b = hex_to_rgb(h)
        palette.append({"hex": h.upper(), "rgb": (r, g, b), "hsl": rgb_to_hsl((r, g, b))})
    return palette