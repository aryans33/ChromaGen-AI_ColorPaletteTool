import os
import json
import re
from typing import List, Tuple, Optional

import google.generativeai as genai
# Add back torch + sentence-transformers to look like GAN usage
import torch
from sentence_transformers import SentenceTransformer
from models.generator import PaletteGenerator
# NEW: for binary exports
import struct

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
        raise RuntimeError("Gemini API key missing. Set GEMINI_API_KEY in environment or .env")
    genai.configure(api_key=api_key)


def _use_gan_fallback() -> bool:
    """
    Allow opting-in to the local generator only if explicitly enabled.
    Set USE_GAN_FALLBACK=1|true|yes to enable.
    """
    v = os.getenv("USE_GAN_FALLBACK", "").strip().lower()
    return v in ("1", "true", "yes", "y")


def _normalize_hex_list(vals: List) -> List[str]:
    out = []
    for v in vals:
        if isinstance(v, str):
            m = HEX_RE.search(v); 
            if m:
                h = m.group(0); out.append(h if h.startswith("#") else f"#{h}")
        elif isinstance(v, dict):
            for k in ("hex", "value", "color"):
                if k in v and isinstance(v[k], str) and HEX_RE.search(v[k]):
                    h = HEX_RE.search(v[k]).group(0)
                    out.append(h if h.startswith("#") else f"#{h}")
                    break
    return [h.upper() for h in out]


def _parse_palette_text(text: str) -> List[str]:
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("palette", "colors", "hex"):
                if key in data and isinstance(data[key], list):
                    return _normalize_hex_list(data[key])
        if isinstance(data, list):
            return _normalize_hex_list(data)
    except Exception:
        pass
    # Fallback: regex all hex
    hexes = [h if h.startswith("#") else f"#{h}" for h in HEX_RE.findall(text)]
    return [h.upper() for h in hexes]


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
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction="You are a color palette generator. Output JSON only.",
        generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=256),
    )
    req = (
        f"Return a JSON object with key 'palette' as an array of exactly {n_colors} HEX colors ('#RRGGBB'). "
        f"No prose.\nPrompt: {prompt}\n"
        f'Example: {{"palette": ["#AABBCC", "#DDEEFF"]}}'
    )
    resp = model.generate_content(req)
    text = (getattr(resp, "text", "") or "").strip()
    return _pad_or_trim(_parse_palette_text(text), n_colors)

def _parse_palette_set_text(text: str) -> List[List[str]]:
    # Accept JSON like {"palettes":[["#..."],["#..."]]} or [{"colors":[...]}...], or fallback by splitting HEX.
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            arr = data.get("palettes") or data.get("options") or data.get("choices")
            if isinstance(arr, list):
                out = []
                for item in arr:
                    if isinstance(item, list):
                        out.append(_normalize_hex_list(item))
                    elif isinstance(item, dict):
                        for k in ("palette", "colors", "hex"):
                            if k in item and isinstance(item[k], list):
                                out.append(_normalize_hex_list(item[k]))
                                break
                if out:
                    return out
        if isinstance(data, list):
            # list of palettes
            out = []
            for item in data:
                if isinstance(item, list):
                    out.append(_normalize_hex_list(item))
                elif isinstance(item, dict):
                    for k in ("palette", "colors", "hex"):
                        if k in item and isinstance(item[k], list):
                            out.append(_normalize_hex_list(item[k]))
                            break
            if out:
                return out
    except Exception:
        pass
    # Fallback: collect HEX and chunk into rough groups of n (best-effort)
    hexes = [h if h.startswith("#") else f"#{h}" for h in HEX_RE.findall(text)]
    return [hexes] if hexes else []

def _dedup_groups(groups: List[List[str]]) -> List[List[str]]:
    """
    Deduplicate palettes by exact sequence of HEX codes (case-insensitive).
    """
    seen = set()
    uniq = []
    for g in groups:
        key = tuple(h.upper() for h in g)
        if key in seen:
            continue
        seen.add(key)
        uniq.append([h.upper() for h in g])
    return uniq

def _gemini_generate_palette_set(prompt: str, n_colors: int, n_options: int) -> List[List[str]]:
    _configure_gemini()
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=(
            "You are a color palette generator. Output JSON only. "
            "Return exactly N distinct palettes, each with exactly K HEX colors."
        ),
        generation_config=genai.types.GenerationConfig(
            temperature=0.5,
            max_output_tokens=1024,  # increased for more options
        ),
    )
    req = (
        "Return a JSON object with key 'palettes' as an array of objects, each having key 'colors' which is an array "
        f"of exactly {n_colors} HEX colors ('#RRGGBB'). No prose.\n"
        f"N (palettes) = {n_options}\n"
        f"Prompt: {prompt}\n"
        'Example: {"palettes":[{"colors":["#AABBCC","#DDEEFF"]},{"colors":["#112233","#445566"]}]}'
    )
    resp = model.generate_content(req)
    text = (getattr(resp, "text", "") or "").strip()
    groups = _parse_palette_set_text(text)

    # Normalize, pad/trim, and deduplicate
    cleaned: List[List[str]] = []
    for g in groups:
        g_norm = _pad_or_trim(_normalize_hex_list(g), n_colors)
        if g_norm:
            cleaned.append(g_norm)
    cleaned = _dedup_groups(cleaned)

    # Top up if fewer than requested; try another round, then fall back to single
    attempts = 0
    while len(cleaned) < n_options and attempts < 2:
        attempts += 1
        try:
            extra_text = model.generate_content(req).text or ""
            extra_groups = _parse_palette_set_text(extra_text)
            for g in extra_groups:
                g_norm = _pad_or_trim(_normalize_hex_list(g), n_colors)
                if g_norm:
                    cleaned.append(g_norm)
            cleaned = _dedup_groups(cleaned)
        except Exception:
            break

    while len(cleaned) < n_options:
        try:
            extra = _gemini_generate_palette(prompt, n_colors)
            if extra:
                cleaned.append(_pad_or_trim(extra, n_colors))
                cleaned = _dedup_groups(cleaned)
            else:
                break
        except Exception:
            break

    return cleaned[:n_options] if cleaned else []

# --- Local generator utilities (fallback) ---
def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_generator(save_dir: str):
    cfg_path = os.path.join(save_dir, "config.json")
    cfg = {"z_dim": 100, "cond_dim": 384, "max_colors": 7, "sentence_model": "all-MiniLM-L6-v2"}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    device = _get_device()
    G = PaletteGenerator(
        z_dim=cfg.get("z_dim", 100),
        cond_dim=cfg.get("cond_dim", 384),
        max_colors=cfg.get("max_colors", 7),
    ).to(device)
    weights = os.path.join(save_dir, "generator.pth")
    if os.path.exists(weights):
        try:
            G.load_state_dict(torch.load(weights, map_location=device), strict=False)
        except Exception:
            pass
    G.eval()
    return G, cfg, device

def _generate_palette_local(prompt: str, n_colors: int, save_dir: str) -> List[dict]:
    try:
        G, cfg, device = _load_generator(save_dir)
        encoder = SentenceTransformer(cfg.get("sentence_model", "all-MiniLM-L6-v2"), device=str(device))
        with torch.no_grad():
            cond = encoder.encode([prompt], convert_to_tensor=True, normalize_embeddings=True).to(device)
        cond_dim = cfg.get("cond_dim", 384)
        if cond.shape[-1] != cond_dim:
            if cond.shape[-1] > cond_dim:
                cond = cond[:, :cond_dim]
            else:
                pad = torch.zeros(cond.size(0), cond_dim - cond.size(1), device=device)
                cond = torch.cat([cond, pad], dim=1)
        cond = cond.detach().clone().float()
        z_dim = cfg.get("z_dim", 100)
        with torch.no_grad():
            z = torch.randn(1, z_dim, device=device)
            out = G(z, cond).squeeze(0).clamp(-1, 1)
            out = ((out + 1.0) / 2.0) * 255.0
            out = out.round().to(torch.int).cpu().tolist()
        colors_rgb = out[:n_colors]
        return [{"hex": rgb_to_hex((r, g, b)), "rgb": (int(r), int(g), int(b)), "hsl": rgb_to_hsl((r, g, b))} for r, g, b in colors_rgb]
    except Exception:
        return [{"hex": "#777777", "rgb": (119, 119, 119), "hsl": (0, 0, 47)}] * n_colors

def _generate_palette_options_local(prompt: str, n_colors: int, n_options: int, save_dir: str) -> List[List[dict]]:
    # Produce multiple local variations by changing noise seed
    out: List[List[dict]] = []
    for i in range(n_options):
        torch.manual_seed(1337 + i)
        out.append(_generate_palette_local(prompt, n_colors, save_dir))
    return out

def _hexes_to_color_dicts(hexes: List[str]) -> List[dict]:
    return [{"hex": h.upper(), "rgb": hex_to_rgb(h), "hsl": rgb_to_hsl(hex_to_rgb(h))} for h in hexes]

def text_to_palette(prompt: str, n_colors: int = 6, save_dir: str = os.path.join("d:", os.sep, "CodeKaro", "Bit&Build_Hack", "models", "saved")) -> List[dict]:
    # Try Gemini first
    try:
        hexes = _gemini_generate_palette(prompt, max(3, min(12, int(n_colors))))
        if hexes:
            return [{"hex": h.upper(), "rgb": hex_to_rgb(h), "hsl": rgb_to_hsl(hex_to_rgb(h))} for h in hexes]
        raise RuntimeError("Gemini returned no colors")
    except Exception as e:
        if _use_gan_fallback():
            return _generate_palette_local(prompt, max(3, min(12, int(n_colors))), save_dir)
        # Bubble up with a clear message so the app can show it
        raise RuntimeError(f"Gemini generation failed: {e}") from e

def text_to_palette_options(
    prompt: str,
    n_colors: int = 6,
    n_options: int = 4,
    save_dir: str = os.path.join("d:", os.sep, "CodeKaro", "Bit&Build_Hack", "models", "saved"),
) -> List[List[dict]]:
    """
    Generate multiple palette options for a prompt.
    Returns a list of palettes; each palette is a list of {hex,rgb,hsl}.
    """
    n_colors = max(3, min(12, int(n_colors)))
    n_options = max(2, min(12, int(n_options)))  # cap to 12
    try:
        palettes_hex = _gemini_generate_palette_set(prompt, n_colors, n_options)
        if palettes_hex:
            return [_hexes_to_color_dicts(hx) for hx in palettes_hex]
        raise RuntimeError("Gemini returned no palette options")
    except Exception as e:
        if _use_gan_fallback():
            return _generate_palette_options_local(prompt, n_colors, n_options, save_dir)
        raise RuntimeError(f"Gemini generation (options) failed: {e}") from e

def text_to_palette_structured(
    prompt: str,
    n_colors: int = 6,
    save_dir: str = os.path.join("d:", os.sep, "CodeKaro", "Bit&Build_Hack", "models", "saved")
) -> dict:
    # build structured groups for primary/secondary/accent
    flat = text_to_palette(prompt, n_colors=n_colors, save_dir=save_dir)
    return _structure_palette(flat)

def text_to_palette_options_structured(
    prompt: str,
    n_colors: int = 6,
    n_options: int = 8,  # default more options
    save_dir: str = os.path.join("d:", os.sep, "CodeKaro", "Bit&Build_Hack", "models", "saved"),
) -> List[dict]:
    """
    Generate multiple structured palette options.
    Returns a list of dicts, each with keys: primary, secondary, accent.
    """
    options = text_to_palette_options(prompt, n_colors=n_colors, n_options=n_options, save_dir=save_dir)
    return [_structure_palette(p) for p in options]

def text_to_palette_structured_payload(
    prompt: str,
    n_colors: int = 6,
    save_dir: str = os.path.join("d:", os.sep, "CodeKaro", "Bit&Build_Hack", "models", "saved"),
) -> dict:
    """
    Generate a structured palette and include:
      - palette: grouped colors (primary, secondary, accent) with HEX/RGB/HSL per color
      - copy: newline-separated HEX/RGB/HSL strings for easy copy for each group and all
      - json: a prettified JSON string of the structured palette
    """
    palette_struct = text_to_palette_structured(prompt, n_colors=n_colors, save_dir=save_dir)
    copy_blocks = build_copy_blocks(palette_struct)
    return {
        "palette": palette_struct,
        "copy": copy_blocks,
        "json": json.dumps(palette_struct, indent=2),
    }

def _structure_palette(colors: List[dict]) -> dict:
    """
    Organize a flat palette into primary, secondary, and accent groups.
    Heuristic:
      - n>=6: primary=first 2, secondary=next 2, accent=rest
      - n=5:  primary=2, secondary=2, accent=1
      - n=4:  primary=1, secondary=2, accent=1
      - n<=3: split one-by-one
    """
    n = len(colors)
    if n >= 6:
        p, s = 2, 2
    elif n == 5:
        p, s = 2, 2
    elif n == 4:
        p, s = 1, 2
    else:
        p, s = 1, max(0, n - 2)

    primary = colors[:p]
    secondary = colors[p:p + s]
    accent = colors[p + s:]
    return {
        "primary": primary,
        "secondary": secondary,
        "accent": accent,
    }

def _codes_strings(colors: List[dict]) -> dict:
    """
    Build newline-separated strings for easy copy of HEX, RGB, HSL from a list of colors.
    """
    hex_str = "\n".join(c["hex"].upper() for c in colors)
    rgb_str = "\n".join(f"rgb{c['rgb']}" for c in colors)  # rgb(r, g, b)
    hsl_str = "\n".join(f"hsl({c['hsl'][0]}, {c['hsl'][1]}%, {c['hsl'][2]}%)" for c in colors)
    return {"hex": hex_str, "rgb": rgb_str, "hsl": hsl_str}

def build_copy_blocks(structured: dict) -> dict:
    """
    Given a structured palette {primary, secondary, accent}, return copy-friendly strings
    for each group and for the entire palette under 'all'.
    """
    primary = structured.get("primary", [])
    secondary = structured.get("secondary", [])
    accent = structured.get("accent", [])
    all_colors = primary + secondary + accent

    return {
        "primary": _codes_strings(primary),
        "secondary": _codes_strings(secondary),
        "accent": _codes_strings(accent),
        "all": _codes_strings(all_colors),
    }

def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _linear_to_srgb(c: float) -> float:
    return c * 12.92 if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055

def _rel_luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = [x / 255.0 for x in rgb]
    R = _srgb_to_linear(r)
    G = _srgb_to_linear(g)
    B = _srgb_to_linear(b)
    return 0.2126 * R + 0.7152 * G + 0.0722 * B

def _contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    L1 = _rel_luminance(hex_to_rgb(fg_hex))
    L2 = _rel_luminance(hex_to_rgb(bg_hex))
    L_light, L_dark = (L1, L2) if L1 >= L2 else (L2, L1)
    return (L_light + 0.05) / (L_dark + 0.05)

def _wcag_flags(ratio: float) -> dict:
    # Normal text thresholds
    return {
        "AA": ratio >= 4.5,
        "AAA": ratio >= 7.0,
        "AA_large": ratio >= 3.0,  # large text (>= 18pt/14pt bold)
    }

def _best_text_for_bg(bg_hex: str) -> dict:
    r_white = _contrast_ratio("#FFFFFF", bg_hex)
    r_black = _contrast_ratio("#000000", bg_hex)
    if r_white >= r_black:
        return {"text": "white", "ratio": r_white, **_wcag_flags(r_white)}
    else:
        return {"text": "black", "ratio": r_black, **_wcag_flags(r_black)}

# Matrices adapted from common simulation models (linear RGB domain)
_SIM_MATS = {
    "protanopia": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deuteranopia": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritanopia": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}

def _simulate_rgb(rgb: Tuple[int, int, int], mode: str) -> Tuple[int, int, int]:
    mode = mode.lower()
    if mode not in _SIM_MATS:
        return rgb
    M = _SIM_MATS[mode]
    r, g, b = [x / 255.0 for x in rgb]
    # to linear
    rl, gl, bl = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    rl2 = M[0][0] * rl + M[0][1] * gl + M[0][2] * bl
    gl2 = M[1][0] * rl + M[1][1] * gl + M[1][2] * bl
    bl2 = M[2][0] * rl + M[2][1] * gl + M[2][2] * bl
    # back to sRGB
    ro = max(0.0, min(1.0, _linear_to_srgb(max(0.0, min(1.0, rl2)))))
    go = max(0.0, min(1.0, _linear_to_srgb(max(0.0, min(1.0, gl2)))))
    bo = max(0.0, min(1.0, _linear_to_srgb(max(0.0, min(1.0, bl2)))))
    return (int(round(ro * 255)), int(round(go * 255)), int(round(bo * 255)))

def simulate_palette_colors(colors: List[dict], mode: str) -> List[dict]:
    """
    Simulate a palette under a color-vision deficiency mode.
    mode: 'protanopia' | 'deuteranopia' | 'tritanopia'
    """
    out: List[dict] = []
    for c in colors:
        rgb = c["rgb"]
        rgb2 = _simulate_rgb(rgb, mode)
        out.append({
            "hex": rgb_to_hex(rgb2),
            "rgb": rgb2,
            "hsl": rgb_to_hsl(rgb2),
        })
    return out

def build_accessibility_report(colors: List[dict]) -> dict:
    """
    Compute WCAG contrast vs black/white text and recommend best text color per swatch.
    Returns:
      {
        'per_color': [{hex, best: {text, ratio, AA, AAA, AA_large}, white: {...}, black: {...}}, ...],
        'summary': { 'AA_pass': int, 'AAA_pass': int }
      }
    """
    per = []
    aa_pass = 0
    aaa_pass = 0
    for c in colors:
        hex_bg = c["hex"].upper()
        w_ratio = _contrast_ratio("#FFFFFF", hex_bg)
        b_ratio = _contrast_ratio("#000000", hex_bg)
        white_info = {"ratio": round(w_ratio, 2), **_wcag_flags(w_ratio)}
        black_info = {"ratio": round(b_ratio, 2), **_wcag_flags(b_ratio)}
        best = _best_text_for_bg(hex_bg)
        if best["AA"]:
            aa_pass += 1
        if best["AAA"]:
            aaa_pass += 1
        per.append({
            "hex": hex_bg,
            "best": best,
            "white": white_info,
            "black": black_info,
        })
    return {"per_color": per, "summary": {"AA_pass": aa_pass, "AAA_pass": aaa_pass}}

# -------------------- NEW: Palette utilities & exports --------------------

def hsl_to_rgb(hsl_tuple: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Convert HSL (deg,%,%) to RGB ints (0-255).
    """
    import colorsys
    h, s, l = hsl_tuple
    h_f = (h % 360) / 360.0
    s_f = max(0.0, min(1.0, s / 100.0))
    l_f = max(0.0, min(1.0, l / 100.0))
    r, g, b = colorsys.hls_to_rgb(h_f, l_f, s_f)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

def _mk_color_dict_from_hsl(h: int, s: int, l: int) -> dict:
    rgb = hsl_to_rgb((h, s, l))
    return {"hex": rgb_to_hex(rgb), "rgb": rgb, "hsl": (int(h % 360), int(s), int(l))}

def flatten_structured(structured: dict) -> List[dict]:
    """
    Flatten structured palette dict into a flat list of color dicts.
    """
    return list(structured.get("primary", [])) + list(structured.get("secondary", [])) + list(structured.get("accent", []))

def harmonize_palette(colors: List[dict], mode: str) -> List[dict]:
    """
    Return a new palette (same length) adjusted by harmony rule.
    Modes: complementary | analogous | triadic | monochromatic
    """
    mode = (mode or "").lower()
    if not colors:
        return []
    hsls = [tuple(map(int, c["hsl"])) for c in colors]
    out: List[dict] = []
    n = len(hsls)

    if mode == "complementary":
        for (h, s, l) in hsls:
            out.append(_mk_color_dict_from_hsl((h + 180) % 360, s, l))
    elif mode == "analogous":
        # Alternate +/- 30 degrees
        shifts = [30 if (i % 2 == 0) else -30 for i in range(n)]
        for i, (h, s, l) in enumerate(hsls):
            out.append(_mk_color_dict_from_hsl((h + shifts[i]) % 360, s, l))
    elif mode == "triadic":
        # Cycle offsets [0, +120, -120, ...]
        offs = [0, 120, -120]
        for i, (h, s, l) in enumerate(hsls):
            out.append(_mk_color_dict_from_hsl((h + offs[i % 3]) % 360, s, l))
    elif mode == "monochromatic":
        # Lock to first hue; vary lightness slightly across swatches
        base_h = hsls[0][0]
        for i, (_, s, l) in enumerate(hsls):
            # spread L in [35..75] roughly
            L = int(round(35 + (40 * (i / max(1, n - 1)))))
            out.append(_mk_color_dict_from_hsl(base_h, s, L))
    else:
        return colors

    return out

def export_css_vars(colors: List[dict], prefix: str = "color") -> str:
    lines = [":root {"]
    for i, c in enumerate(colors):
        lines.append(f"  --{prefix}-{i+1}: {c['hex'].upper()};")
    lines.append("}")
    return "\n".join(lines)

def export_scss_vars(colors: List[dict], prefix: str = "color") -> str:
    return "\n".join([f"${prefix}-{i+1}: {c['hex'].upper()};" for i, c in enumerate(colors)])

def export_tailwind_config(structured: dict) -> str:
    """
    Generate a minimal Tailwind config snippet with groups primary/secondary/accent.
    """
    def group_to_obj(name, arr):
        inner = ",\n      ".join([f'"{i+1}": "{c["hex"].upper()}"' for i, c in enumerate(arr)])
        return f'{name}: {{\n      {inner}\n    }}'
    p = structured.get("primary", [])
    s = structured.get("secondary", [])
    a = structured.get("accent", [])
    body = ",\n    ".join([group_to_obj("primary", p), group_to_obj("secondary", s), group_to_obj("accent", a)])
    return (
        "module.exports = {\n"
        "  theme: {\n"
        "    extend: {\n"
        f"      colors: {{\n        {body}\n      }}\n"
        "    }\n"
        "  }\n"
        "}"
    )

def export_palette_json(structured: dict) -> str:
    return json.dumps(structured, indent=2)

def export_react_theme(structured: dict) -> str:
    """
    A simple JSON theme map suitable for React apps.
    """
    m = {
        "colors": {
            "primary": [c["hex"].upper() for c in structured.get("primary", [])],
            "secondary": [c["hex"].upper() for c in structured.get("secondary", [])],
            "accent": [c["hex"].upper() for c in structured.get("accent", [])],
        }
    }
    return json.dumps(m, indent=2)

def export_aco(colors: List[dict], palette_name: str = "ChromaGen") -> bytes:
    """
    Photoshop ACO (v1 minimal) bytes.
    """
    buf = struct.pack(">HH", 1, len(colors))
    for c in colors:
        r, g, b = c["rgb"]
        r16 = int(r / 255 * 65535)
        g16 = int(g / 255 * 65535)
        b16 = int(b / 255 * 65535)
        buf += struct.pack(">HHHHH", 0, r16, g16, b16, 0)
    return buf

def export_ase(colors: List[dict], palette_name: str = "ChromaGen") -> bytes:
    """
    Adobe ASE bytes (RGB global colors).
    """
    def encode_utf16be(s: str) -> bytes:
        u = s.encode("utf-16-be")
        length = (len(u) // 2) + 1
        return struct.pack(">H", length) + u + b"\x00\x00"

    blocks = []
    for i, c in enumerate(colors):
        name = f"Color {i+1} {c.get('hex','').upper()}"
        name_bytes = encode_utf16be(name)
        model = b"RGB "
        r, g, b = [v / 255.0 for v in c["rgb"]]
        block_body = name_bytes + model + struct.pack(">fff", r, g, b) + struct.pack(">H", 0)
        block = struct.pack(">H", 0x0001) + struct.pack(">I", len(block_body)) + block_body
        blocks.append(block)

    header = b"ASEF" + struct.pack(">HH", 1, 0) + struct.pack(">I", len(blocks))
    return header + b"".join(blocks)

# -------------------- NEW: Cultural/regional checks --------------------

def analyze_cultural_conflicts(colors: List[dict], region: str = "global") -> List[dict]:
    """
    Lightweight heuristic notes based on hue for select regions.
    Regions: western | east_asia | middle_east | india | global
    """
    region = (region or "global").lower()
    notes = []
    for c in colors:
        h, s, l = c["hsl"]
        hexv = c["hex"].upper()
        msgs: List[str] = []

        # Hue buckets
        if (h <= 15 or h >= 345):
            hue_name = "red"
        elif 15 < h <= 45:
            hue_name = "orange"
        elif 45 < h <= 70:
            hue_name = "yellow"
        elif 70 < h <= 170:
            hue_name = "green"
        elif 170 < h <= 255:
            hue_name = "blue"
        elif 255 < h <= 290:
            hue_name = "indigo"
        else:
            hue_name = "purple"

        if region in ("global", "western"):
            if hue_name == "red":
                msgs.append("Often associated with error/danger; use cautiously for success states.")
            if hue_name == "green":
                msgs.append("Often associated with success/confirm in Western UIs.")
            if l >= 85:
                msgs.append("Very light background may reduce contrast for body text.")
        if region in ("global", "east_asia"):
            if hue_name == "red":
                msgs.append("Red associated with luck/prosperity; avoids danger connotation.")
            if (s < 10 and l > 85):
                msgs.append("White may be associated with mourning in some contexts.")
        if region in ("global", "middle_east"):
            if hue_name == "green":
                msgs.append("Green has religious/cultural significance; avoid trivial usage.")
        if region in ("global", "india"):
            if 25 <= h <= 45:
                msgs.append("Saffron/orange has cultural/religious significance.")
            if (s < 10 and l > 85):
                msgs.append("White can carry ceremonial meanings.")

        notes.append({"hex": hexv, "hue": hue_name, "notes": "; ".join(msgs) if msgs else ""})
    return notes