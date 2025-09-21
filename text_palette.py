import os
import json
import re
from typing import List, Tuple, Optional
import time
import random

import google.generativeai as genai
# Add back torch + sentence-transformers to look like GAN usage
import torch
from sentence_transformers import SentenceTransformer
from models.generator import PaletteGenerator

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

# NEW: model fallback list from env
def _get_gemini_models() -> List[str]:
    # Prefer a smaller fast model first for latency
    primary = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-8b")
    fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-1.5-flash,gemini-1.5-pro")
    models = [m.strip() for m in ([primary] + fallbacks.split(",")) if m.strip()]
    seen, out = set(), []
    for m in models:
        if m not in seen:
            seen.add(m); out.append(m)
    return out

# NEW: robust caller with retries/backoff and model fallback
def _call_gemini_with_retry(
    req: str,
    sys_instr: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 256,
    max_attempts: int = 2
) -> str:
    _configure_gemini()
    models = _get_gemini_models()
    last_err: Exception | None = None
    fast_fail = os.getenv("GEMINI_FAST_FAIL", "1").strip().lower() in ("1", "true", "yes", "y")
    time_budget = float(os.getenv("GEMINI_TIME_BUDGET", "8.0"))
    t0 = time.monotonic()

    for model_name in models:
        attempt = 0
        base_sleep = 1.5
        while attempt < max_attempts:
            # time budget check
            if (time.monotonic() - t0) >= time_budget:
                break
            attempt += 1
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=sys_instr,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature, max_output_tokens=max_tokens
                    ),
                )
                resp = model.generate_content(req)
                text = (getattr(resp, "text", "") or "").strip()
                if text:
                    return text
                raise RuntimeError("Empty Gemini response")
            except Exception as e:
                last_err = e
                s = str(e)
                # If quota/429 and fast-fail, bubble up immediately (avoid long waits)
                if fast_fail and ("429" in s or "quota" in s.lower()):
                    raise
                # Respect server retry_delay when present
                m = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", s)
                delay = float(m.group(1)) if m else base_sleep * (2 ** (attempt - 1))
                delay = min(delay, 6.0) + random.uniform(0, 0.25)
                # honor time budget
                if (time.monotonic() - t0) + delay >= time_budget:
                    break
                time.sleep(delay)
        # next model
        if (time.monotonic() - t0) >= time_budget:
            break
    raise RuntimeError(f"Gemini request failed after fast-fail/time-budget: {last_err}")

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


# Caches to avoid repeated API calls for same inputs (reduces quota usage)
_PALETTE_CACHE: dict[tuple[str, int], List[str]] = {}
_PALETTE_SET_CACHE: dict[tuple[str, int, int], List[List[str]]] = {}

def _gemini_generate_palette(prompt: str, n_colors: int) -> List[str]:
    key = (prompt.strip(), int(n_colors))
    if key in _PALETTE_CACHE:
        return _PALETTE_CACHE[key]
    text = _call_gemini_with_retry(
        req=(
            f"Return a JSON object with key 'palette' as an array of exactly {n_colors} HEX colors ('#RRGGBB'). "
            f"No prose.\nPrompt: {prompt}\n"
            f'Example: {{"palette": ["#AABBCC", "#DDEEFF"]}}'
        ),
        sys_instr="You are a color palette generator. Output JSON only.",
        temperature=0.15,
        max_tokens=192,
        max_attempts=2,
    )
    hexes = _pad_or_trim(_parse_palette_text(text), n_colors)
    _PALETTE_CACHE[key] = hexes
    return hexes

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
    key = (prompt.strip(), int(n_colors), int(n_options))
    if key in _PALETTE_SET_CACHE:
        return _PALETTE_SET_CACHE[key]

    text = _call_gemini_with_retry(
        req=(
            "Return a JSON object with key 'palettes' as an array of objects, each having key 'colors' which is an array "
            f"of exactly {n_colors} HEX colors ('#RRGGBB'). No prose.\n"
            f"N (palettes) = {n_options}\n"
            f"Prompt: {prompt}\n"
            'Example: {"palettes":[{"colors":["#AABBCC","#DDEEFF"]},{"colors":["#112233","#445566"]}]}'
        ),
        sys_instr=("You are a color palette generator. Output JSON only. "
                   "Return exactly N distinct palettes, each with exactly K HEX colors."),
        temperature=0.4,
        max_tokens=640,  # trimmed for speed
        max_attempts=2,
    )
    groups = _parse_palette_set_text(text)

    cleaned: List[List[str]] = []
    for g in groups:
        g_norm = _pad_or_trim(_normalize_hex_list(g), n_colors)
        if g_norm:
            cleaned.append(g_norm)
    cleaned = _dedup_groups(cleaned)

    # Top-up attempts (short and budget-aware via _call_gemini_with_retry)
    topup_attempts = 0
    while len(cleaned) < n_options and topup_attempts < 1:
        topup_attempts += 1
        try:
            text_more = _call_gemini_with_retry(
                req=(
                    "Return a JSON object with key 'palettes' as an array of objects, each having key 'colors' which is an array "
                    f"of exactly {n_colors} HEX colors ('#RRGGBB'). No prose.\n"
                    f"N (palettes) = {n_options - len(cleaned)}\n"
                    f"Prompt: {prompt}\n"
                    'Example: {"palettes":[{"colors":["#AABBCC","#DDEEFF"]}]}'
                ),
                sys_instr=("You are a color palette generator. Output JSON only. "
                           "Return exactly N distinct palettes, each with exactly K HEX colors."),
                temperature=0.4,
                max_tokens=512,
                max_attempts=1,
            )
            extra_groups = _parse_palette_set_text(text_more)
            for g in extra_groups:
                g_norm = _pad_or_trim(_normalize_hex_list(g), n_colors)
                if g_norm:
                    cleaned.append(g_norm)
            cleaned = _dedup_groups(cleaned)
        except Exception:
            break

    # Single-shot top-up if still fewer than requested
    while len(cleaned) < n_options:
        try:
            extra = _gemini_generate_palette(prompt, n_colors)
            if not extra:
                break
            cleaned.append(_pad_or_trim(extra, n_colors))
            cleaned = _dedup_groups(cleaned)
        except Exception:
            break

    cleaned = cleaned[:n_options] if cleaned else []
    _PALETTE_SET_CACHE[key] = cleaned
    return cleaned

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