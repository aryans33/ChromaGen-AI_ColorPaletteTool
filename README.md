# ChromaGen – AI Color Palette Generator

Streamlit app to generate production‑ready color palettes from:

- Image: extract dominant colors.
- Text: prompt to palette.
- Website: parse inline/linked CSS colors.
  Includes structured outputs (primary/secondary/accent), accessibility tools, exports, typography suggestions, live previews, and a tiny REST‑like API.

## Features

- Modes
  - Image → Palette with multiple variants and selection.
  - Text → Palette (single + options).
  - Website → Palette (follows linked stylesheets, de‑dupes near colors).
- Structured palette
  - Primary, Secondary, Accent groups with Hex/RGB/HSL copy blocks and JSON.
- Accessibility & UX
  - WCAG AA/AAA contrast table per color.
  - Color‑blindness simulations (Protanopia/Deuteranopia/Tritanopia).
  - Contrast previews with Auto/Black/White/custom palette text.
  - Dyslexia‑friendly previews with font/size/line-height/letter-spacing controls.
  - Live landing preview + storyboard mock.
  - Adaptive Theme (Light + Dark) variants.
- Exports
  - CSS variables, SCSS vars, Tailwind config, React theme JSON, palette JSON.
  - Adobe swatches: ACO and ASE.
- Collaboration & History
  - Simple local upvote/downvote per palette.
  - History with thumbnails, load/unload, delete, clear.
- REST‑like API
  - `/?api=1&prompt=<text>&n=<count>` returns structured palette JSON.

## Project structure

- `app.py` – Streamlit UI and orchestration.
- `image_palette.py` – Image processing utilities (extraction, preview image, payload).
- `text_palette.py` – Text/LLM palette generation, accessibility, exports, utilities.
- `.env` – Environment variables (not committed in production).

## Requirements

- Python 3.10+
- pip packages:
  - streamlit
  - python-dotenv
  - pillow
  - requests

Install:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install streamlit python-dotenv pillow requests
```

## Environment variables

Create a `.env` in the project root:

```
# optional – CPU cap for workers
LOKY_MAX_CPU_COUNT=4

# optional – used by text generation (if your backend requires it)
GEMINI_API_KEY=YOUR_KEY

# optional – improves typography suggestions (Google Fonts Web API)
GOOGLE_FONTS_API_KEY=YOUR_KEY
```

Do not commit real keys to version control.

## Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Usage

- Image to Palette
  - Upload an image, adjust “Number of colors”, review structured palette, run accessibility checks, export assets, and save to history.
- Text to Palette
  - Enter a short prompt, generate, compare options, and select one to deep‑dive (accessibility, exports, previews).
- Website to Palette
  - Paste a URL, toggle “Include linked stylesheets” (recommended), choose “Max colors”, extract, and proceed as above.

Tips:

- Use the “Dark mode” toggle for app theme only (does not affect export values).
- Each section exposes “Extras” tabs for Contrast, Dyslexia, Exports, Storyboard, Live Preview, Typography, and Adaptive Theme.

## REST‑like API

Quick JSON palette for programmatic use:

- Endpoint: `GET /?api=1&prompt=<text>&n=<count>`
- Example:

```bash
curl "http://localhost:8501/?api=1&prompt=sunset+beach&n=6"
```

Response:

```json
{
  "primary": [...],
  "secondary": [...],
  "accent": [...]
}
```

## Troubleshooting

- “No colors detected” for a site: enable “Include linked stylesheets”, try the homepage, or increase “Max colors”.
- Network timeouts: check connectivity/firewall; the app uses a short timeout for CSS fetches.
- Fonts API suggestions do nothing: set `GOOGLE_FONTS_API_KEY` or proceed without (fallback works).
- Avoid committing `.env` with real keys.

## License

Add a license file (e.g., MIT) appropriate for your use.

## Acknowledgements

Built with Streamlit, Pillow, and Google Fonts.
