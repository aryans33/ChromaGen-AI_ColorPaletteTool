import os
import numpy as np
import pandas as pd
from PIL import Image, ImageColor
from sklearn.cluster import KMeans
import colorsys
import json
from typing import List, Dict

# Set environment variable to suppress joblib warning
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

def extract_palette_from_image(uploaded_file, n_colors=8):
    """
    Extract dominant colors from an image using K-means clustering.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        n_colors: Number of colors to extract (default: 8)
    
    Returns:
        dict: Contains 'colors' list with hex, rgb, hsl values and 'image' object
    """
    try:
        image = Image.open(uploaded_file)
        
        # Resize large images for performance
        h, w = image.size
        if (h > 1024) or (w > 1024):
            image = image.resize((int(h/2), int(w/2)))
        if (h > 2048) or (w > 2048):
            image = image.resize((int(h/3), int(w/3)))
        if (h > 3096) or (w > 3096):
            image = image.resize((int(h/4), int(w/4)))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Flatten image data
        data = pd.DataFrame(list(image.getdata()), columns=['R', 'G', 'B'])
        
        # Apply K-means clustering with updated parameters
        kmeans = KMeans(
            n_clusters=n_colors, 
            n_init=10, 
            random_state=42,
            init='k-means++'
        ).fit(data)
        centers = kmeans.cluster_centers_.astype(int)
        
        # Sort by luminance (brightest to darkest)
        luminance = [0.2126*r + 0.7152*g + 0.0722*b for r, g, b in centers]
        sorted_indices = np.argsort(luminance)[::-1]  # Reverse for bright to dark
        centers_sorted = centers[sorted_indices]
        
        # Convert to various formats
        colors = []
        for color in centers_sorted:
            r, g, b = color
            hex_color = rgb_to_hex((r, g, b))
            hsl_color = rgb_to_hsl((r, g, b))
            
            colors.append({
                'hex': hex_color,
                'rgb': (int(r), int(g), int(b)),
                'hsl': hsl_color
            })
        
        return {
            'colors': colors,
            'image': image
        }
    
    except Exception as e:
        raise Exception(f'Error processing image: {str(e)}')

def rgb_to_hex(rgb_tuple):
    """Convert RGB tuple to HEX string."""
    r, g, b = rgb_tuple
    return '#{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b))

def hex_to_rgb(hex_color):
    """Convert HEX string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hsl(rgb_tuple):
    """Convert RGB tuple to HSL tuple."""
    r, g, b = [x/255.0 for x in rgb_tuple]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (int(h*360), int(s*100), int(l*100))

def hsl_to_rgb(hsl_tuple):
    """Convert HSL tuple to RGB tuple."""
    h, s, l = hsl_tuple[0]/360.0, hsl_tuple[1]/100.0, hsl_tuple[2]/100.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r*255), int(g*255), int(b*255))

def create_palette_image(colors, width=400, height=100):
    """Create a visual representation of the color palette."""
    palette_img = Image.new('RGB', (width, height))
    color_width = width // len(colors)
    
    pixels = palette_img.load()
    for i, color_data in enumerate(colors):
        rgb = color_data['rgb']
        for x in range(i * color_width, min((i + 1) * color_width, width)):
            for y in range(height):
                pixels[x, y] = rgb
    
    return palette_img

def _structure_palette(colors: List[dict]) -> dict:
	"""
	Group a flat list into primary, secondary, and accent using the same heuristic as text.
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
	return {"primary": primary, "secondary": secondary, "accent": accent}

def _codes_strings(colors: List[dict]) -> dict:
	"""
	Newline-separated strings for easy copy (HEX/RGB/HSL).
	"""
	hex_str = "\n".join(c["hex"].upper() for c in colors)
	rgb_str = "\n".join(f"rgb{c['rgb']}" for c in colors)
	hsl_str = "\n".join(f"hsl({c['hsl'][0]}, {c['hsl'][1]}%, {c['hsl'][2]}%)" for c in colors)
	return {"hex": hex_str, "rgb": rgb_str, "hsl": hsl_str}

def build_copy_blocks(structured: dict) -> dict:
	"""
	Copy-friendly strings for each group and for the entire palette.
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

def palette_payload_from_colors(colors: List[dict]) -> dict:
	"""
	Build a payload with grouped palette, copy blocks, and pretty JSON from a flat colors list.
	"""
	structured = _structure_palette(colors)
	return {
		"palette": structured,
		"copy": build_copy_blocks(structured),
		"json": json.dumps(structured, indent=2),
	}

def extract_palette_from_image_structured_payload(file_like, n_colors: int = 6) -> dict:
	"""
	Run extraction, then return the structured payload (primary/secondary/accent + copy + json).
	"""
	res = extract_palette_from_image(file_like, n_colors=n_colors)
	return palette_payload_from_colors(res["colors"])
