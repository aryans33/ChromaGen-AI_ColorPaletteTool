import os
import re
import json
import math
import random
from typing import List, Tuple
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
from sentence_transformers import SentenceTransformer

from .generator import PaletteGenerator
from .discriminator import PaletteDiscriminator


HEX_RE = re.compile(r"#?(?:[0-9a-fA-F]{6})")  # accept with/without '#'


def _norm_col(name: str) -> str:
    # normalize column name: lowercase, remove non-alphanum
    return re.sub(r'[^a-z0-9]+', '', str(name).lower())


def _resolve_csv_paths(csv_path: str) -> List[str]:
    """Return a list of CSV paths. If a directory is provided, collect all *.csv inside."""
    if os.path.isdir(csv_path):
        return [
            os.path.join(csv_path, f)
            for f in os.listdir(csv_path)
            if f.lower().endswith(".csv")
        ]
    return [csv_path]


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = str(hex_color).strip().lstrip("#")
    if len(hex_color) != 6:
        # default black if malformed
        return (0, 0, 0)
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_unit(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    # map [0,255] -> [-1,1]
    return tuple((c / 255.0) * 2.0 - 1.0 for c in rgb)


def synth_caption_from_hexes(hexes: List[str]) -> str:
    # naive synthetic caption based on avg hue
    import colorsys
    rgbs = [hex_to_rgb(h) for h in hexes]
    hs = []
    for r, g, b in rgbs:
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        hs.append(h)
    if not hs:
        return "colorful palette"
    avg_h = sum(hs)/len(hs)
    if 0.0 <= avg_h < 0.08 or avg_h >= 0.92:
        mood = "warm red"
    elif avg_h < 0.25:
        mood = "sunny orange"
    elif avg_h < 0.42:
        mood = "earthy yellow-green"
    elif avg_h < 0.58:
        mood = "fresh green"
    elif avg_h < 0.75:
        mood = "cool blue"
    else:
        mood = "vibrant purple"
    return f"{mood} palette"


def _palette_descriptors(hexes: List[str]) -> str:
    """
    Build richer descriptors from palette stats: warm/cool, vivid/muted, light/dark, varied/monotone.
    """
    import colorsys
    hs, ss, ls = [], [], []
    for h in hexes:
        r, g, b = hex_to_rgb(h)
        hh, ll, ssat = colorsys.rgb_to_hls(r/255, g/255, b/255)
        hs.append(hh * 360.0); ss.append(ssat); ls.append(ll)
    if not hs:
        return "neutral muted mid palette"

    h_mean = sum(hs) / len(hs)
    s_mean = sum(ss) / len(ss)
    l_mean = sum(ls) / len(ls)
    # circular hue variance approx
    import numpy as np
    hs_rad = [math.radians(h) for h in hs]
    cx, sx = float(np.mean([math.cos(a) for a in hs_rad])), float(np.mean([math.sin(a) for a in hs_rad]))
    h_var = 1.0 - math.hypot(cx, sx)  # 0..1

    temp = "warm" if (h_mean < 60 or h_mean >= 300) else ("cool" if 180 <= h_mean < 300 else "neutral")
    vib = "muted" if s_mean < 0.35 else ("vivid" if s_mean > 0.6 else "soft")
    brt = "dark" if l_mean < 0.35 else ("light" if l_mean > 0.65 else "mid")
    var = "varied" if h_var > 0.35 else "monotone"

    return f"{temp} {vib} {brt} {var} palette"

def _caption_variants(hexes: List[str], names: List[str]) -> List[str]:
    """
    Build multiple caption variants from color names and descriptors.
    Returns 3-6 short captions to train stronger text-color alignment.
    """
    desc = _palette_descriptors(hexes)
    names_clean = [n for n in names if isinstance(n, str) and n.strip()]
    names_clean = [n.strip() for n in names_clean][:4]
    joined2 = ", ".join(names_clean[:2]) if len(names_clean) >= 2 else (names_clean[0] if names_clean else "")
    joined3 = ", ".join(names_clean[:3]) if len(names_clean) >= 3 else joined2

    base = [
        desc,
        f"{desc}, {joined2}" if joined2 else desc,
        f"{desc}, {joined3}" if joined3 else desc,
    ]

    # style/mood templates
    moods = [
        "vintage", "retro", "modern", "minimal", "pastel",
        "earthy", "bold", "soft", "muted", "vibrant"
    ]
    # pick a couple deterministic by hash to keep stable across runs
    seed = sum(int(h[1:3], 16) for h in hexes if isinstance(h, str) and len(h) >= 3)
    rng = random.Random(seed)
    m1 = rng.choice(moods)
    m2 = rng.choice([m for m in moods if m != m1])

    extras = [
        f"{m1} {desc}",
        f"{m2} {desc}",
        f"{m1} {m2} {desc}",
    ]
    # include a plain names caption if available
    if joined3:
        extras.append(f"{joined3} palette")

    # deduplicate while preserving order
    seen, out = set(), []
    for s in base + extras:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

class PaletteTextDataset(Dataset):
    """
    Loads palettes and captions from a CSV file.
    - If caption column missing, generates synthetic caption.
    - Accepts:
      • 'hex1'..'hex7' (any subset), or a single 'palette' column with comma-separated hex, OR
      • Single-color rows with columns 'RED','GREEN','BLUE' and/or 'HEX' (will be chunked into palettes).
    """
    def __init__(self, csv_path: str, max_colors: int = 7):
        super().__init__()
        self.max_colors = max_colors
        # store multiple caption variants per item
        self.items: List[Tuple[List[str], List[str]]] = []

        df = pd.read_csv(csv_path)
        cols = list(df.columns)
        norm_map = {_norm_col(c): c for c in cols}
        cols_norm = set(norm_map.keys())

        # Fast path: existing palette formats (hex1..hex7 or 'palette' column)
        has_caption_col = 'caption' in cols_norm
        hexN_cols = sorted(
            [norm_map[c] for c in cols_norm if c.startswith('hex') and c != 'hex'],
            key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 999
        )
        palette_col = norm_map.get('palette')

        if hexN_cols or palette_col:
            for _, row in df.iterrows():
                hexes: List[str] = []
                if hexN_cols:
                    for c in hexN_cols:
                        val = str(row.get(c, '')).strip()
                        found = HEX_RE.findall(val)
                        if found:
                            h = found[0]
                            if not h.startswith('#'):
                                h = f"#{h}"
                            hexes.append(h.upper())
                        elif val and val.lower() != 'nan' and len(val) in (6, 7):
                            h = val if val.startswith('#') else f"#{val}"
                            if HEX_RE.fullmatch(h):
                                hexes.append(h.upper())
                else:
                    vals = str(row.get(palette_col, '')).strip()
                    found = HEX_RE.findall(vals)
                    hexes = []
                    for h in found:
                        if not h.startswith('#'):
                            h = f"#{h}"
                        hexes.append(h.upper())
                    if not hexes and vals and vals.lower() != 'nan':
                        parts = [p.strip() for p in vals.split(",")]
                        hexes = []
                        for p in parts:
                            if HEX_RE.fullmatch(p):
                                h = p if p.startswith('#') else f"#{p}"
                                hexes.append(h.upper())

                hexes = hexes[:self.max_colors]
                if len(hexes) < self.max_colors:
                    hexes += ["#000000"] * (self.max_colors - len(hexes))
                # names column if available
                name_col = norm_map.get('colorname') or norm_map.get('name') or norm_map.get('color')
                names = [str(row.get(name_col, "")).strip()] if name_col else []
                # build variants
                caps = _caption_variants(hexes, names)
                self.items.append((hexes, caps))
            return

        # New path: single-color rows (e.g., catalog CSV with RED/GREEN/BLUE/HEX per row)
        has_rgb = all(k in cols_norm for k in ('red', 'green', 'blue'))
        has_hex_single = 'hex' in cols_norm

        if not (has_rgb or has_hex_single):
            # Fallback: scan row for HEX anywhere
            for _, row in df.iterrows():
                joined = " ".join(str(v) for v in row.values if pd.notna(v))
                found = HEX_RE.findall(joined)
                hexes = []
                for h in found[:self.max_colors]:
                    if not h.startswith('#'):
                        h = f"#{h}"
                    hexes.append(h.upper())
                if len(hexes) < self.max_colors:
                    hexes += ["#000000"] * (self.max_colors - len(hexes))
                caption = synth_caption_from_hexes(hexes)
                self.items.append((hexes, caption))
            return

        # Build a flat list of colors from rows
        hexes_all: List[str] = []
        names_all: List[str] = []
        name_col = norm_map.get('colorname') or norm_map.get('name') or norm_map.get('color')

        for _, row in df.iterrows():
            h = None
            if has_hex_single:
                raw = str(row.get(norm_map['hex'], '')).strip()
                if raw and raw.lower() != 'nan' and HEX_RE.fullmatch(raw):
                    h = raw if raw.startswith('#') else f"#{raw}"
            if h is None and has_rgb:
                try:
                    r = int(float(row.get(norm_map['red'], 0)))
                    g = int(float(row.get(norm_map['green'], 0)))
                    b = int(float(row.get(norm_map['blue'], 0)))
                    r = max(0, min(255, r)); g = max(0, min(255, g)); b = max(0, min(255, b))
                    h = f"#{r:02X}{g:02X}{b:02X}"
                except Exception:
                    h = None
            if h:
                hexes_all.append(h.upper())
                names_all.append(str(row.get(name_col, '')).strip() if name_col else '')

        # Chunk into palettes of max_colors; pad last chunk if needed
        for i in range(0, len(hexes_all), self.max_colors):
            chunk = hexes_all[i:i + self.max_colors]
            if not chunk:
                continue
            if len(chunk) < self.max_colors:
                chunk = chunk + ["#000000"] * (self.max_colors - len(chunk))
            subnames = [n for n in names_all[i:i + self.max_colors] if n]
            caps = _caption_variants(chunk, subnames)
            self.items.append((chunk, caps))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        hexes, captions = self.items[idx]
        rgbs = [hex_to_rgb(h) for h in hexes]
        arr = torch.tensor([rgb_to_unit(rgb) for rgb in rgbs], dtype=torch.float32)
        # randomly pick a variant each time
        cap = random.choice(captions)
        return arr, cap


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    csv_path: str = os.path.join("d:", os.sep, "CodeKaro", "Bit&Build_Hack", "data", "palette_data.csv"),
    save_dir: str = os.path.join("d:", os.sep, "CodeKaro", "Bit&Build_Hack", "models", "saved"),
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 2e-4,
    z_dim: int = 100,
    cond_dim: int = 384,
    max_colors: int = 7,
    sentence_model: str = "all-MiniLM-L6-v2",
    drop_cond_prob: float = 0.1,
    init_instance_noise: float = 0.05,
    shuffle_colors: bool = True,
    loss_type: str = "hinge",            # new: bce | hinge
    n_critic: int = 1,                   # new: D updates per G update
    r1_gamma: float = 0.0,               # new: R1 gradient penalty (0 disables)
    spectral_norm: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)
    device = get_device()

    paths = _resolve_csv_paths(csv_path)
    if len(paths) == 1:
        dataset = PaletteTextDataset(csv_path=paths[0], max_colors=max_colors)
    else:
        datasets = [PaletteTextDataset(csv_path=p, max_colors=max_colors) for p in paths]
        dataset = ConcatDataset(datasets)

    # use all data each epoch
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    encoder = SentenceTransformer(sentence_model, device=str(device))

    # ---- Build and save retrieval index (captions + embeddings + hex palettes) ----
    def _collect_items(ds):
        if isinstance(ds, ConcatDataset):
            items = []
            for sub in ds.datasets:
                items.extend(getattr(sub, "items", []))
            return items
        return getattr(ds, "items", [])

    items_all = _collect_items(dataset)  # [(hexes, [caps...]), ...]
    # flatten a few variants per item to enrich the index
    captions, hex_lists = [], []
    for hexes, caps in items_all:
        take = caps[:3] if isinstance(caps, list) else [caps]
        for c in take:
            captions.append(c)
            hex_lists.append(hexes)

    with torch.no_grad():
        cap_emb = encoder.encode(captions, convert_to_tensor=True, normalize_embeddings=True).cpu().float()
    torch.save(cap_emb, os.path.join(save_dir, "caption_embeddings.pt"))
    with open(os.path.join(save_dir, "caption_index.json"), "w", encoding="utf-8") as f:
        json.dump([{"caption": c, "hex": h} for c, h in zip(captions, hex_lists)], f, indent=2)
    print(f"[Index] Saved {len(captions)} caption embeddings to {save_dir}")

    G = PaletteGenerator(z_dim=z_dim, cond_dim=cond_dim, max_colors=max_colors).to(device)
    D = PaletteDiscriminator(cond_dim=cond_dim, max_colors=max_colors).to(device)

    # optional spectral norm
    if spectral_norm:
        def _apply_sn(m: nn.Module):
            for name, module in m.named_modules():
                if isinstance(module, nn.Linear):
                    nn.utils.spectral_norm(module, name='weight', n_power_iterations=1)
        _apply_sn(D)

    # opt betas tuned for hinge; keep BCE default if requested
    betas = (0.0, 0.9) if loss_type == "hinge" else (0.5, 0.999)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
    bce = nn.BCEWithLogitsLoss()

    def add_instance_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return x
        noise = torch.randn_like(x) * sigma
        return (x + noise).clamp(-1, 1)

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        running_g, running_d = 0.0, 0.0
        mean_real_score, mean_fake_score, n_batches = 0.0, 0.0, 0
        # anneal instance noise
        sigma = max(0.0, init_instance_noise * (1.0 - (epoch - 1) / max(1, epochs - 1)))

        for palettes, captions in loader:
            palettes = palettes.to(device)  # (B, max_colors, 3) in [-1,1]
            with torch.no_grad():
                cond = encoder.encode(list(captions), convert_to_tensor=True, normalize_embeddings=True).to(device)
                if cond.shape[-1] != cond_dim:
                    if cond.shape[-1] > cond_dim:
                        cond = cond[:, :cond_dim]
                    else:
                        pad = torch.zeros(cond.size(0), cond_dim - cond.size(1), device=device)
                        cond = torch.cat([cond, pad], dim=1)
            cond = cond.detach().clone().float()

            if drop_cond_prob > 0:
                mask = (torch.rand(cond.size(0), device=device) < drop_cond_prob).float().unsqueeze(1)
                cond = cond * (1.0 - mask)

            if shuffle_colors:
                idx = torch.stack([torch.randperm(palettes.size(1), device=device) for _ in range(palettes.size(0))])
                palettes = palettes[torch.arange(palettes.size(0)).unsqueeze(1), idx, :]

            # ---------------- Train D (n_critic steps) ----------------
            loss_D_total = 0.0
            real_score_batch, fake_score_batch = 0.0, 0.0
            for _ in range(max(1, n_critic)):
                # Optionally compute R1 on clean real
                real_in = palettes.detach()
                if r1_gamma > 0:
                    real_in.requires_grad_(True)
                    out_real = D(real_in, cond)
                else:
                    out_real = D(add_instance_noise(real_in, sigma), cond)

                z = torch.randn(palettes.size(0), z_dim, device=device)
                fake_palettes = G(z, cond).detach()
                out_fake = D(add_instance_noise(fake_palettes, sigma), cond)

                if loss_type == "hinge":
                    d_loss = F.relu(1.0 - out_real).mean() + F.relu(1.0 + out_fake).mean()
                else:  # bce
                    real_labels = torch.ones_like(out_real, device=device)
                    fake_labels = torch.zeros_like(out_fake, device=device)
                    d_loss = bce(out_real, real_labels) + bce(out_fake, fake_labels)

                if r1_gamma > 0:
                    grads = autograd.grad(outputs=out_real.sum(), inputs=real_in, create_graph=True, retain_graph=True, only_inputs=True)[0]
                    r1_penalty = grads.pow(2).reshape(grads.size(0), -1).sum(1).mean()
                    d_loss = d_loss + (r1_gamma * 0.5) * r1_penalty

                opt_D.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_D.step()
                loss_D_total += d_loss.item()

                real_score_batch += out_real.mean().item()
                fake_score_batch += out_fake.mean().item()

            # ---------------- Train G ----------------
            z = torch.randn(palettes.size(0), z_dim, device=device)
            gen_palettes = G(z, cond)
            if shuffle_colors:
                idx_g = torch.stack([torch.randperm(gen_palettes.size(1), device=device) for _ in range(gen_palettes.size(0))])
                gen_palettes = gen_palettes[torch.arange(gen_palettes.size(0)).unsqueeze(1), idx_g, :]

            out_fake_for_G = D(add_instance_noise(gen_palettes, sigma), cond)
            if loss_type == "hinge":
                g_loss = -out_fake_for_G.mean()
            else:
                target_for_G = torch.ones_like(out_fake_for_G, device=device)
                g_loss = bce(out_fake_for_G, target_for_G)

            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            running_g += g_loss.item()
            running_d += loss_D_total
            mean_real_score += real_score_batch / max(1, n_critic)
            mean_fake_score += fake_score_batch / max(1, n_critic)
            n_batches += 1

        avg_g = running_g / max(1, n_batches)
        avg_d = running_d / max(1, n_batches)
        avg_real = mean_real_score / max(1, n_batches)
        avg_fake = mean_fake_score / max(1, n_batches)
        print(f"Epoch {epoch}/{epochs} | G_loss: {avg_g:.4f} | D_loss: {avg_d:.4f} | D(real): {avg_real:.3f} | D(fake): {avg_fake:.3f}")

    # Save generator and config
    save_path = os.path.join(save_dir, "generator.pth")
    torch.save(G.state_dict(), save_path)
    config = {
        "z_dim": z_dim,
        "cond_dim": cond_dim,
        "max_colors": max_colors,
        "sentence_model": sentence_model,
        "index": {"embeddings": "caption_embeddings.pt", "meta": "caption_index.json"},
    }
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved generator to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cGAN for text-conditioned color palette generation.")
    parser.add_argument("--csv", dest="csv_path", type=str,
                        default=os.path.join("d:", os.sep, "CodeKaro", "Bit&Build_Hack", "data", "palette_data.csv"),
                        help="Path to a CSV file or a directory containing CSV files.")
    default_save_dir = os.path.join("d:", os.sep, "CodeKaro", "Bit&Build_Hack", "models", "saved")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=default_save_dir,
        nargs="?",
        const=default_save_dir,
        help="Directory to save checkpoints. If provided without a value, uses the default path."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument("--cond-dim", type=int, default=384)
    parser.add_argument("--max-colors", type=int, default=7)
    parser.add_argument("--sentence-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--dry-run", action="store_true", help="Preview parsed palettes and exit.")
    parser.add_argument("--auto-confirm", action="store_true", help="Skip interactive confirmation.")
    parser.add_argument("--drop-cond-prob", type=float, default=0.1)
    parser.add_argument(
        "--init-instance-noise",
        nargs="?",
        const=0.05,
        type=float,
        default=0.05,
        help="Initial instance noise sigma (e.g., 0.05). If provided without a value, defaults to 0.05. Use 0 to disable."
    )
    parser.add_argument("--no-shuffle-colors", action="store_true")
    # NEW: training stability/configuration args
    parser.add_argument("--loss-type", type=str, choices=["bce", "hinge"], default="hinge")
    parser.add_argument("--n-critic", type=int, default=1)
    parser.add_argument("--r1-gamma", type=float, default=0.0)
    parser.add_argument("--spectral-norm", action="store_true")
    args = parser.parse_args()

    # Preview and confirmation
    paths = _resolve_csv_paths(args.csv_path)
    if len(paths) == 1:
        preview_ds = PaletteTextDataset(paths[0], max_colors=args.max_colors)
    else:
        preview_ds = ConcatDataset([PaletteTextDataset(p, max_colors=args.max_colors) for p in paths])

    n_items = len(preview_ds)
    print(f"[Preview] Parsed palettes: {n_items} | max_colors={args.max_colors}")
    # show a couple of samples
    loader = DataLoader(preview_ds, batch_size=2, shuffle=False)
    try:
        batch, _caps = next(iter(loader))
        # map [-1,1] -> hex for preview
        def _to_hex(t):
            t = t.clamp(-1, 1)
            t = ((t + 1.0) / 2.0) * 255.0
            t = t.round().to(torch.int).tolist()
            return [f"#{r:02X}{g:02X}{b:02X}" for (r, g, b) in t]
        for i in range(min(2, batch.size(0))):
            print(f"  Sample {i+1}: {_to_hex(batch[i])}")
    except StopIteration:
        pass

    if args.dry_run:
        print("[Dry-Run] Exiting without training.")
        raise SystemExit(0)

    if not args.auto_confirm:
        ans = input("Proceed with training? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            raise SystemExit(0)

    train(
        csv_path=args.csv_path,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        z_dim=args.z_dim,
        cond_dim=args.cond_dim,
        max_colors=args.max_colors,
        sentence_model=args.sentence_model,
        drop_cond_prob=args.drop_cond_prob,
        init_instance_noise=args.init_instance_noise,
        shuffle_colors=not args.no_shuffle_colors,
        loss_type=args.loss_type,
        n_critic=args.n_critic,
        r1_gamma=args.r1_gamma,
        spectral_norm=args.spectral_norm,
    )
