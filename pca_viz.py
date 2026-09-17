"""pca_viz.py -- PCA of V-JEPA patch features -> RGB map per frame (the V-JEPA 2.1 paper visual).

  python pca_viz.py clip.mp4                   # writes outputs\\<name>_pca.png (+ .gif)
  python pca_viz.py photo.jpg
  python pca_viz.py clip.mp4 --per-frame-pca   # fit PCA per time step instead of jointly

PCA is fit jointly over all tokens of the clip, so colours are comparable across frames: a
temporally consistent model keeps the same object the same colour.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import jepa_common as jc


def pca_rgb(feats: torch.Tensor, k: int = 3) -> torch.Tensor:
    """feats [..., D] -> [..., 3] in [0,1]."""
    shape = feats.shape[:-1]
    f = feats.reshape(-1, feats.shape[-1]).float()
    f = f - f.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(f, q=k, center=False, niter=6)
    proj = f @ V[:, :k]
    lo, hi = proj.quantile(0.01, dim=0), proj.quantile(0.99, dim=0)   # robust min/max
    proj = ((proj - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    return proj.reshape(*shape, k)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="image or video file")
    jc.add_common_args(p)
    p.add_argument("--per-frame-pca", action="store_true")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args()

    device = jc.get_device()
    dtype = jc.resolve_dtype(a.dtype, device)
    encoder, name, spec = jc.load_encoder(a.model, device, dtype)
    x, disp, _ = jc.load_media(a.input, spec, a.frames, a.stride, a.res)
    t, h, w = jc.token_grid(x, family=spec["family"])

    with torch.inference_mode(), jc.autocast(device, dtype):
        tokens = encoder(x.to(device, dtype)).float()
    jc.check_finite(tokens, a.dtype)
    grid = tokens.reshape(t, h, w, -1)
    print(f"[features] tokens {tuple(tokens.shape)} -> grid {tuple(grid.shape)} (T',H',W',D)")

    rgb = torch.stack([pca_rgb(g) for g in grid]) if a.per_frame_pca else pca_rgb(grid)  # [T',h,w,3]
    res = disp.shape[-1]
    maps = F.interpolate(rgb.permute(0, 3, 1, 2), size=(res, res), mode="nearest")  # [T',3,res,res]

    # one display frame per temporal token (first frame of each tubelet)
    step = max(1, disp.shape[0] // t)
    frames = disp[::step][:t].cpu()
    if frames.shape[0] < t:
        frames = torch.cat([frames, frames[-1:].repeat(t - frames.shape[0], 1, 1, 1)])

    to_np = lambda im: (im.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    top = np.concatenate([to_np(f) for f in frames], 1)
    bot = np.concatenate([to_np(m) for m in maps.cpu()], 1)
    sheet = Image.fromarray(np.concatenate([top, bot], 0))
    cols = min(t, 8)  # wrap long clips into rows of 8
    if t > cols:
        rows = []
        for r in range(0, t, cols):
            seg_t = np.concatenate([to_np(f) for f in frames[r:r+cols]], 1)
            seg_b = np.concatenate([to_np(m) for m in maps[r:r+cols].cpu()], 1)
            pad = cols * res - seg_t.shape[1]
            if pad:
                seg_t = np.pad(seg_t, ((0, 0), (0, pad), (0, 0)))
                seg_b = np.pad(seg_b, ((0, 0), (0, pad), (0, 0)))
            rows += [seg_t, seg_b]
        sheet = Image.fromarray(np.concatenate(rows, 0))

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(a.input).stem + f"_{name}_pca"
    png = outdir / f"{stem}.png"
    sheet.save(png)
    print(f"[saved] {png}  ({sheet.width}x{sheet.height}; top: frames, bottom: PCA RGB)")
    if t > 1 and not a.no_gif:
        gif = outdir / f"{stem}.gif"
        pairs = [Image.fromarray(np.concatenate([to_np(f), to_np(m)], 1)) for f, m in zip(frames, maps.cpu())]
        pairs[0].save(gif, save_all=True, append_images=pairs[1:], duration=250, loop=0)
        print(f"[saved] {gif}  (frame | PCA side by side, {t} steps)")


if __name__ == "__main__":
    main()
