"""similarity.py -- cosine similarity between every clip (or image) in a folder.

  python similarity.py path\\to\\clips
  python similarity.py clips --pool max --outdir outputs

Each clip -> mean-pooled encoder tokens -> L2-normalised vector. Prints the matrix, each clip's
nearest neighbour, and saves outputs\\similarity.csv + similarity.png (heatmap).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import jepa_common as jc


def heatmap(sim: np.ndarray, names: list[str], path: Path, cell: int = 48):
    n = len(names)
    label_w = min(260, 8 * max(len(s) for s in names) + 10)
    img = Image.new("RGB", (label_w + n * cell, label_w + n * cell), "white")
    d = ImageDraw.Draw(img)
    lo, hi = float(sim.min()), float(sim.max())
    for i in range(n):
        d.text((4, label_w + i * cell + cell // 2 - 6), names[i][:32], fill="black")
        col = Image.new("RGBA", (label_w, 14), (255, 255, 255, 0))
        ImageDraw.Draw(col).text((2, 0), names[i][:32], fill="black")
        img.paste(col.rotate(90, expand=True), (label_w + i * cell + cell // 2 - 7, 0), col.rotate(90, expand=True))
        for j in range(n):
            v = (sim[i, j] - lo) / (hi - lo + 1e-9)
            c = (int(255 * v), int(80 + 100 * (1 - abs(v - .5) * 2)), int(255 * (1 - v)))
            x0, y0 = label_w + j * cell, label_w + i * cell
            d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=c)
            d.text((x0 + 8, y0 + cell // 2 - 6), f"{sim[i, j]:.2f}", fill="white")
    img.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="folder of videos and/or images")
    jc.add_common_args(p)
    p.add_argument("--pool", default="mean", choices=["mean", "max"])
    p.add_argument("--outdir", default="outputs")
    a = p.parse_args()

    files = jc.list_media(a.folder)
    if len(files) < 2:
        raise SystemExit(f"Need at least 2 videos/images in {a.folder}, found {len(files)}.")

    device = jc.get_device()
    dtype = jc.resolve_dtype(a.dtype, device)
    encoder, name, spec = jc.load_encoder(a.model, device, dtype)

    embs, names = [], []
    for f in files:
        try:
            x, _, raw = jc.load_media(f, spec, a.frames, a.stride, a.res)
        except Exception as e:
            print(f"[skip] {f.name}: {type(e).__name__}: {e}")
            continue
        with torch.inference_mode(), jc.autocast(device, dtype):
            tok = encoder(x.to(device, dtype)).float()
        jc.check_finite(tok, a.dtype)
        v = tok.mean(1) if a.pool == "mean" else tok.amax(1)
        embs.append(F.normalize(v, dim=-1)[0].cpu())
        names.append(f.name)
        print(f"[embed] {f.name:<40} {tuple(raw.shape)} -> {tuple(tok.shape)} -> {tuple(v.shape)}")
    if len(embs) < 2:
        raise SystemExit("Fewer than 2 files decoded.")

    E = torch.stack(embs)
    sim = (E @ E.T).numpy()
    n = len(names)
    short = [s[:14] for s in names]
    print("\nCosine similarity:")
    print(" " * 16 + "".join(f"{s:>16}" for s in short))
    for i in range(n):
        print(f"{short[i]:<16}" + "".join(f"{sim[i, j]:>16.3f}" for j in range(n)))

    print("\nNearest neighbour:")
    masked = sim.copy()
    np.fill_diagonal(masked, -np.inf)
    for i in range(n):
        j = int(masked[i].argmax())
        print(f"  {names[i]:<40} -> {names[j]:<40} {sim[i, j]:.3f}")
    off = sim[~np.eye(n, dtype=bool)]
    print(f"\nOff-diagonal: mean {off.mean():.3f}, min {off.min():.3f}, max {off.max():.3f}  "
          f"(pooled JEPA features are anisotropic -- compare values relatively, not to 0)")

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "similarity.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow([""] + names)
        for i in range(n):
            wr.writerow([names[i]] + [f"{v:.4f}" for v in sim[i]])
    heatmap(sim, names, outdir / "similarity.png")
    torch.save({"model": name, "names": names, "embeddings": E, "similarity": torch.from_numpy(sim)},
               outdir / "similarity_embeddings.pt")
    print(f"[saved] {outdir / 'similarity.csv'}, {outdir / 'similarity.png'}, {outdir / 'similarity_embeddings.pt'}")


if __name__ == "__main__":
    main()
