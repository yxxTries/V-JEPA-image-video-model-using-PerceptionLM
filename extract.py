"""extract.py -- image or video -> V-JEPA embeddings. Prints tensor shapes.

  python extract.py path\\to\\clip.mp4
  python extract.py path\\to\\photo.jpg --model L --out feats.pt
  python extract.py clip.mp4 --hierarchical      # 4 intermediate normed layers (2.1 models)
"""
import argparse
import time

import torch

import jepa_common as jc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="image or video file")
    jc.add_common_args(p)
    p.add_argument("--hierarchical", action="store_true", help="also return the 4 hierarchical layers")
    p.add_argument("--out", default=None, help="save features to this .pt file")
    a = p.parse_args()

    device = jc.get_device()
    dtype = jc.resolve_dtype(a.dtype, device)
    encoder, name, spec = jc.load_encoder(a.model, device, dtype)

    x, disp, raw = jc.load_media(a.input, spec, a.frames, a.stride, a.res)
    print(f"[input] {a.input}: decoded {tuple(raw.shape)} uint8 (T,C,H,W) via "
          f"{'PIL' if jc.is_image(a.input) else jc.video_backend()} -> model input {tuple(x.shape)} (B,C,T,H,W)")
    x = x.to(device, dtype)
    t, h, w = jc.token_grid(x, family=spec["family"])

    if a.hierarchical and not hasattr(encoder, "hierarchical_layers"):
        raise SystemExit("--hierarchical needs a V-JEPA 2.1 model (B/L/g/G).")
    if a.hierarchical:
        encoder.out_layers = list(encoder.hierarchical_layers)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode(), jc.autocast(device, dtype):
        out = encoder(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    layers = out if isinstance(out, list) else [out]
    tokens = layers[-1].float()
    jc.check_finite(tokens, a.dtype)
    pooled = tokens.mean(1)

    print(f"[grid]   tokens = T'{t} x H'{h} x W'{w} = {t*h*w}")
    if a.hierarchical:
        for li, o in zip(encoder.hierarchical_layers, layers):
            print(f"[layer {li:>2}] {tuple(o.shape)}")
    print(f"[tokens] {tuple(tokens.shape)}  (B, N, D)")
    print(f"[grid]   {tuple(tokens.reshape(1, t, h, w, -1).shape)}  (B, T', H', W', D)")
    print(f"[pooled] {tuple(pooled.shape)}  mean-pooled clip embedding, L2 norm {pooled.norm():.2f}")
    print(f"[time]   {dt*1000:.0f} ms forward  {jc.gpu_mem_str(device)}")

    if a.out:
        payload = {"model": name, "input": a.input, "grid": (t, h, w), "tokens": tokens.cpu(), "pooled": pooled.cpu()}
        if a.hierarchical:
            payload["hierarchical"] = {li: o.float().cpu() for li, o in zip(encoder.hierarchical_layers, layers)}
        torch.save(payload, a.out)
        print(f"[saved]  {a.out}")


if __name__ == "__main__":
    main()
