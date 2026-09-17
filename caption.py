"""caption.py -- image or video -> caption: V-JEPA 2.1 ViT-L -> trained MLP projector -> Qwen2.5-1.5B-Instruct.

  python caption.py photo.jpg
  python caption.py clip.mp4                   # V-JEPA video tokens, averaged over time onto the 24x24 grid
  python caption.py clip.mp4 --video frames    # encode each sampled frame as an image instead, then average
  python caption.py photo.jpg --ckpt outputs\\projector_g12.pt --max-new-tokens 40

Uses the newest outputs\\projector_*.pt by default. Stage 1 trains on short BLIP captions of images only (no
instruction tuning), so expect one-line, alt-text style captions.
"""
import argparse
import time
from pathlib import Path

import torch

import jepa_common as jc
import vlm_common as vc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="image or video file")
    p.add_argument("--ckpt", type=Path, default=None, help=r"projector checkpoint (default: newest outputs\projector_*.pt)")
    p.add_argument("--dtype", default=jc.DEFAULT_DTYPE, choices=list(jc.DTYPES))
    p.add_argument("--frames", type=int, default=jc.DEFAULT_FRAMES, help="video frames to sample (default 16)")
    p.add_argument("--stride", type=int, default=None, help="take consecutive frames every N (default: span the clip)")
    p.add_argument("--video", default="clip", choices=["clip", "frames"],
                   help="clip: V-JEPA video tokens mean-pooled over time (default); frames: each frame as an image")
    p.add_argument("--max-new-tokens", type=int, default=60)
    a = p.parse_args()
    a.input = a.input.strip().strip('"')  # tolerate quotes / trailing spaces from drag-and-drop

    ckpt_path = a.ckpt or vc.latest_projector()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg, info = ckpt["config"], ckpt["train"]
    print(f"[ckpt]   {ckpt_path.name}: {cfg['encoder']} -> {cfg['grid']}x{cfg['grid']} tokens -> projector -> "
          f"{cfg['llm']} (trained on {info['samples']:,} samples, final loss {info['final_loss']:.2f})")

    device = jc.get_device()
    dtype = jc.resolve_dtype(a.dtype, device)
    encoder, name, spec = jc.load_encoder(cfg["encoder"], device, dtype)
    t0 = time.time()
    tok, llm = vc.load_llm(cfg["llm"], device, dtype)
    proj = vc.Projector(cfg["d_vision"], llm.config.hidden_size)
    proj.load_state_dict(ckpt["projector"])
    proj = proj.to(device, dtype).eval()
    print(f"[load]   {cfg['llm']} + projector, {str(dtype).replace('torch.', '')} on {device} ({time.time() - t0:.1f}s)")

    # 1) media -> V-JEPA input, exactly like extract.py
    x, _, raw = jc.load_media(a.input, spec, a.frames, a.stride)
    print(f"[input]  {a.input}: decoded {tuple(raw.shape)} uint8 (T,C,H,W) -> model input {tuple(x.shape)} (B,C,T,H,W)")
    frames_mode = a.video == "frames" and x.shape[2] > 1
    if frames_mode:
        if spec["family"] != "2.1":
            raise SystemExit("--video frames needs a V-JEPA 2.1 model (it has an image patch embedding)")
        x = x.transpose(1, 2).reshape(-1, 3, 1, *x.shape[-2:])  # [T, C, 1, H, W]: a batch of one-frame images
        print(f"[frames] {tuple(x.shape)} -- each frame goes through the image path, like the training data")
    x = x.to(device, dtype)
    t, h, w = jc.token_grid(x, family=spec["family"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode(), jc.autocast(device, dtype):
        tokens = encoder(x)  # 2) frozen V-JEPA: [B, t*h*w, D]
        if frames_mode:  # T images x 576 tokens -> one T x 24 x 24 grid
            t, tokens = x.shape[0], tokens.reshape(1, -1, tokens.shape[-1])
        print(f"[tokens] {tuple(tokens.shape)} (B, T'{t} x H'{h} x W'{w}, D)")
        pooled = vc.pool_tokens(tokens.float(), t, h, w, cfg["grid"])  # 3) mean over time, pool to training grid
        print(f"[pooled] {tuple(pooled.shape)} (B, {cfg['grid']}x{cfg['grid']}, D)")
        img = proj(pooled.to(dtype))  # 4) projector -> LLM embedding space, wrapped in the chat prompt
        pre, post = vc.prompt_ids(tok, cfg["prompt"])
        embeds = vc.build_inputs(llm, pre, post, img)
        print(f"[embeds] projector {tuple(img.shape)} (B, N, d_llm) -> inputs_embeds {tuple(embeds.shape)} = "
              f"{len(pre)} prompt + {img.shape[1]} image + {len(post)} prompt tokens")
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = vc.generate(llm, tok, embeds, a.max_new_tokens)  # 5) greedy decoding, new tokens only
    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    print(f"[gen]    {tuple(out.shape)} new token ids at {out.shape[1] / (t2 - t1):.0f} tok/s")
    print(f"[time]   {(t1 - t0) * 1000:.0f} ms encode+project, {(t2 - t1) * 1000:.0f} ms generate  {jc.gpu_mem_str(device)}")
    print(f"\ncaption: {tok.decode(out[0], skip_special_tokens=True).strip()}")


if __name__ == "__main__":
    main()
