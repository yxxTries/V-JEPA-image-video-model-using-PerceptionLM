"""classify.py -- Something-Something-v2 action recognition -> top-5 labels.

  python classify.py path\\to\\clip.mp4
  python classify.py clip.mp4 --stride 4

Checkpoint: facebook/vjepa2-vitl-fpc16-256-ssv2 (Hugging Face, downloads on first run, ~1.3 GB).
It is a V-JEPA 2 ViT-L backbone + attentive probe trained on SSv2 with 16-frame clips @256.
Meta has not released SSv2 heads for V-JEPA 2.1 yet, so this is the newest available classifier.
SSv2 labels are hand-object interactions ("Pushing something from left to right"), not sports/scenes.
"""
import argparse
import time

import torch

import jepa_common as jc

CKPT = "facebook/vjepa2-vitl-fpc16-256-ssv2"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="video file")
    p.add_argument("--checkpoint", default=CKPT, help="HF id (e.g. facebook/vjepa2-vitg-fpc64-384-ssv2 for ViT-g, 64 frames)")
    p.add_argument("--dtype", default=jc.DEFAULT_DTYPE, choices=list(jc.DTYPES))
    p.add_argument("--frames", type=int, default=None, help="default: checkpoint's frames_per_clip (16)")
    p.add_argument("--stride", type=int, default=None, help="consecutive frames every N (default: span whole clip)")
    p.add_argument("--topk", type=int, default=5)
    a = p.parse_args()

    from transformers import AutoModelForVideoClassification, AutoVideoProcessor

    device = jc.get_device()
    dtype = jc.resolve_dtype(a.dtype, device)
    t0 = time.time()
    processor = AutoVideoProcessor.from_pretrained(a.checkpoint)
    model = AutoModelForVideoClassification.from_pretrained(a.checkpoint, dtype=dtype, attn_implementation="sdpa")
    model = model.to(device).eval()
    print(f"[load] {a.checkpoint}: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, "
          f"{len(model.config.id2label)} classes, {str(dtype).replace('torch.', '')} on {device} ({time.time()-t0:.1f}s)")

    n = a.frames or model.config.frames_per_clip
    video = jc.read_video(a.input, n, a.stride)  # uint8 [T,C,H,W]
    print(f"[input] {a.input}: {tuple(video.shape)} via {jc.video_backend()}")
    inputs = processor(video, return_tensors="pt")
    pix = inputs["pixel_values_videos"].to(device, dtype)
    print(f"[input] pixel_values_videos {tuple(pix.shape)} (B,T,C,H,W)")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), jc.autocast(device, dtype):
        logits = model(pixel_values_videos=pix).logits.float()
    jc.check_finite(logits, a.dtype)
    probs = logits.softmax(-1)[0]
    top = probs.topk(a.topk)

    print(f"\nTop-{a.topk} SSv2 actions:")
    for rank, (pr, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        print(f"  {rank}. {pr*100:5.1f}%  {model.config.id2label[idx]}")
    if device.type == "cuda":
        print(f"\n[mem] {jc.gpu_mem_str(device)}")


if __name__ == "__main__":
    main()
