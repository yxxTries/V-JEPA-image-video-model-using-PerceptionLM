"""hooks.py -- per-layer activations of the V-JEPA encoder via nnsight.

  python hooks.py clip.mp4                        # stats for every transformer block
  python hooks.py clip.mp4 --layers 0 11 23 --save outputs\\acts.pt
  python hooks.py clip.mp4 --ablate 12            # zero block 12's residual update, measure effect

Reads block outputs with `model.blocks[i].output[0].save()` inside `model.trace(x)`.
(V-JEPA 2.1 blocks return (tokens, attn); V-JEPA 2 blocks return tokens.)
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from nnsight import NNsight

import jepa_common as jc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="image or video file")
    jc.add_common_args(p)
    p.add_argument("--layers", nargs="+", type=int, default=None, help="block indices (default: all)")
    p.add_argument("--save", default=None, help="save activations dict to .pt (fp16, CPU)")
    p.add_argument("--ablate", type=int, default=None, help="skip block N (output := input) and compare final tokens")
    a = p.parse_args()

    device = jc.get_device()
    dtype = jc.resolve_dtype(a.dtype, device)
    encoder, name, spec = jc.load_encoder(a.model, device, dtype)
    x, _, _ = jc.load_media(a.input, spec, a.frames, a.stride, a.res)
    x = x.to(device, dtype)
    t, h, w = jc.token_grid(x, family=spec["family"])

    model = NNsight(encoder)
    tuple_out = spec["family"] == "2.1"   # 2.1 blocks return (tokens, attn); V-JEPA 2 blocks return tokens
    n_blocks = len(encoder.blocks)
    layers = a.layers if a.layers is not None else list(range(n_blocks))
    for li in layers:
        if not 0 <= li < n_blocks:
            raise SystemExit(f"layer {li} out of range (model has {n_blocks} blocks)")

    # --- 1) record ---------------------------------------------------------------------------------
    saved = {}
    with torch.inference_mode(), jc.autocast(device, dtype):
        with model.trace(x):
            emb = model.patch_embed_img.output.save() if (t == 1 and spec["family"] == "2.1" and x.shape[2] == 1) \
                else model.patch_embed.output.save()
            for li in range(n_blocks):          # modules must be touched in execution order
                if li in layers:
                    out = model.blocks[li].output
                    saved[li] = (out[0] if tuple_out else out).save()
            final = model.output.save()

    final = final.float()
    print(f"[grid] T'{t} x H'{h} x W'{w} = {t*h*w} tokens ; patch_embed {tuple(emb.shape)} ; final {tuple(final.shape)}")
    print(f"\n{'layer':>5} {'shape':>18} {'mean|x|':>9} {'std':>8} {'max|x|':>9} {'cos(prev)':>10} {'cos(final)':>11} {'eff.rank':>9}")
    prev = emb.float()
    stats = {}
    for li in layers:
        a_ = saved[li].float()
        norm = a_.norm(dim=-1).mean().item()
        cos_prev = F.cosine_similarity(a_, prev, dim=-1).mean().item() if prev.shape == a_.shape else float("nan")
        cos_fin = F.cosine_similarity(a_, final, dim=-1).mean().item()
        # effective rank (participation ratio) of token covariance -- collapse indicator
        z = a_[0] - a_[0].mean(0)
        sv = torch.linalg.svdvals(z[: min(4096, z.shape[0])])
        erank = (sv.sum() ** 2 / (sv ** 2).sum()).item()
        stats[li] = dict(mean_norm=norm, std=a_.std().item(), max_abs=a_.abs().max().item(),
                         cos_prev=cos_prev, cos_final=cos_fin, eff_rank=erank)
        print(f"{li:>5} {str(tuple(a_.shape)):>18} {norm:9.2f} {stats[li]['std']:8.3f} {stats[li]['max_abs']:9.1f} "
              f"{cos_prev:10.3f} {cos_fin:11.3f} {erank:9.1f}")
        prev = a_
    if device.type == "cuda":
        print(f"\n[mem] {jc.gpu_mem_str(device)}")

    # --- 2) optional intervention --------------------------------------------------------------------
    if a.ablate is not None:
        k = a.ablate
        with torch.inference_mode(), jc.autocast(device, dtype):
            with model.trace(x):
                inp = model.blocks[k].inputs[0][0]          # residual stream entering block k
                out = model.blocks[k].output
                model.blocks[k].output = (inp, out[1]) if tuple_out else inp   # skip the block
                ablated = model.output.save()
        cos = F.cosine_similarity(ablated.float(), final, dim=-1)
        print(f"\n[ablate] block {k} skipped -> final tokens cos vs clean: mean {cos.mean():.4f}, min {cos.min():.4f}; "
              f"pooled cos {F.cosine_similarity(ablated.float().mean(1), final.mean(1)).item():.4f}")

    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": name, "input": a.input, "grid": (t, h, w), "stats": stats,
                    "activations": {li: saved[li].half().cpu() for li in layers}}, a.save)
        print(f"[saved] {a.save}")


if __name__ == "__main__":
    main()
