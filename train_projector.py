"""train_projector.py -- LLaVA / PerceptionLM stage 1: train only the MLP projector on cached V-JEPA features.

  python train_projector.py                         # 1 epoch over data\\llava20k: fp16, micro-batch 4 x accum 8 = 32
  python train_projector.py --micro-batch 16 --accum 2 --grad-ckpt   # less VRAM, ~30% slower
  python train_projector.py --grid 24 --grad-ckpt --micro-batch 8 --accum 4   # all 576 tokens (llava_prep.py --grid 24)
  python train_projector.py --limit 400 --max-steps 5   # smoke test

RTX 5060 Laptop (8 GB): micro-batch 4 peaks at 5.1 GiB allocated / 5.4 GiB reserved, 12 samples/s, 20k samples in
~28 min. Micro-batch 8 (6.4 GiB) overflows once the desktop's own ~1.5 GiB is counted, and Windows then pages VRAM to
system RAM without an error (training runs ~8x slower): watch the [vram] line.

Frozen: V-JEPA 2.1 ViT-L (not even loaded -- llava_prep.py cached its features) and Qwen2.5-1.5B-Instruct (fp16).
Trained: Linear(1024->1536) -> GELU -> Linear(1536->1536). AdamW lr 1e-3, no weight decay, 3% warmup then cosine
decay (LLaVA's stage-1 schedule), grad clip 1.0, fp16 autocast + GradScaler.
Loss: next-token cross-entropy on the caption tokens (+ the closing <|im_end|>, so generation knows when to stop).
Writes outputs\\projector_g<grid>.pt (projector weights + config) and outputs\\projector_g<grid>_loss.csv.
"""
import argparse
import csv
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch

import jepa_common as jc
import vlm_common as vc

warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")  # GradScaler may skip the first fp16 steps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=vc.DATA, help="folder written by llava_prep.py")
    p.add_argument("--grid", type=int, default=12, help="which feature cache (llava_prep.py --grid), default 12")
    p.add_argument("--model", default="L", help="V-JEPA model the cache was made with (default L)")
    p.add_argument("--llm", default=vc.LLM_ID, help=f"frozen LLM (default {vc.LLM_ID})")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"], help="LLM weights + autocast dtype")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--micro-batch", type=int, default=4, help="samples per forward/backward -- the VRAM knob (default 4)")
    p.add_argument("--accum", type=int, default=8, help="micro-batches per optimizer step (default 8 -> batch 32)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--grad-ckpt", action="store_true", help="recompute LLM activations in backward (less VRAM, ~30%% slower)")
    p.add_argument("--limit", type=int, default=None, help="train on the first N samples only")
    p.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps")
    p.add_argument("--log-every", type=int, default=25)
    a = p.parse_args()

    device = jc.get_device()
    dtype = jc.resolve_dtype(a.dtype, device)
    torch.manual_seed(0)

    # 1) cached V-JEPA features: float16 memmap, only each batch's rows are read from disk
    name, _ = jc.resolve_model(a.model)
    npy = a.data / f"feats_{name}_g{a.grid}.npy"
    meta = json.loads(npy.with_suffix(".json").read_text()) if npy.with_suffix(".json").exists() else {}
    if not npy.exists() or meta.get("done") != meta.get("shape", [0])[0]:
        raise SystemExit(f"{npy} missing or incomplete -- run llava_prep.py --grid {a.grid} first.")
    feats = np.load(npy, mmap_mode="r")
    samples = json.loads((a.data / "samples.json").read_text(encoding="utf-8"))
    N = min(len(samples), a.limit or len(samples))
    print(f"[data]   {npy.name}: {feats.shape} float16 (N, tokens, D) memmap; training on {N:,} samples")

    # 2) frozen LLM + token ids of the fixed prompt and of every caption
    t0 = time.time()
    tok, llm = vc.load_llm(a.llm, device, dtype)
    pre, post = vc.prompt_ids(tok)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    caps = [tok(s["caption"], add_special_tokens=False).input_ids[:63] + [im_end] for s in samples[:N]]
    width = -(-max(map(len, caps)) // 8) * 8  # one caption width for every batch -> identical tensor shapes each step
    print(f"[llm]    {a.llm}: {sum(q.numel() for q in llm.parameters()) / 1e9:.2f}B params frozen, hidden "
          f"{llm.config.hidden_size}, {str(dtype).replace('torch.', '')} on {device} ({time.time() - t0:.0f}s)")
    print(f"[seq]    {len(pre)} prompt + {feats.shape[1]} image + {len(post)} prompt tokens + caption "
          f"(mean {np.mean([len(c) for c in caps]):.1f}, max {max(map(len, caps))} incl. <|im_end|>, padded to {width})")
    if a.grad_ckpt:
        llm.gradient_checkpointing_enable()
        llm.train()  # HF only checkpoints in train mode; Qwen2 has no dropout so outputs are identical

    # 3) the only trainable module: fp32 master weights, forward runs in fp16 under autocast
    proj = vc.Projector(feats.shape[2], llm.config.hidden_size).to(device)
    opt = torch.optim.AdamW(proj.parameters(), lr=a.lr, weight_decay=0.0)
    per_epoch = N // (a.micro_batch * a.accum)
    steps = min(a.epochs * per_epoch, a.max_steps or 10**9)
    warm = max(1, round(0.03 * steps))

    def lr_factor(s):  # linear warmup, then cosine to 0
        return (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    amp = dict(device_type=device.type, dtype=dtype, enabled=device.type == "cuda")
    print(f"[proj]   {sum(q.numel() for q in proj.parameters()) / 1e6:.2f}M trainable params | {steps} steps x batch "
          f"{a.micro_batch}x{a.accum}={a.micro_batch * a.accum} | lr {a.lr:g}, {warm} warmup steps")

    gen = torch.Generator().manual_seed(0)
    log, window, step = [], [], 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for epoch in range(a.epochs):
        order = torch.randperm(N, generator=gen).tolist()
        for s in range(per_epoch):
            if step >= steps:
                break
            step_loss = 0.0
            for m in range(a.accum):
                k = (s * a.accum + m) * a.micro_batch
                idx = sorted(order[k:k + a.micro_batch])  # sorted -> sequential reads from the memmap
                x = torch.from_numpy(np.asarray(feats[idx])).to(device).float()  # [b, tokens, 1024]
                tgt = torch.full((len(idx), width), vc.IGNORE, dtype=torch.long)  # [b, width] caption ids, IGNORE-padded
                for r, i in enumerate(idx):
                    tgt[r, :len(caps[i])] = torch.tensor(caps[i])
                with torch.autocast(**amp):
                    img = proj(x)  # [b, tokens, 1536]
                    tgt = tgt.to(device)
                    embeds = vc.build_inputs(llm, pre, post, img, tgt)  # [b, prompt + tokens + prompt + width, 1536]
                    loss = vc.caption_loss(llm, embeds, tgt)
                scaler.scale(loss / a.accum).backward()
                step_loss += loss.item() / a.accum
                if step == 0 and m == 0:
                    print(f"[shapes] feats {tuple(x.shape)} -> projector {tuple(img.shape)} -> inputs_embeds "
                          f"{tuple(embeds.shape)}, targets {tuple(tgt.shape)} ({int((tgt != vc.IGNORE).sum())} caption "
                          f"tokens) -> loss {loss.item():.3f}")
                    emb_rms = llm.get_input_embeddings().weight[::64].float().pow(2).mean().sqrt()  # row sample; a full fp32 copy would pin 1.8 GiB of VRAM
                    print(f"[scale]  rms: image tokens {img.float().pow(2).mean().sqrt():.3f} vs text embeddings {emb_rms:.3f}")
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0).item()
            lr = opt.param_groups[0]["lr"]
            scaler.step(opt)  # skipped automatically if fp16 grads overflowed
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            log.append((step, round(step_loss, 4), lr, round(gnorm, 4)))
            window.append(step_loss)
            if not math.isfinite(step_loss):
                print(f"[warn]   non-finite loss at step {step}: fp16 overflow inside the LLM? Try --dtype bf16.")
            if step == 1 or step % a.log_every == 0 or step == steps:
                el = time.time() - t0
                print(f"[step {step:>4}/{steps}] loss {np.mean(window):.3f}  lr {lr:.1e}  |grad| {gnorm:.2f}  "
                      f"{step * a.micro_batch * a.accum / el:.1f} samples/s  {jc.gpu_mem_str(device)}  "
                      f"{el / 60:.1f} min, ETA {(steps - step) * el / step / 60:.1f} min")
                window = []
            if step == 1 and device.type == "cuda":
                free, total = torch.cuda.mem_get_info()
                print(f"[vram]   {free / 2**30:.2f} GiB of {total / 2**30:.2f} GiB still free on the GPU (Windows silently "
                      f"spills into slow shared RAM when this hits 0 -- lower --micro-batch then)")

    el = time.time() - t0
    final = float(np.mean([r[1] for r in log[-50:]]))
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    out = vc.OUT / f"projector_g{a.grid}.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({"projector": proj.state_dict(),
                "config": {"encoder": name, "grid": a.grid, "d_vision": feats.shape[2], "llm": a.llm, "prompt": vc.PROMPT},
                "train": {"samples": step * a.micro_batch * a.accum, "steps": step, "batch": a.micro_batch * a.accum,
                          "lr": a.lr, "dtype": a.dtype, "first_loss": log[0][1], "final_loss": final,
                          "minutes": el / 60, "peak_gib": peak}}, out)
    with open(out.with_name(out.stem + "_loss.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("step", "loss", "lr", "grad_norm"), *log])
    print(f"[done]   {step} steps, {step * a.micro_batch * a.accum:,} samples in {el / 60:.1f} min | loss "
          f"{log[0][1]:.3f} -> {final:.3f} (mean of last 50 steps) | {jc.gpu_mem_str(device)}")
    print(f"[saved]  {out} ({out.stat().st_size / 2**20:.1f} MiB) + {out.stem}_loss.csv")

    # 4) sanity check: greedy captions for three training images (caption.py does unseen images/videos)
    llm.eval()
    for i in order[:3]:
        with torch.inference_mode(), torch.autocast(**amp):
            embeds = vc.build_inputs(llm, pre, post, proj(torch.from_numpy(np.array(feats[i:i + 1])).to(device).float()))
            ids = vc.generate(llm, tok, embeds, 40)
        print(f"[sample] {samples[i]['image']}\n           BLIP: {samples[i]['caption']}\n           ours: "
              f"{tok.decode(ids[0], skip_special_tokens=True).strip()}")


if __name__ == "__main__":
    main()
