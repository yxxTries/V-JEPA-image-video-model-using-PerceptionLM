"""bench.py -- peak VRAM and latency by model size x frame count (synthetic input, encoder only).

  python bench.py                                  # sizes B,L ; frames 8,16,32 ; fp16 @ native res
  python bench.py --sizes B L g --frames 16 32 64
  python bench.py --sizes G --random-weights       # skip the multi-GB download; timing is identical

Out-of-memory configs are recorded as OOM and the sweep continues. Results -> outputs\\bench.csv.
"""
import argparse
import csv
import gc
import statistics
import time
from pathlib import Path

import torch

import jepa_common as jc


def run_one(encoder, spec, frames, res, device, dtype, warmup, iters, batch):
    x = torch.randn(batch, 3, frames, res, res, device=device, dtype=dtype)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.inference_mode(), jc.autocast(device, dtype):
        for _ in range(warmup):
            encoder(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            encoder(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    t, h, w = jc.token_grid(x, family=spec["family"])
    return dict(tokens=t * h * w,
                ms_median=statistics.median(times) * 1000,
                ms_min=min(times) * 1000,
                clips_per_s=batch / statistics.median(times),
                peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                activ_gib=(torch.cuda.max_memory_allocated() - base) / 2**30)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", nargs="+", default=["B", "L"], help="aliases or full names, e.g. B L g G 2-L")
    p.add_argument("--frames", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--res", type=int, default=None, help="default: native (384 for 2.1)")
    p.add_argument("--dtype", default=jc.DEFAULT_DTYPE, choices=list(jc.DTYPES))
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--random-weights", action="store_true", help="don't download checkpoints (same speed/VRAM)")
    p.add_argument("--compile", action="store_true", help="torch.compile the encoder (needs Triton; see README)")
    p.add_argument("--out", default="outputs/bench.csv")
    a = p.parse_args()

    device = jc.get_device(require_cuda=True)
    dtype = jc.resolve_dtype(a.dtype, device)
    props = torch.cuda.get_device_properties(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"[gpu] {props.name}  {props.total_memory/2**30:.1f} GiB  capability {props.major}.{props.minor}  "
          f"torch {torch.__version__} CUDA {torch.version.cuda}  dtype {a.dtype}")

    rows = []
    for size in a.sizes:
        name, spec = jc.resolve_model(size)
        res = a.res or spec["res"]
        try:
            torch.cuda.reset_peak_memory_stats()
            encoder, name, spec = jc.load_encoder(name, device, dtype, pretrained=not a.random_weights)
        except torch.OutOfMemoryError:
            print(f"[OOM] {name}: weights do not fit")
            rows.append(dict(model=name, frames="-", res=res, status="OOM(weights)"))
            gc.collect(); torch.cuda.empty_cache()
            continue
        weights_gib = torch.cuda.memory_allocated() / 2**30
        if a.compile:
            encoder = torch.compile(encoder)
        for fr in a.frames:
            row = dict(model=name, params=spec["params"], frames=fr, res=res, batch=a.batch, dtype=a.dtype,
                       weights_gib=round(weights_gib, 2))
            try:
                row.update(run_one(encoder, spec, fr, res, device, dtype, a.warmup, a.iters, a.batch), status="ok")
                print(f"  {name:<20} {fr:>3}f @{res}  tokens {row['tokens']:>6}  {row['ms_median']:8.1f} ms  "
                      f"peak {row['peak_alloc_gib']:.2f} GiB (reserved {row['peak_reserved_gib']:.2f})")
            except torch.OutOfMemoryError:
                row["status"] = "OOM"
                print(f"  {name:<20} {fr:>3}f @{res}  OOM")
            gc.collect(); torch.cuda.empty_cache()
            rows.append(row)
        del encoder
        gc.collect(); torch.cuda.empty_cache()

    keys = ["model", "params", "frames", "res", "batch", "dtype", "tokens", "weights_gib", "activ_gib",
            "peak_alloc_gib", "peak_reserved_gib", "ms_median", "ms_min", "clips_per_s", "status"]
    print("\n" + f"{'model':<20}{'frames':>7}{'tokens':>8}{'weights':>9}{'peak GiB':>10}{'median ms':>11}{'clips/s':>9}  status")
    for r in rows:
        f = lambda k, fmt: (format(r[k], fmt) if isinstance(r.get(k), (int, float)) else str(r.get(k, "-")))
        print(f"{r['model']:<20}{f('frames','>7'):>7}{f('tokens','>8'):>8}{f('weights_gib','.2f'):>9}"
              f"{f('peak_alloc_gib','.2f'):>10}{f('ms_median','.1f'):>11}{f('clips_per_s','.2f'):>9}  {r['status']}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
