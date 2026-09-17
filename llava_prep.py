"""llava_prep.py -- LLaVA-Pretrain subset (images + BLIP captions) -> cached V-JEPA features. Run once before training.

  python llava_prep.py                  # 20k samples, 24x24 tokens average-pooled to 12x12 (144 x 1024), ~5.5 GiB
  python llava_prep.py --grid 24        # keep all 576 tokens per image (~22 GiB)
  python llava_prep.py --n 2000 --data D:\\llava2k

Step 1, download: a seeded random subset of huggingface.co/datasets/liuhaotian/LLaVA-Pretrain. Only the central
  directory of images.zip and the chosen JPEGs are fetched (HTTP range requests), never the whole multi-GB zip.
  Captions come from blip_laion_cc_sbu_558k.json, whose "gpt" turn is the BLIP caption.
Step 2, cache: V-JEPA 2.1 ViT-L (same loader + preprocessing as extract.py) encodes each image once ->
  feats_<model>_g<grid>.npy, float16 [N, grid*grid, 1024], row i = samples.json[i]. Training never runs the encoder.
Both steps resume where they stopped.
"""
import argparse
import json
import random
import struct
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch

import jepa_common as jc
import vlm_common as vc

REPO = "liuhaotian/LLaVA-Pretrain"
_tls = threading.local()  # one HTTP session per download thread


# ------------------------------------------------------------------------------------------------ 1. download
def resolve(filename):
    """(signed CDN url, size) of a repo file. Range requests then hit the CDN directly (no Hub call per image)."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    meta = get_hf_file_metadata(hf_hub_url(REPO, filename, repo_type="dataset"))
    return meta.location, meta.size


def get_range(url, start, end):
    """Bytes [start, end) of a remote file."""
    if not hasattr(_tls, "session"):
        _tls.session = requests.Session()
    r = _tls.session.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=60, stream=True)
    if r.status_code != 206:  # never fall back to streaming the whole zip
        r.close()
        raise IOError(f"HTTP {r.status_code} for a range request")
    return r.content


class RemoteFile:
    """Just enough of a file object (seek/tell/read) for zipfile to read the zip's central directory remotely."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        data = get_range(self.url, self.pos, end) if end > self.pos else b""
        self.pos = end
        return data


def fetch_member(src, info, dst):
    """One JPEG: range-read its local header + data, inflate if needed, check CRC, write dst."""
    for attempt in range(5):
        url = src["url"]
        try:
            start = info.header_offset  # local header (30 bytes) + name + extra field (read generously) + data
            raw = get_range(url, start, min(start + 30 + len(info.filename.encode()) + 1024 + info.compress_size, src["size"]))
            if raw[:4] != b"PK\x03\x04":
                raise IOError("bad local header")
            n_name, n_extra = struct.unpack("<HH", raw[26:30])
            body = raw[30 + n_name + n_extra:][:info.compress_size]
            if info.compress_type == zipfile.ZIP_DEFLATED:
                body = zlib.decompress(body, -15)
            if zlib.crc32(body) != info.CRC:
                raise IOError("CRC mismatch")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.with_suffix(".part").write_bytes(body)
            dst.with_suffix(".part").replace(dst)
            return True
        except Exception as e:
            last = e
            if "HTTP 403" in str(e) or "HTTP 410" in str(e):  # signed CDN url expired -> resolve once, retry
                with src["lock"]:
                    if src["url"] == url:
                        src["url"] = resolve("images.zip")[0]
            time.sleep(2 ** attempt)
    print(f"[warn]   {dst.name}: {last}")
    return False


def download(a):
    """Pick the subset, fetch its JPEGs, write samples.json = [{id, image, caption}, ...]."""
    path = a.data / "samples.json"
    if path.exists():
        samples = json.loads(path.read_text(encoding="utf-8"))
        print(f"[data]   {path} exists: {len(samples):,} samples (delete it to pick a new subset)")
        return samples

    from huggingface_hub import hf_hub_download
    t0 = time.time()
    records = json.loads(Path(hf_hub_download(REPO, "blip_laion_cc_sbu_558k.json", repo_type="dataset")).read_text(encoding="utf-8"))
    url, size = resolve("images.zip")
    zf = zipfile.ZipFile(RemoteFile(url, size))  # reads end-of-zip records + central directory (~tens of MB)
    members = {"/".join(i.filename.split("/")[-2:]): i for i in zf.infolist() if not i.is_dir()}
    print(f"[data]   {REPO}: {len(records):,} captions, images.zip = {size / 2**30:.1f} GiB / {len(members):,} files "
          f"(index read in {time.time() - t0:.0f}s)")
    picks = random.Random(a.seed).sample([r for r in records if r["image"] in members], a.n)

    img_dir = a.data / "images"
    todo = [r for r in picks if not (img_dir / r["image"]).exists()]
    src = {"url": url, "size": size, "lock": threading.Lock()}
    mib = sum(members[r["image"]].file_size for r in todo) / 2**20
    print(f"[data]   downloading {len(todo):,} of {a.n:,} JPEGs ({mib:.0f} MiB) with {a.threads} threads -> {img_dir}")
    t0, failed = time.time(), 0
    with ThreadPoolExecutor(a.threads) as pool:
        for k, ok in enumerate(pool.map(lambda r: fetch_member(src, members[r["image"]], img_dir / r["image"]), todo), 1):
            failed += not ok
            if k % 2000 == 0 or k == len(todo):
                print(f"[data]   {k:,}/{len(todo):,}  {k / (time.time() - t0):.0f} img/s  {failed} failed")

    def blip_caption(r):
        return next(c["value"] for c in r["conversations"] if c["from"] == "gpt").strip()

    samples = [{"id": r["id"], "image": r["image"], "caption": blip_caption(r)} for r in picks if (img_dir / r["image"]).exists()]
    path.write_text(json.dumps(samples, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"[data]   {len(samples):,} image-caption pairs -> {path}  (e.g. {samples[0]['image']}: \"{samples[0]['caption']}\")")
    return samples


# ------------------------------------------------------------------------------------------------ 2. cache
class Images(torch.utils.data.Dataset):
    """JPEG -> V-JEPA input [3, 1, 384, 384] (resize short side, center crop, ImageNet norm, like extract.py)."""

    def __init__(self, paths, spec):
        self.paths, self.spec = paths, spec

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x, _ = jc.to_model_input(jc.read_image(self.paths[i]), self.spec)  # [1, 3, 1, 384, 384]
        return x[0]


def cache(a, samples):
    name, spec = jc.resolve_model(a.model)
    npy = a.data / f"feats_{name}_g{a.grid}.npy"
    meta_path, N = npy.with_suffix(".json"), len(samples)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("done") == N and npy.exists():
        print(f"[cache]  {npy.name} already complete {tuple(meta['shape'])}")
        return

    device = jc.get_device()
    dtype = jc.resolve_dtype(a.dtype, device)
    encoder, name, spec = jc.load_encoder(a.model, device, dtype)
    shape = [N, a.grid * a.grid, encoder.embed_dim]
    start = meta.get("done", 0) if meta.get("shape") == shape and npy.exists() else 0
    feats = np.lib.format.open_memmap(npy, mode="r+" if start else "w+", dtype=np.float16, shape=tuple(shape))
    meta = {"model": name, "grid": a.grid, "shape": shape, "done": start, "dtype": a.dtype}
    print(f"[cache]  {npy} {feats.shape} float16 (N, tokens, D) = {feats.nbytes / 2**30:.2f} GiB, starting at row {start}")

    paths = [a.data / "images" / s["image"] for s in samples]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(Images(paths, spec), range(start, N)),
                                         batch_size=a.batch, num_workers=a.workers, pin_memory=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    i, bad, t0 = start, 0, time.time()
    for b, x in enumerate(loader):
        x = x.to(device, dtype, non_blocking=True)  # [B, 3, 1, 384, 384]
        t, h, w = jc.token_grid(x, family=spec["family"])
        with torch.inference_mode(), jc.autocast(device, dtype):
            tokens = encoder(x)  # [B, t*h*w, D] = [B, 576, 1024]
        pooled = vc.pool_tokens(tokens.float(), t, h, w, a.grid)  # [B, grid*grid, D]
        bad += int((~torch.isfinite(pooled)).flatten(1).any(1).sum())
        feats[i:i + len(x)] = pooled.half().cpu().numpy()
        i += len(x)
        if b == 0:
            print(f"[shapes] input {tuple(x.shape)} (B,C,T,H,W) -> tokens {tuple(tokens.shape)} (B, {t}x{h}x{w}, D) "
                  f"-> pooled {tuple(pooled.shape)} (B, {a.grid}x{a.grid}, D)")
        if b % 25 == 24 or i == N:  # checkpoint progress so an interrupted run resumes here
            feats.flush()
            meta["done"] = i
            meta_path.write_text(json.dumps(meta))
            el = time.time() - t0
            print(f"[cache]  {i:,}/{N:,}  {(i - start) / el:.0f} img/s  ETA {(N - i) * el / (i - start) / 60:.1f} min  "
                  f"{jc.gpu_mem_str(device)}")
    if bad:
        print(f"[warn]   {bad} images gave non-finite features in {a.dtype}; delete {npy.name} and use --dtype bf16")
    print(f"[done]   encoded {N - start:,} images in {(time.time() - t0) / 60:.1f} min -> {npy}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20000, help="image-caption pairs (default 20000)")
    p.add_argument("--grid", type=int, default=12, help="pool the 24x24 token grid to grid x grid (default 12 = 144 tokens; 24 = all 576)")
    p.add_argument("--data", type=Path, default=vc.DATA, help=r"output folder (default .\data\llava20k, or env JEPA_DATA)")
    p.add_argument("--model", default="L", help=jc.model_help())
    p.add_argument("--dtype", default=jc.DEFAULT_DTYPE, choices=list(jc.DTYPES))
    p.add_argument("--batch", type=int, default=32, help="encoder batch size (default 32)")
    p.add_argument("--workers", type=int, default=4, help="JPEG decode processes (default 4)")
    p.add_argument("--threads", type=int, default=16, help="parallel downloads (default 16)")
    p.add_argument("--seed", type=int, default=0, help="subset seed (default 0)")
    a = p.parse_args()

    a.data.mkdir(parents=True, exist_ok=True)
    if "onedrive" in str(a.data.resolve()).lower():
        print(f"[note]   {a.data} is inside OneDrive, which will try to sync the multi-GB cache. "
              f"--data C:\\jepa_data keeps it local.")
    samples = download(a)
    cache(a, samples)


if __name__ == "__main__":
    main()
