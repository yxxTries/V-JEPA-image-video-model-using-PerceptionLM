"""Shared helpers for the JEPA bench: model registry, loading, video/image I/O, preprocessing.

Every script in this folder imports this file, so keep it next to them.

Weights download on first use into %TORCH_HOME% (default: %USERPROFILE%\\.cache\\torch\\hub).
Model code comes from github.com/facebookresearch/vjepa2 via torch.hub (pinned commit below),
or from a local clone if you set VJEPA2_REPO=C:\\path\\to\\vjepa2.
"""
from __future__ import annotations

import math
import os
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

# Windows consoles default to cp1252; make prints safe.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows without Developer Mode
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")            # duplicate OpenMP runtimes on Windows

HERE = Path(__file__).resolve().parent

warnings.filterwarnings("ignore", message=".*sdp_kernel.*")          # deprecated ctx mgr in Meta's code
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")  # deprecated import path in Meta's code

# facebookresearch/vjepa2 HEAD as of 2026-03-23 (V-JEPA 2.1 release + fixes).
VJEPA2_REF = os.environ.get("VJEPA2_REF", "204698b45b3712590f06245fbfba32d3be539812")
CKPT_BASE = "https://dl.fbaipublicfiles.com/vjepa2"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------------------------
# Model registry.  hub = torch.hub entry in facebookresearch/vjepa2, file = checkpoint on
# dl.fbaipublicfiles.com, key = which encoder weights to take from the checkpoint.
# ---------------------------------------------------------------------------------------------
MODELS = {
    # V-JEPA 2.1 (released 2026-03-16) -- newest; dense, temporally consistent features
    "vjepa2.1-vitb-384": dict(hub="vjepa2_1_vit_base_384", file="vjepa2_1_vitb_dist_vitG_384", key="ema_encoder", res=384, family="2.1", params="80M"),
    "vjepa2.1-vitl-384": dict(hub="vjepa2_1_vit_large_384", file="vjepa2_1_vitl_dist_vitG_384", key="ema_encoder", res=384, family="2.1", params="300M"),
    "vjepa2.1-vitg-384": dict(hub="vjepa2_1_vit_giant_384", file="vjepa2_1_vitg_384", key="target_encoder", res=384, family="2.1", params="1B"),
    "vjepa2.1-vitG-384": dict(hub="vjepa2_1_vit_gigantic_384", file="vjepa2_1_vitG_384", key="target_encoder", res=384, family="2.1", params="2B"),
    # V-JEPA 2 (2025-06) -- kept for comparison
    "vjepa2-vitl-256": dict(hub="vjepa2_vit_large", file="vitl", key="target_encoder", res=256, family="2", params="300M"),
    "vjepa2-vith-256": dict(hub="vjepa2_vit_huge", file="vith", key="target_encoder", res=256, family="2", params="600M"),
    "vjepa2-vitg-256": dict(hub="vjepa2_vit_giant", file="vitg", key="target_encoder", res=256, family="2", params="1B"),
    "vjepa2-vitg-384": dict(hub="vjepa2_vit_giant_384", file="vitg-384", key="target_encoder", res=384, family="2", params="1B"),
}
ALIASES = {"B": "vjepa2.1-vitb-384", "L": "vjepa2.1-vitl-384", "g": "vjepa2.1-vitg-384", "G": "vjepa2.1-vitG-384",
           "2-L": "vjepa2-vitl-256", "2-H": "vjepa2-vith-256", "2-g": "vjepa2-vitg-256", "2-g384": "vjepa2-vitg-384"}
DEFAULT_MODEL = "L"          # V-JEPA 2.1 ViT-L/16 @ 384
DEFAULT_FRAMES = 16
DEFAULT_DTYPE = "fp16"


def resolve_model(name: str) -> tuple[str, dict]:
    name = ALIASES.get(name, name)
    if name not in MODELS:
        raise SystemExit(f"Unknown model '{name}'. Choose from: {', '.join(list(ALIASES) + list(MODELS))}")
    return name, MODELS[name]


def model_help() -> str:
    return "model name or alias (" + ", ".join(f"{k}={v}" for k, v in ALIASES.items()) + ")"


# ---------------------------------------------------------------------------------------------
# Device / dtype
# ---------------------------------------------------------------------------------------------
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def get_device(require_cuda: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    msg = ("CUDA not available -- torch %s (built for CUDA %s). Re-run setup.ps1 and see README "
           "'torch.cuda.is_available() is False'." % (torch.__version__, torch.version.cuda))
    if require_cuda:
        raise SystemExit(msg)
    print("[warn] " + msg + " Falling back to CPU (slow, fp32).")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return DTYPES[name]


def autocast(device: torch.device, dtype: torch.dtype):
    """Weights are cast to `dtype`; autocast keeps RoPE math in fp32 and SDPA/Linear in `dtype`."""
    enabled = device.type == "cuda" and dtype != torch.float32
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


# ---------------------------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------------------------
def _clean_keys(sd: dict) -> dict:
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


def load_encoder(name: str = DEFAULT_MODEL, device=None, dtype=torch.float16, pretrained: bool = True,
                 verbose: bool = True):
    """Build a V-JEPA encoder (predictor discarded) and load weights.

    Note: upstream hub code has VJEPA_BASE_URL pointing at localhost, so we build with
    pretrained=False and download the checkpoint ourselves.
    """
    name, spec = resolve_model(name)
    device = device or get_device()
    pretrained = pretrained and os.environ.get("JEPA_RANDOM_WEIGHTS", "0") != "1"

    t0 = time.time()
    local = os.environ.get("VJEPA2_REPO")
    if local:
        encoder, predictor = torch.hub.load(local, spec["hub"], source="local", pretrained=False)
    else:
        encoder, predictor = torch.hub.load(f"facebookresearch/vjepa2:{VJEPA2_REF}", spec["hub"],
                                            pretrained=False, trust_repo=True, skip_validation=True,
                                            verbose=False)
    del predictor

    if pretrained:
        url = f"{CKPT_BASE}/{spec['file']}.pt"
        if verbose:
            print(f"[load] {name}: checkpoint {url} (downloads once, then cached)")
        try:
            ckpt = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=True)
        except Exception as e:  # older checkpoints may pickle non-tensor objects
            if "weights_only" not in str(e) and "Unsupported global" not in str(e):
                raise
            ckpt = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=False)
        sd = _clean_keys(ckpt[spec["key"]])
        del ckpt
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
        missing = [k for k in missing if "pos_embed" not in k]
        unexpected = [k for k in unexpected if "pos_embed" not in k]
        if missing or unexpected:
            print(f"[warn] state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    elif verbose:
        print(f"[load] {name}: RANDOM weights (no download)")

    encoder = encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
    if verbose:
        n = sum(p.numel() for p in encoder.parameters()) / 1e6
        print(f"[load] {name}: {n:.0f}M params, {len(encoder.blocks)} blocks, dim {encoder.embed_dim}, "
              f"{str(dtype).replace('torch.', '')} on {device} ({time.time() - t0:.1f}s)")
    return encoder, name, spec


# ---------------------------------------------------------------------------------------------
# Media I/O
# ---------------------------------------------------------------------------------------------
_DECODER = None


def _register_ffmpeg_dlls():
    """Python 3.8+ on Windows ignores PATH for extension-module DLLs, so torchcodec can't find
    FFmpeg's avcodec-*.dll unless we register the folder. Looks in: $JEPA_FFMPEG_BIN, .\\ffmpeg\\bin
    (created by setup.ps1), then every PATH entry containing avcodec-*.dll."""
    if os.name != "nt":
        return None
    candidates = [os.environ.get("JEPA_FFMPEG_BIN"), str(HERE / "ffmpeg" / "bin")]
    candidates += os.environ.get("PATH", "").split(os.pathsep)
    for d in candidates:
        if d and Path(d).is_dir() and any(Path(d).glob("avcodec-*.dll")):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return d
    return None


def video_backend() -> str:
    """'torchcodec' if it imports (needs FFmpeg shared DLLs), else 'pyav'."""
    global _DECODER
    if _DECODER is None and os.environ.get("JEPA_VIDEO_BACKEND", "").lower() == "pyav":
        _DECODER = "pyav"
    if _DECODER is None:
        _register_ffmpeg_dlls()
        try:
            from torchcodec.decoders import VideoDecoder  # noqa: F401
            _DECODER = "torchcodec"
        except Exception as e:
            try:
                import av  # noqa: F401
                _DECODER = "pyav"
                print(f"[video] torchcodec unavailable ({type(e).__name__}: {str(e).splitlines()[0][:120]}); using PyAV")
            except ImportError:
                raise SystemExit("No video decoder: install FFmpeg for torchcodec, or `uv pip install av`.")
    return _DECODER


def sample_indices(n_total: int, n_frames: int, stride: int | None = None, start: int = 0) -> list[int]:
    """Uniformly span the clip (stride=None) or take consecutive frames every `stride` frames."""
    if n_total <= 0:
        raise ValueError("video has no frames")
    if stride:
        idx = [start + i * stride for i in range(n_frames)]
    else:
        idx = torch.linspace(0, n_total - 1, n_frames).round().long().tolist()
    return [min(i, n_total - 1) for i in idx]


def read_video(path, n_frames: int = DEFAULT_FRAMES, stride: int | None = None) -> torch.Tensor:
    """Return uint8 tensor [T, C, H, W]."""
    path = str(path)
    if video_backend() == "torchcodec":
        from torchcodec.decoders import VideoDecoder
        dec = VideoDecoder(path, dimension_order="NCHW")
        idx = sample_indices(len(dec), n_frames, stride)
        return dec.get_frames_at(indices=idx).data
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames = [torch.from_numpy(f.to_ndarray(format="rgb24")) for f in container.decode(stream)]
    idx = sample_indices(len(frames), n_frames, stride)
    return torch.stack([frames[i] for i in idx]).permute(0, 3, 1, 2).contiguous()


def read_image(path) -> torch.Tensor:
    """Return uint8 tensor [1, C, H, W]."""
    import numpy as np
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)[None]


def is_image(path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def list_media(folder) -> list[Path]:
    exts = VIDEO_EXTS | IMAGE_EXTS
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts)


def resize_crop(frames: torch.Tensor, res: int) -> torch.Tensor:
    """uint8/float [T,C,H,W] -> float [T,C,res,res] in [0,1]: short side to res*256/224, center crop."""
    x = frames.float() / 255.0 if frames.dtype == torch.uint8 else frames.float()
    short = int(res * 256 / 224)
    _, _, h, w = x.shape
    scale = short / min(h, w)
    nh, nw = max(short, round(h * scale)), max(short, round(w * scale))
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)
    top, left = (nh - res) // 2, (nw - res) // 2
    return x[:, :, top:top + res, left:left + res].clamp(0, 1)


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def to_model_input(frames: torch.Tensor, spec: dict, res: int | None = None):
    """uint8 [T,C,H,W] -> (x [1,C,T,res,res], display frames [T,C,res,res] in [0,1]).

    Images (T=1): V-JEPA 2.1 has a dedicated image patch embed (tubelet 1). V-JEPA 2 has no image
    path, so the frame is duplicated to form one 2-frame tubelet.
    Videos: T must be even (tubelet size 2) -- the last frame is repeated if needed.
    """
    res = res or spec["res"]
    disp = resize_crop(frames, res)
    T = disp.shape[0]
    if T == 1 and spec["family"] == "2":
        disp_in = disp.repeat(2, 1, 1, 1)
    elif T > 1 and T % 2:
        disp_in = torch.cat([disp, disp[-1:]], 0)
    else:
        disp_in = disp
    x = normalize(disp_in).permute(1, 0, 2, 3)[None].contiguous()  # [1,C,T,H,W]
    return x, disp


def token_grid(x: torch.Tensor, patch: int = 16, tubelet: int = 2, family: str = "2.1") -> tuple[int, int, int]:
    """(T', H', W') token grid for an input [B,C,T,H,W]."""
    _, _, T, H, W = x.shape
    t = 1 if (T == 1 and family == "2.1") else T // tubelet
    return t, H // patch, W // patch


def load_media(path, spec, frames=DEFAULT_FRAMES, stride=None, res=None):
    raw = read_image(path) if is_image(path) else read_video(path, frames, stride)
    x, disp = to_model_input(raw, spec, res)
    return x, disp, raw


def gpu_mem_str(device) -> str:
    if device.type != "cuda":
        return ""
    return f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB alloc / {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved"


def check_finite(t: torch.Tensor, dtype_name: str):
    if not torch.isfinite(t).all():
        print(f"[warn] non-finite values in output with {dtype_name}. Try --dtype bf16 (RTX 50xx supports it).")


def add_common_args(p, frames=True):
    p.add_argument("--model", default=DEFAULT_MODEL, help=model_help())
    p.add_argument("--dtype", default=DEFAULT_DTYPE, choices=list(DTYPES), help="weights/activation dtype (default fp16)")
    p.add_argument("--res", type=int, default=None, help="input resolution (default: model's native, 384 for 2.1)")
    if frames:
        p.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="frames per clip (default 16)")
        p.add_argument("--stride", type=int, default=None, help="take consecutive frames every N (default: span whole clip)")
    return p
