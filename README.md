# JEPA bench (Windows + RTX 5060)

Small, self-contained scripts for poking at Meta's **V-JEPA** video encoders on one consumer Blackwell GPU.

## Which models (as of Sept 2026)

| Release | Date | What's used here |
|---|---|---|
| **V-JEPA 2.1** | 2026-03-16 | **Default.** ViT-B/L/g/G @384. Better dense, temporally consistent features. Weights from `dl.fbaipublicfiles.com`, code via `torch.hub` (`facebookresearch/vjepa2`, pinned commit). Not in `transformers` yet (PR #45497 still open). |
| V-JEPA 2 | 2025-06 | ViT-L/H/g @256, g @384. Used by `classify.py`: the only released SSv2 heads are for V-JEPA 2 (`facebook/vjepa2-vitl-fpc16-256-ssv2`). |

No official V-JEPA release newer than 2.1 had shown up as of Sept 2026. VL-JEPA, the vision-language variant, isn't covered here.

**Defaults:** `--model L` (V-JEPA 2.1 ViT-L/16 @384, 300M), `--dtype fp16`, `--frames 16`.
Aliases: `B L g G` = 2.1 sizes, `2-L 2-H 2-g 2-g384` = V-JEPA 2.

A 16-frame 384px clip becomes 8 x 24 x 24 = **4608 tokens** (tubelet 2x16x16). An image becomes 1 x 24 x 24 = 576 tokens. 2.1 has its own image patch embedding. V-JEPA 2 doesn't, so the image is used as a 2-frame clip.

## Run order

```powershell
# 0. once: from this folder
powershell -ExecutionPolicy Bypass -File .\setup.ps1     # or: run.bat setup
.\.venv\Scripts\Activate.ps1

# 1. get a couple of test clips (Kinetics-mini samples used in the HF docs)
mkdir clips
curl.exe -L -o clips\archery.mp4 https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4
curl.exe -L -o clips\bowling.mp4 https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4

# 2. scripts (each runs on its own; first run downloads code + weights)
python extract.py    clips\archery.mp4
python classify.py   clips\bowling.mp4
python pca_viz.py    clips\archery.mp4
python similarity.py clips
python bench.py
python hooks.py      clips\archery.mp4 --ablate 12
```

### run.bat (easiest)

Double-click `run.bat` for a menu. It activates `.venv`, checks that `python` is the venv interpreter, and offers to run setup if `.venv` is missing. Then pick a script and drag a clip into the window. You can also call it from a terminal, with no manual activation:

```bat
run.bat extract clips\archery.mp4 --model L
run.bat similarity clips
run.bat bench --sizes B L --frames 8 16
run.bat shell      & rem cmd prompt with the venv active
run.bat setup
```

Every script has `--help`. Shared options: `--model`, `--dtype {fp16,bf16,fp32}`, `--res`, `--frames`, `--stride` (take consecutive frames every N instead of spreading the frames across the whole clip).

`jepa_common.py` holds the shared loader and video I/O. Keep it next to the scripts.

## What each script should print

**setup.ps1**. It finishes with lines like:
```
torch       2.14.0+cu130  (CUDA 13.0, ...)
GPU         NVIDIA GeForce RTX 5060  |  8.0 GiB  |  compute capability 12.0
arch list   ... sm_120 ...
CUDA tensor fp16 2048x2048 matmul OK on cuda:0: ...
SDPA fp16 (ViT-L 16-frame @384 token count) OK
video decoder: torchcodec
```

**extract.py** turns an image or video into embeddings.
```
[input]  clips\archery.mp4: decoded (16, 3, H, W) uint8 (T,C,H,W) via torchcodec -> model input (1, 3, 16, 384, 384)
[grid]   tokens = T'8 x H'24 x W'24 = 4608
[tokens] (1, 4608, 1024)  (B, N, D)
[grid]   (1, 8, 24, 24, 1024)
[pooled] (1, 1024)
[time]   ... ms forward  peak ... GiB
```
With an image you get `(1, 576, 1024)`. `--hierarchical` also prints the 4 intermediate normed layers (5, 11, 17, 23 for ViT-L). `--out f.pt` saves everything.

**classify.py** prints the top-5 Something-Something-v2 actions:
```
Top-5 SSv2 actions:
  1.  41.3%  Pushing something from left to right
  ...
```
SSv2 has 174 hand-object *motion* classes ("Moving something up"), not scene or sport names. So Kinetics clips get plausible motion labels, not "archery". Use `--checkpoint facebook/vjepa2-vitg-fpc64-384-ssv2 --frames 64` for the 1B-param ViT-g head (tight on 8 GB).

**pca_viz.py** writes `outputs\<clip>_<model>_pca.png`: the top row is frames and the bottom row is the first 3 PCA components as RGB, one per temporal token. It also writes a `.gif`. PCA is fit jointly across frames, so the same object should keep the same colour over time. That consistency is what 2.1 improves. Try `--model 2-L` to compare with V-JEPA 2.

**similarity.py** prints an NxN cosine matrix, each clip's nearest neighbour, and off-diagonal stats. It saves `outputs\similarity.csv`, `similarity.png` (heatmap) and `similarity_embeddings.pt`. Mean-pooled JEPA features share a large common direction, so unrelated clips often score 0.5-0.8. Compare scores to each other, not to 0.

**bench.py** does a sweep with random input and writes `outputs\bench.csv`:
```
model                frames  tokens  weights  peak GiB  median ms  clips/s  status
vjepa2.1-vitb-384         8    2304      ...      ...        ...      ...  ok
vjepa2.1-vitl-384        16    4608      ...      ...        ...      ...  ok
...                      64   18432      ...                                OOM
```
`--sizes B L g G --frames 8 16 32 64`. `--random-weights` skips the checkpoint downloads (speed and VRAM are the same). Use it for g/G. OOM configs are recorded and the sweep carries on.

**hooks.py** uses nnsight to print per-block activation stats:
```
layer              shape   mean|x|      std    max|x|  cos(prev)  cos(final)  eff.rank
    0     (1, 4608, 1024)     ...
   23     (1, 4608, 1024)     ...
[ablate] block 12 skipped -> final tokens cos vs clean: mean 0.9xx ...
```
`--layers 0 11 23` picks blocks and `--save outputs\acts.pt` dumps the activations. The core pattern, if you want to write your own interventions:
```python
from nnsight import NNsight
m = NNsight(encoder); acts = {}
with m.trace(x):
    for i in range(len(encoder.blocks)):
        acts[i] = m.blocks[i].output[0].save()   # 2.1 blocks return (tokens, attn)
```
In nnsight 0.5+, define containers *before* the `with`, touch modules in execution order, and run it from a `.py` file (nnsight reads the source).

## V-JEPA → LLM captioner (LLaVA / PerceptionLM stage 1)

A frozen V-JEPA 2.1 ViT-L feeds a frozen **Qwen2.5-1.5B-Instruct** (fp16) through a 2-layer MLP projector, `Linear(1024→1536) → GELU → Linear(1536→1536)`. Only the projector (3.9M params) trains.

```powershell
python llava_prep.py          # 20k LLaVA-Pretrain images + BLIP captions -> data\llava20k, V-JEPA features cached once
python train_projector.py     # 1 epoch -> outputs\projector_g12.pt
python caption.py photo.jpg   # or a video: python caption.py clips\archery.mp4 [--video frames]
```

In the `run.bat` menu these are **D**, **T** and **C**. To just caption things, double-click `caption.bat` and drag images or videos into its window, or drop files onto `caption.bat` itself. `vlm_common.py` holds the projector, the prompt layout and the loss.

```
image -> V-JEPA tokens (1, 576, 1024) -> 2x2 average pool (1, 144, 1024) -> projector (1, 144, 1536)
<|im_start|>system ...<|im_end|> <|im_start|>user\n [144 image tokens] \nDescribe this image.<|im_end|> <|im_start|>assistant\n caption<|im_end|>
```

Loss is computed only on the caption tokens and the closing `<|im_end|>`. Videos are encoded as clips (8 x 24 x 24 tokens) and averaged over time onto the same grid.

Measured on the RTX 5060 Laptop GPU (8 GB), fp16:

| Step | Time | PyTorch VRAM peak |
|---|---|---|
| `llava_prep.py` download: 20k JPEGs (948 MiB) via range requests into the 25.5 GiB `images.zip` | ~3 min | – |
| `llava_prep.py` cache: 20,000 x 144 x 1024 float16 = 5.5 GiB | 11.2 min (30 img/s) | 1.4 GiB |
| `train_projector.py`: 625 steps, batch 32 = micro-batch 4 x accum 8 | 27.6 min (12 samples/s) | 5.1 GiB alloc / 5.4 GiB reserved |
| `caption.py`: image / 16-frame clip | ~1.4 s / ~2 s after loading | 3.5 / 3.8 GiB |

The loss went from 6.29 to 3.81 (mean of the last 125 steps), with no fp16 overflow steps. On held-out images the captions get the gist ("a bride and her bridesmaids at a wedding", "a 12v power supply with a switch and an indicator light") but often miss the specifics. The Kinetics clips only get generic person descriptions. That is expected for stage 1, which only aligns the two models; instruction tuning (stage 2) is what adds detail. See `outputs\caption_tests.log` and `outputs\caption_samples\`.

- `llava_prep.py --grid 24` keeps all 576 tokens (~22 GiB cache). Then train with `--grid 24 --grad-ckpt --micro-batch 8 --accum 4` (not timed here; expect several times longer).
- `caption.py --video frames` encodes each sampled frame as an image instead, which is closer to the training data.
- If the loss ever turns NaN, use `--dtype bf16`.
- `data\llava20k` is ~6.4 GB and lives inside OneDrive\Desktop. OneDrive wasn't running during setup. If you turn sync on, move it and pass `--data C:\jepa_data`, or `setx JEPA_DATA C:\jepa_data`.

## Where things download

| What | When | Where |
|---|---|---|
| vjepa2 model code (GitHub zip) | first 2.1 or 2 load | `%USERPROFILE%\.cache\torch\hub\facebookresearch_vjepa2_<commit>` |
| V-JEPA checkpoints | first use of each size | `%USERPROFILE%\.cache\torch\hub\checkpoints\` |
| SSv2 classifier | first `classify.py` | `%USERPROFILE%\.cache\huggingface\hub\` |
| Qwen2.5-1.5B-Instruct (~3 GB) | first `train_projector.py` / `caption.py` | `%USERPROFILE%\.cache\huggingface\hub\` |
| LLaVA-Pretrain captions JSON | first `llava_prep.py` | `%USERPROFILE%\.cache\huggingface\hub\` |
| 20k images + V-JEPA feature cache (~6.4 GB) | `llava_prep.py` | `.\data\llava20k\` (or `--data`) |

To move the caches to another drive: `setx TORCH_HOME D:\cache\torch` and `setx HF_HOME D:\cache\hf`, then open a new terminal. Checkpoints hold the encoder, EMA encoder and predictor, so they are much bigger than the encoder alone. The whole file is loaded into CPU RAM once, and only the encoder goes to the GPU.

Offline or behind a firewall: `git clone https://github.com/facebookresearch/vjepa2` and `set VJEPA2_REPO=C:\path\to\vjepa2`.

## Windows troubleshooting

**`torch.cuda.is_available()` is False / `CUDA error: no kernel image is available` / `sm_120 is not compatible`**
You have a CPU wheel or a wheel older than CUDA 12.8. Blackwell needs cu128+. Run `python -c "import torch;print(torch.__version__, torch.version.cuda)"`; it should say `+cu130` (or `+cu128`). Fix with `.\setup.ps1 -Recreate`. For cu130 the driver must be **580+** (`nvidia-smi`). On an older driver, update it or use `.\setup.ps1 -Recreate -Cuda cu128` (torch 2.11).

**`RuntimeError: Could not load libtorchcodec` / `FileNotFoundError ... avcodec-*.dll`**
Python ignores `PATH` when loading extension DLLs. `jepa_common.py` registers `.\ffmpeg\bin`, `%JEPA_FFMPEG_BIN%`, or any PATH folder with `avcodec-*.dll`. Checks:
- You need a **shared** FFmpeg build (DLLs). setup.ps1 fetches 8.1, and torchcodec 0.16 supports FFmpeg 4-8 (9 on newer wheels). The static "essentials/full" builds have no DLLs.
- winget install: `setx JEPA_FFMPEG_BIN "<winget ffmpeg folder>\bin"`.
- Don't want to deal with it? `set JEPA_VIDEO_BACKEND=pyav`. PyAV bundles its own FFmpeg. It is slower on long videos because it decodes every frame.

**OneDrive is syncing `.venv` / slow installs / "file in use" errors**
This folder is on your OneDrive Desktop. `.venv` has thousands of files, so right-click it and choose *Free up space*, or pause syncing during setup. Moving the project to e.g. `C:\jepa` avoids this. After moving it, run `powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Recreate`, because the venv stores absolute paths.

**`running scripts is disabled on this system`**
Use `powershell -ExecutionPolicy Bypass -File .\setup.ps1`, or `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**`torch.OutOfMemoryError` (8 GB card)**
Lower `--frames` (32, then 16, then 8) or `--res` (e.g. 256). Use B/L instead of g/G. Close apps that use the GPU (browsers, games, OBS). Token count grows linearly with frames and with res². ViT-G weights alone are about 4 GB in fp16.

**NaN/Inf warning in fp16**
Use `--dtype bf16`. RTX 50xx supports bfloat16, and Meta evaluates in bf16. RoPE is kept in fp32 under autocast either way.

**`HTTP Error 403/404` downloading `.pt`, or `urlopen error`**
Corporate proxy or antivirus. Download the file from the README table at github.com/facebookresearch/vjepa2 into `%USERPROFILE%\.cache\torch\hub\checkpoints\` with the same filename.

**`ModuleNotFoundError: No module named 'app.vjepa_2_1'` / `'src.hub'`**
You have a file or folder named `app`, `src` or `evals` next to the scripts, and it shadows the vjepa2 hub code. Rename it. A half-downloaded hub cache causes the same error: delete `%USERPROFILE%\.cache\torch\hub\facebookresearch_vjepa2_*`.

**`OSError: [WinError 1455] The paging file is too small`**
Loading a large checkpoint needs RAM plus virtual memory. Raise the page file size (System > Advanced > Performance > Virtual memory), or use a smaller model.

**`OMP: Error #15: Initializing libiomp5md.dll`**
Handled by setting `KMP_DUPLICATE_LIB_OK=TRUE` in `jepa_common.py`. If it still appears, you are importing torch from a different environment. Activate `.venv`.

**`UnicodeEncodeError` in the console**
`setx PYTHONUTF8 1` and open a new terminal.

**Hugging Face symlink warning / slow cache**
This is harmless on Windows. Turning on Developer Mode enables symlinks.

**`torch.compile` (`bench.py --compile`) fails**
Triton isn't bundled on Windows. Install it with `uv pip install triton-windows`, or skip `--compile`.

**Training suddenly runs ~8x slower, with no error**
When the card is full, Windows moves VRAM into system RAM instead of raising out-of-memory. Check the `[vram]` line that `train_projector.py` prints after step 1 and keep it above 0. Close GPU-heavy apps (animated wallpapers, games, video), or use `--micro-batch 2 --accum 16` or `--grad-ckpt`. The script pads every caption to one width so tensor shapes never change and PyTorch's cache doesn't fragment.

**Long path errors when extracting the hub zip**
`reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f` (admin), or set `TORCH_HOME` to a short path like `C:\th`.

## Notes and caveats

- Upstream `src/hub/backbones.py` points checkpoint URLs at `localhost`, so a plain `torch.hub.load(..., pretrained=True)` fails. `jepa_common.load_encoder` builds the model with `pretrained=False` and downloads from `dl.fbaipublicfiles.com` itself.
- Preprocessing follows Meta's eval: resize the short side to `res*256/224`, center crop, ImageNet mean/std.
- Weights are cast to fp16 and the forward pass runs under `torch.autocast`. Meta's RoPE code mixes dtypes, and pure `.half()` without autocast would fail inside SDPA.
- Testing hooks: `set JEPA_RANDOM_WEIGHTS=1` runs any script with random weights and no download.

Sources: [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2), [HF V-JEPA 2 docs](https://huggingface.co/docs/transformers/model_doc/vjepa2), [transformers V-JEPA 2.1 issue](https://github.com/huggingface/transformers/issues/45496), [torchcodec](https://github.com/meta-pytorch/torchcodec), [PyTorch wheels](https://download.pytorch.org/whl/cu130).
