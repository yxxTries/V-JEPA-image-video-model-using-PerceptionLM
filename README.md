# JEPA bench

A small research playground for **V-JEPA 2.1**, Meta's self-supervised video model. It has two halves:

1. **Look inside the encoder.** Turn images and clips into embeddings, visualise what the model sees, compare clips, measure speed and VRAM, and read per-layer activations.
2. **Teach it to talk.** A tiny captioner built with the LLaVA / PerceptionLM *stage 1* recipe: the vision encoder and the language model stay **frozen**, and only a small MLP "projector" is trained to translate V-JEPA features into something Qwen2.5 can read.

Everything runs on a single 8 GB laptop GPU (RTX 5060). Nothing here needs a cluster.

## The idea in one paragraph

V-JEPA is trained by predicting the *features* of masked parts of a video, never pixels and never text. So it learns a strong visual representation but has no words attached to it. The question this repo pokes at: how much of that representation is already linguistically usable? Freeze everything, train a 3.9M-parameter MLP between the two models on 20k image-caption pairs, and see what comes out. Answer so far: the gist, not the specifics — which is exactly what stage-1 alignment is supposed to give you.

## What each file does

**Shared plumbing**

| File | What's in it |
|---|---|
| `jepa_common.py` | Loads V-JEPA checkpoints, decodes video/images, does Meta's preprocessing (resize short side, center crop, ImageNet normalize), handles dtype/autocast and the FFmpeg DLL mess on Windows. Every script imports this. |
| `vlm_common.py` | The captioner half: the projector module, the ChatML prompt layout, the fixed-shape caption loss, greedy generation. |

**Probing the encoder**

| Script | What it does |
|---|---|
| `extract.py` | Image or video → embeddings. Prints tensor shapes at each stage. `--hierarchical` also returns 4 intermediate normed layers. The "hello world" of the repo. |
| `pca_viz.py` | Fits a 3-component PCA over all patch tokens of a clip and paints them as RGB — the picture from the V-JEPA 2.1 paper. Fitting jointly across frames means a temporally consistent model keeps an object the same colour over time. |
| `similarity.py` | Mean-pools each clip in a folder into one L2-normalised vector, prints the cosine-similarity matrix and each clip's nearest neighbour, saves a CSV + heatmap. A quick check that the embedding space is sane. |
| `classify.py` | Something-Something-v2 action recognition, top-5 labels. Uses a V-JEPA **2** ViT-L + attentive probe, because Meta hasn't released SSv2 heads for 2.1. SSv2 labels are hand-object interactions, not sports. |
| `hooks.py` | Per-layer activations via `nnsight`. Stats for every transformer block, optional save to `.pt`, and `--ablate N` zeroes a block's residual update so you can measure what it contributed. |
| `bench.py` | Sweeps model size × frame count, recording latency and peak VRAM on synthetic input. OOM configs are logged and the sweep continues. → `outputs\bench.csv`. |

**The captioner (three steps, in order)**

| Script | What it does |
|---|---|
| `llava_prep.py` | Picks a seeded 20k subset of LLaVA-Pretrain, downloads only those JPEGs (HTTP range reads into a 25.5 GiB zip — no full download), then runs V-JEPA once per image and caches pooled features to a float16 memmap. Resumable. Training never touches the encoder again. |
| `train_projector.py` | Trains **only** the MLP projector on those cached features. Loss is next-token cross-entropy on the caption tokens only — image and prompt positions are masked out. → `outputs\projector_g12.pt` (15 MiB). |
| `caption.py <image\|video>` | Runs the whole chain and prints a caption. `--video clip` (default) mean-pools V-JEPA's video tokens over time; `--video frames` encodes sampled frames as images instead, which is closer to the training data. |

Full pipeline diagram, model architectures, hyperparameters and results: **[README_captioner.md](README_captioner.md)**.

**Runners:** `setup.ps1` builds the `.venv` (uv, correct CUDA wheel, shared FFmpeg build). `run.bat` is a double-click menu that activates the venv and lets you drag a clip in. `caption.bat` jumps straight to captioning.

## Quickstart

```bat
powershell -ExecutionPolicy Bypass -File .\setup.ps1

run.bat                                    & rem menu, drag a clip in
run.bat extract clips\archery.mp4 --model L
run.bat similarity clips
run.bat bench --sizes B L --frames 8 16
run.bat shell                              & rem cmd prompt with the venv active
```

Every script has `--help`. Shared options: `--model`, `--dtype {fp16,bf16,fp32}`, `--res`, `--frames`, `--stride` (consecutive frames every N, instead of spreading them across the whole clip).

## Which models

| Release | Date | What's used here |
|---|---|---|
| **V-JEPA 2.1** | 2026-03-16 | **Default.** ViT-B/L/g/G @384. Better dense, temporally consistent features. Weights from `dl.fbaipublicfiles.com`, code via `torch.hub` (`facebookresearch/vjepa2`, pinned commit). Not in `transformers` yet (PR #45497 still open). |
| V-JEPA 2 | 2025-06 | ViT-L/H/g @256, g @384. Used by `classify.py`: the only released SSv2 heads are for V-JEPA 2. |

No official V-JEPA release newer than 2.1 had shown up as of Sept 2026. VL-JEPA, the vision-language variant, isn't covered here.

**Defaults:** `--model L` (V-JEPA 2.1 ViT-L/16 @384, 300M), `--dtype fp16`, `--frames 16`.
Aliases: `B L g G` = 2.1 sizes, `2-L 2-H 2-g 2-g384` = V-JEPA 2.

A 16-frame 384px clip becomes 8 × 24 × 24 = **4608 tokens** (tubelet 2×16×16). An image becomes 1 × 24 × 24 = 576 tokens. 2.1 has its own image patch embedding; V-JEPA 2 doesn't, so an image is fed as a 2-frame clip.

## What came out of it (RTX 5060 Laptop, 8 GB)

| Step | Time | Peak GPU memory |
|---|---|---|
| Download 20k images | ~3 min | – |
| Cache V-JEPA features | 11.2 min (30 img/s) | 1.4 GiB |
| Train projector (1 epoch, 625 steps) | 27.6 min (12 samples/s) | 5.1 GiB |
| Caption one image | ~1.4 s (after ~20 s model load) | 3.5 GiB |

- Training loss 6.29 → 3.80.
- Unseen images get the gist ("a bride and her bridesmaids at a wedding") but miss specifics. Video captions stay generic.
- Two findings worth repeating: on Windows, a full card silently spills VRAM into system RAM instead of raising OOM (throughput drops ~8×, no error) — padding every caption to a fixed width stops the cache fragmentation that caused it. And `random.Random(0).sample(pool, k)` returns the same leading items for any `k`, so a small "held-out" subset drawn with the same seed isn't held out at all.

Next steps would be more data, more image tokens (`--grid 24`), or stage-2 instruction tuning where the LLM trains too.

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
