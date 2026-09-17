# JEPA bench

A small image/video captioning model built with the LLaVA / PerceptionLM recipe. The vision encoder and the LLM stay frozen; only a small MLP projector is trained to translate video features into word embeddings.

## Which models (as of Sept 2026)

| Release | Date | What's used here |
|---|---|---|
| **V-JEPA 2.1** | 2026-03-16 | **Default.** ViT-B/L/g/G @384. Better dense, temporally consistent features. Weights from `dl.fbaipublicfiles.com`, code via `torch.hub` (`facebookresearch/vjepa2`, pinned commit). Not in `transformers` yet (PR #45497 still open). |
| V-JEPA 2 | 2025-06 | ViT-L/H/g @256, g @384. Used by `classify.py`: the only released SSv2 heads are for V-JEPA 2 (`facebook/vjepa2-vitl-fpc16-256-ssv2`). |

No official V-JEPA release newer than 2.1 had shown up as of Sept 2026. VL-JEPA, the vision-language variant, isn't covered here.

**Defaults:** `--model L` (V-JEPA 2.1 ViT-L/16 @384, 300M), `--dtype fp16`, `--frames 16`.
Aliases: `B L g G` = 2.1 sizes, `2-L 2-H 2-g 2-g384` = V-JEPA 2.

A 16-frame 384px clip becomes 8 x 24 x 24 = **4608 tokens** (tubelet 2x16x16). An image becomes 1 x 24 x 24 = 576 tokens. 2.1 has its own image patch embedding. V-JEPA 2 doesn't, so the image is used as a 2-frame clip.


```

### run.bat 

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
