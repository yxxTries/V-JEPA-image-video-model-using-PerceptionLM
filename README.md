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


Sources: [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2), [HF V-JEPA 2 docs](https://huggingface.co/docs/transformers/model_doc/vjepa2), [transformers V-JEPA 2.1 issue](https://github.com/huggingface/transformers/issues/45496), [torchcodec](https://github.com/meta-pytorch/torchcodec), [PyTorch wheels](https://download.pytorch.org/whl/cu130).
