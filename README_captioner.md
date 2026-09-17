# V-JEPA → Qwen2.5 captioner

A small image/video captioning model built with the LLaVA / PerceptionLM **stage 1** recipe. The vision encoder and the LLM stay frozen; only a small MLP projector is trained to translate video features into word embeddings.

## How the pipeline works

```
image / video
  │  resize short side to 438, center-crop 384×384, ImageNet normalize
  ▼
V-JEPA 2.1 ViT-L (frozen, fp16)      image → 576 tokens × 1024     16-frame clip → 4608 tokens × 1024
  │  average over time (videos), 2×2 average pool 24×24 → 12×12
  ▼
144 tokens × 1024
  │
  ▼
MLP projector (trained)              Linear 1024→1536 → GELU → Linear 1536→1536
  │
  ▼
144 tokens × 1536, placed inside Qwen's chat prompt:
  <|im_start|>system …<|im_end|> <|im_start|>user [144 image tokens] Describe this image.<|im_end|> <|im_start|>assistant
  ▼
Qwen2.5-1.5B-Instruct (frozen, fp16) → caption (greedy decoding, up to 60 tokens)
```

It runs in three steps:

1. **`llava_prep.py`** picks a seeded random 20k subset of LLaVA-Pretrain. It downloads only those JPEGs, using HTTP range requests into the 25.5 GiB `images.zip`. It then runs V-JEPA once per image and caches the pooled features to `data\llava20k\feats_vjepa2.1-vitl-384_g12.npy` (float16, 20000 × 144 × 1024, 5.5 GiB). Training never runs the encoder.
2. **`train_projector.py`** trains only the projector on the cached features. The loss is next-token cross-entropy on the caption tokens (plus the closing `<|im_end|>`); image and prompt positions get no loss. Captions are padded to one fixed width (48) so tensor shapes, and VRAM use, stay constant. Output: `outputs\projector_g12.pt`.
3. **`caption.py <image or video>`** runs the full chain above. For videos, `--video clip` (default) averages V-JEPA's video tokens over time. `--video frames` encodes each sampled frame as an image instead, which is closer to the training data.

Shared code lives in `vlm_common.py`. The encoder loader and preprocessing are reused from `jepa_common.py`. To run: `run.bat` menu **D / T / C**, or double-click `caption.bat`.

## Models

| Role | Model | Size | Precision | Trained here? | License |
|---|---|---|---|---|---|
| Vision encoder | V-JEPA 2.1 ViT-L/16 @384 (Meta FAIR) | ~300M (305M counted) | fp16 | No, frozen | MIT |
| Projector | 2-layer MLP (LLaVA-1.5 "mlp2x_gelu" shape) | 3.94M | fp32 weights, fp16 compute | **Yes** | – |
| Language model | Qwen2.5-1.5B-Instruct (Alibaba Qwen) | 1.54B | fp16 | No, frozen | Apache 2.0 |

**V-JEPA 2.1 ViT-L/16** (released 2026-03-16; checkpoint `vjepa2_1_vitl_dist_vitG_384.pt`, EMA encoder weights)
- Self-supervised video model: it learns by predicting the features of masked parts of images and videos, with no text labels. That is why it needs a trained projector to "talk" to an LLM.
- The 2.1 recipe adds a dense predictive loss on all tokens, deep self-supervision on intermediate layers, and separate tokenizers for images and videos. The ViT-L checkpoint is distilled from the 2B ViT-G.
- Architecture: 24 blocks, width 1024, 16 heads, MLP 4096, GELU, 3D rotary position embeddings, final LayerNorm.
- Patches: 16×16 pixels. Images use their own patch embedding (1 frame); videos use 2-frame tubelets. That gives a 24×24 token grid per image and 8×24×24 for a 16-frame clip.
- Code comes from `facebookresearch/vjepa2` via `torch.hub` (pinned commit); weights come from `dl.fbaipublicfiles.com`.

**Qwen2.5-1.5B-Instruct** (Qwen2.5 series released 2024-09-19)
- Dense decoder-only transformer, pretrained on up to 18T tokens, then instruction-tuned.
- Architecture: 28 layers, hidden size 1536, 12 query / 2 key-value heads (grouped-query attention), SwiGLU MLP 8960, RMSNorm, rotary position embeddings, 151,936-token vocabulary, tied input/output embeddings.
- Uses the ChatML prompt format (`<|im_start|>` / `<|im_end|>`). Loaded through Hugging Face `transformers` (~3 GB download on first use).

**Projector**
- `Linear(1024→1536) → GELU → Linear(1536→1536)`, 3,935,232 parameters.
- Saved with its config (encoder name, grid, LLM id, prompt) in `projector_g12.pt` (15 MiB).

## Training data and settings

- **Data:** [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain), the "LAION/CC/SBU BLIP-Caption Concept-balanced 558K" set LLaVA uses for stage 1. It has 558,128 web images with short synthetic BLIP captions. Here: a random 20,000 of them (seed 0, 948 MiB of JPEGs). Check the dataset card for usage terms.
- **Settings:** 1 epoch = 625 steps. Batch 32 (micro-batch 4 × gradient accumulation 8). AdamW, lr 1e-3, no weight decay, 19 warmup steps then cosine decay, gradient clipping 1.0, fp16 autocast with loss scaling.

## Results on this PC (RTX 5060 Laptop GPU, 8 GB)

| Step | Time | GPU memory (PyTorch peak) |
|---|---|---|
| Download 20k images | ~3 min | – |
| Cache V-JEPA features | 11.2 min (30 images/s) | 1.4 GiB |
| Train projector | 27.6 min (12 samples/s) | 5.1 GiB allocated / 5.4 GiB reserved |
| Caption one image | ~1.4 s after ~20 s model load | 3.5 GiB |

- Training loss went from 6.29 to 3.80.
- On unseen images, captions get the gist ("a bride and her bridesmaids at a wedding") but often miss specifics. Video captions are generic.
- This is expected for stage 1, which only aligns the two models. Better captions would need more data, more tokens (`--grid 24`), or stage 2 instruction tuning, where the LLM is also trained.

Software: Windows, Python 3.12, torch 2.14.0+cu130, transformers 5.17.0.
