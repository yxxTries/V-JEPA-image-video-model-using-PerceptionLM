"""vlm_common.py -- shared pieces of the V-JEPA -> Qwen2.5 captioner (llava_prep, train_projector, caption).

Recipe = LLaVA / PerceptionLM "stage 1": the V-JEPA encoder and the LLM stay frozen; only a 2-layer MLP
projector learns to map V-JEPA tokens into the LLM's input-embedding space.

What the LLM sees (Qwen chat template, image tokens at the start of the user turn):
  <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n [N image tokens] \nDescribe this image.<|im_end|>\n
  <|im_start|>assistant\n caption<|im_end|>          <- loss only on these caption tokens
"""
import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Describe this image."
DATA = Path(os.environ.get("JEPA_DATA", HERE / "data" / "llava20k"))  # images, samples.json, feature cache
IGNORE = -100  # label value cross_entropy skips (system/prompt/image positions)
warnings.filterwarnings("ignore", message=".*repetition_penalty.*")  # Qwen's 1.1 penalty then covers new tokens only


class Projector(nn.Module):
    """Linear(1024 -> 1536) -> GELU -> Linear(1536 -> 1536), i.e. LLaVA-1.5's mlp2x_gelu. The only trained part."""

    def __init__(self, d_vision: int = 1024, d_llm: int = 1536):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(d_vision, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm)

    def forward(self, x):  # [B, N, d_vision] -> [B, N, d_llm]
        return self.fc2(self.act(self.fc1(x)))


def pool_tokens(tokens: torch.Tensor, t: int, h: int, w: int, grid: int) -> torch.Tensor:
    """V-JEPA tokens [B, t*h*w, D] -> [B, grid*grid, D]: mean over time (videos), then average-pool h x w to grid x grid."""
    B, _, D = tokens.shape
    x = tokens.reshape(B, t, h, w, D).mean(1).permute(0, 3, 1, 2)  # [B, D, h, w]
    if (h, w) != (grid, grid):
        x = F.adaptive_avg_pool2d(x, grid)  # 24x24 -> 12x12 is an exact 2x2 mean
    return x.flatten(2).transpose(1, 2)  # [B, grid*grid, D]


def load_llm(llm_id: str, device, dtype):
    """Tokenizer + causal LM with frozen weights in `dtype`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(llm_id)
    llm = AutoModelForCausalLM.from_pretrained(llm_id, dtype=dtype, attn_implementation="sdpa")
    return tok, llm.to(device).eval().requires_grad_(False)


def prompt_ids(tok, prompt: str = PROMPT) -> tuple[list[int], list[int]]:
    """Chat-template token ids that go before and after the image tokens."""
    text = tok.apply_chat_template([{"role": "user", "content": "<image>\n" + prompt}],
                                   tokenize=False, add_generation_prompt=True)
    before, after = text.split("<image>")
    return tok(before, add_special_tokens=False).input_ids, tok(after, add_special_tokens=False).input_ids


def build_inputs(llm, pre: list[int], post: list[int], img: torch.Tensor, targets: torch.Tensor | None = None):
    """[pre] + img [B, N, d] + [post] (+ caption ids when training) -> inputs_embeds [B, L, d].

    targets: [B, C] caption ids right-padded with IGNORE; they fill the last C positions. Padding only ever sits
    after every real token, so a causal LM never attends to it and no attention mask is needed.
    """
    emb = llm.get_input_embeddings()
    B, dev = img.shape[0], img.device
    parts = [emb(torch.tensor(pre, device=dev)).expand(B, -1, -1), img.to(emb.weight.dtype),
             emb(torch.tensor(post, device=dev)).expand(B, -1, -1)]
    if targets is not None:
        parts.append(emb(targets.clamp(min=0)))  # pad positions embed token 0; the loss ignores them
    return torch.cat(parts, 1)


def caption_loss(llm, embeds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy on the caption only (image + prompt positions get no loss).

    targets [B, C] occupy the last C input positions; the hidden state one position earlier predicts each of them.
    Fixed C across steps keeps every tensor shape identical, which stops CUDA cache fragmentation on 8 GB cards.
    """
    C = targets.shape[1]
    h = llm.model(inputs_embeds=embeds, use_cache=False).last_hidden_state[:, -C - 1:-1]  # [B, C, d]
    logits = llm.lm_head(h).float()  # [B, C, vocab]
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE)


def generate(llm, tok, embeds: torch.Tensor, max_new_tokens: int = 60) -> torch.Tensor:
    """Greedy decoding from input embeddings. Returns only the new token ids [B, G]."""
    mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
    return llm.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                        temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)


def latest_projector() -> Path:
    ckpts = sorted(OUT.glob("projector_*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise SystemExit("No outputs\\projector_*.pt yet -- run llava_prep.py, then train_projector.py.")
    return ckpts[-1]
