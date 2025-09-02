#!/usr/bin/env python3
# PyAiModel-TFormer-BPERoPESwiGLU — generator.py
# Train LLaMA-style Transformer (RMSNorm + RoPE + SwiGLU, no bias) with SentencePiece
# and export a fully compliant GGUF v3 using the official `gguf` library.
# Plus optional post-export quantization via llama.cpp's llama-quantize.
# Author: Artur Strazewicz — 2025 — MIT

import argparse, json, os, math, time, shutil, subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sentencepiece as spm
from sentencepiece import SentencePieceProcessor
import gguf

# ----------------------------
# Config / Model (LLaMA-like)
# ----------------------------
@dataclass
class LlamaConfig:
    vocab_size: int
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 8
    n_embd: int = 512
    block_size: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rms_eps: float = 1e-5
    dtype: str = "float16"  # export dtype: float16|float32 (bfloat16 -> cast to f32)

def _head_dim(cfg: LlamaConfig) -> int:
    return cfg.n_embd // cfg.n_head

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        x2 = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(x2 + self.eps)
        return (self.weight * x).type_as(self.weight)

class RoPE:
    def __init__(self, dim: int, base: float = 10000.0):
        self.dim = dim
        self.base = base
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv
    def cos_sin(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
    @staticmethod
    def apply(x, cos, sin):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        xr = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x * cos + xr * sin

class LlamaAttention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        d = cfg.n_embd
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_kv_head
        self.head_dim = d // cfg.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = RoPE(self.head_dim, base=cfg.rope_theta)
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        cos, sin = self.rope.cos_sin(T, x.device)
        q = RoPE.apply(q, cos, sin)
        k = RoPE.apply(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        y = self.drop(y)
        return y

class LlamaMLP(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        d = cfg.n_embd
        hidden = int(4 * d * 2 // 3)  # SwiGLU sizing
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))

class LlamaBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, eps=cfg.rms_eps)
        self.attn = LlamaAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, eps=cfg.rms_eps)
        self.mlp = LlamaMLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x

class LlamaModel(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # NOT tied
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ----------------------------
# Tokenizer / Dataset
# ----------------------------
def train_sentencepiece(corpus: Path, out_dir: Path, vocab_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "tokenizer"
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=-1,
        byte_fallback=True,
        normalization_rule_name="nmt_nfkc_cf",
    )
    return prefix.with_suffix(".model")

def encode_spm(sp: SentencePieceProcessor, text: str):
    ids = [sp.bos_id()] + sp.encode(text) + [sp.eos_id()]
    return ids

class TextDataset(Dataset):
    def __init__(self, ids: list[int], block: int):
        self.ids = ids
        self.block = block
    def __len__(self): return max(0, len(self.ids) - self.block - 1)
    def __getitem__(self, i):
        x = torch.tensor(self.ids[i:i + self.block], dtype=torch.long)
        y = torch.tensor(self.ids[i + 1:i + self.block + 1], dtype=torch.long)
        return x, y

# ----------------------------
# GGUF export (official)
# ----------------------------
def _size_label(n: int) -> str:
    if n >= 1_000_000_000:
        v = n / 1e9;  s = "B"
    elif n >= 1_000_000:
        v = n / 1e6;  s = "M"
    elif n >= 1_000:
        v = n / 1e3;  s = "K"
    else:
        return str(n)
    x = f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{x}{s}"

def _torch_to_export_np(t: torch.Tensor, want: str):
    # We only emit float16 or float32 tensors for maximum compatibility.
    if want == "float16":
        return t.detach().to(torch.float16).contiguous().cpu().numpy()
    else:
        return t.detach().to(torch.float32).contiguous().cpu().numpy()

def export_to_gguf(model: LlamaModel, cfg: LlamaConfig, spm_path: Path, out_path: Path, run_name: str):
    writer = gguf.GGUFWriter(str(out_path), arch="llama")

    # Core model metadata
    writer.add_name(run_name)
    writer.add_context_length(cfg.block_size)
    writer.add_embedding_length(cfg.n_embd)
    writer.add_block_count(cfg.n_layer)
    writer.add_feed_forward_length(int(4 * cfg.n_embd * 2 // 3))
    writer.add_head_count(cfg.n_head)
    writer.add_head_count_kv(cfg.n_kv_head)
    writer.add_rope_dimension_count(_head_dim(cfg))
    writer.add_rope_freq_base(float(cfg.rope_theta))

    # Extra llama.* keys
    head_dim = cfg.n_embd // cfg.n_head
    writer.add_uint32("llama.attention.key_length", head_dim)
    writer.add_uint32("llama.attention.value_length", head_dim)
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", float(cfg.rms_eps))

    # --- TOKENIZER + BASIC METADATA (LM Studio–safe minimal) ---
    sp = SentencePieceProcessor(model_file=str(spm_path))
    n_sp = sp.vocab_size()
    assert cfg.vocab_size == n_sp, f"vocab mismatch: cfg={cfg.vocab_size} vs spm={n_sp}"

    tokens = [sp.id_to_piece(i) for i in range(n_sp)]
    writer.add_tokenizer_model("llama")   # important: SPM/unigram is "llama"
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)

    _bos = sp.bos_id();  _eos = sp.eos_id();  _unk = sp.unk_id()
    writer.add_bos_token_id(_bos if _bos >= 0 else 1)
    writer.add_eos_token_id(_eos if _eos >= 0 else 2)
    writer.add_unk_token_id(_unk if _unk >= 0 else 0)

    writer.add_uint32("llama.vocab_size", n_sp)

    # FILE TYPE ↔ DTYPE (do not mutate cfg)
    out_dtype = "float16" if cfg.dtype == "float16" else "float32"
    writer.add_uint32("general.file_type", 1 if out_dtype == "float16" else 0)

    # Show params in LM Studio
    total_params = sum(p.numel() for p in model.parameters())
    writer.add_uint64("general.parameter_count", int(total_params))
    writer.add_string("general.size_label", _size_label(total_params))

    # === TENSORS ===
    sd = model.state_dict()

    def add(name: str, tensor: torch.Tensor, force_f32: bool = False):
        arr = _torch_to_export_np(tensor, "float32" if force_f32 else out_dtype)
        writer.add_tensor(name, arr)

    # Embeddings / output
    add("token_embd.weight", sd["tok_emb.weight"])
    add("output_norm.weight", sd["norm.weight"], force_f32=True)  # <-- FP32
    add("output.weight", sd["lm_head.weight"])

    # Blocks
    for i in range(cfg.n_layer):
        p = f"blocks.{i}."
        add(f"blk.{i}.attn_norm.weight", sd[p + "attn_norm.weight"], force_f32=True)  # <-- FP32
        add(f"blk.{i}.attn_q.weight", sd[p + "attn.q_proj.weight"])
        add(f"blk.{i}.attn_k.weight", sd[p + "attn.k_proj.weight"])
        add(f"blk.{i}.attn_v.weight", sd[p + "attn.v_proj.weight"])
        add(f"blk.{i}.attn_output.weight", sd[p + "attn.o_proj.weight"])
        add(f"blk.{i}.ffn_norm.weight", sd[p + "ffn_norm.weight"], force_f32=True)  # <-- FP32
        add(f"blk.{i}.ffn_gate.weight", sd[p + "mlp.gate_proj.weight"])
        add(f"blk.{i}.ffn_up.weight", sd[p + "mlp.up_proj.weight"])
        add(f"blk.{i}.ffn_down.weight", sd[p + "mlp.down_proj.weight"])

    # Write file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    return str(out_path)

# ----------------------------
# Post-export quantization (llama.cpp)
# ----------------------------
def _find_llama_quantize(custom_path: str | None = None) -> str | None:
    exe = "llama-quantize.exe" if os.name == "nt" else "llama-quantize"
    cands = []
    if custom_path:
        cands.append(custom_path)
    if os.environ.get("LLAMA_QUANTIZE"):
        cands.append(os.environ["LLAMA_QUANTIZE"])
    cands += [
        str(Path("llama.cpp") / "build" / "bin" / exe),
        str(Path("llama.cpp") / "bin" / exe),
        exe,  # PATH
    ]
    for c in cands:
        w = shutil.which(c)
        if w:
            return w
        if Path(c).exists():
            return str(Path(c).resolve())
    return None

def quantize_gguf(in_gguf: str, qtype: str = "Q3_K_L",
                  out_gguf: str | None = None,
                  quant_bin: str | None = None,
                  allow_requantize: bool = False) -> str:
    if out_gguf is None:
        p = Path(in_gguf)
        out_gguf = str(p.with_name(p.stem + f".{qtype}.gguf"))

    bin_path = _find_llama_quantize(quant_bin)
    if not bin_path:
        raise FileNotFoundError(
            "llama-quantize not found. Provide --quant-bin or set LLAMA_QUANTIZE env var."
        )

    cmd = [bin_path]
    if allow_requantize:
        cmd.append("--allow-requantize")
    cmd += [str(in_gguf), str(out_gguf), qtype]

    run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"llama-quantize failed ({run.returncode}):\n{run.stdout}")
    return out_gguf

# ----------------------------
# Train loop
# ----------------------------
def train_once(run_name: str, data_path: str, models_dir: str,
               vocab_size: int, block_size: int,
               n_layer: int, n_head: int, n_embd: int,
               lr: float, batch_size: int, epochs: int,
               weight_dtype: str = "float16", amp: bool = True,
               progress_cb=None,
               quant: str | None = None,
               quant_bin: str | None = None,
               keep_fp16: bool = True) -> str:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if progress_cb:
        progress_cb(json.dumps({
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_name0": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "torch": torch.__version__,
        }))

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    corpus = Path(data_path)
    assert corpus.is_file(), f"Missing {data_path}"

    # 1) SentencePiece (train tokenizer for requested vocab_size)
    run_dir = models_dir / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    spm_path = train_sentencepiece(corpus, run_dir, vocab_size)
    sp = SentencePieceProcessor(model_file=str(spm_path))
    n_sp = sp.vocab_size()

    # 2) Encode corpus
    text = corpus.read_text(encoding="utf-8", errors="ignore")
    ids = encode_spm(sp, text)
    ids = torch.tensor(ids, dtype=torch.long)

    # split
    val_split = 0.01
    n = len(ids)
    train_len = max(0, int(n * (1.0 - val_split)))
    train_ids, val_ids = ids[:train_len], ids[train_len:] if n - train_len > block_size + 1 else ids[:]
    train_ds = TextDataset(train_ids.tolist(), block_size)
    val_ds = TextDataset(val_ids.tolist(), block_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # 3) Model
    cfg = LlamaConfig(
        vocab_size=n_sp,
        n_layer=n_layer, n_head=n_head, n_kv_head=n_head,
        n_embd=n_embd, block_size=block_size, dtype=weight_dtype
    )
    model = LlamaModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and amp))

    # log params once
    total_params = sum(p.numel() for p in model.parameters())
    if progress_cb:
        progress_cb(f"PARAMS {total_params} ({_size_label(total_params)}) | "
                    f"d_model={cfg.n_embd}, heads={cfg.n_head}, layers={cfg.n_layer}, "
                    f"ctx={cfg.block_size}, vocab={cfg.vocab_size}")
        progress_cb(f"Training started (device={device}, AMP={scaler.is_enabled()})")

    total_steps = max(1, epochs * len(train_loader))
    start = time.time()
    global_step = 0
    best_val = float("inf")

    for ep in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {ep + 1}/{epochs}")
        for xb, yb in pbar:
            t0 = time.time()
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                _, loss = model(xb, yb)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            global_step += 1
            pct = 100.0 * global_step / total_steps
            step_time = time.time() - t0
            elapsed = (time.time() - start) / 60.0
            eta = (elapsed / max(1e-9, pct / 100.0)) - elapsed if pct > 0 else 0.0
            if progress_cb:
                progress_cb(
                    f"Progress:  {pct:.2f}% | epoch {ep + 1}/{epochs} | step {global_step % len(train_loader)}/{len(train_loader)} | "
                    f"loss {loss.item():.4f} | s_it {step_time:.2f}s/it | elapsed {elapsed:.2f}m | ETA {eta:.2f}m")

        # val
        model.eval()
        with torch.no_grad():
            vloss = []
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                _, vl = model(xb, yb)
                vloss.append(vl.item())
        if vloss:
            mv = float(np.mean(vloss))
            best_val = min(best_val, mv)
            if progress_cb: progress_cb(f"Validation loss: {mv:.4f}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_file = models_dir / f"{run_name}-{stamp}.gguf"

    # 4) Export GGUF
    saved = export_to_gguf(model, cfg, spm_path, out_file, run_name=run_name)

    # copy tokenizer next to gguf (optional for downstream tools)
    try:
        shutil.copy2(spm_path, out_file.with_suffix(".tokenizer.model"))
    except Exception:
        pass

    # 5) Optional quantization
    final_path = saved
    if quant:
        if progress_cb: progress_cb(f"Quantizing to {quant} ...")
        q_out = quantize_gguf(saved, qtype=quant, quant_bin=quant_bin)
        final_path = q_out
        if not keep_fp16:
            try:
                os.remove(saved)
                if progress_cb: progress_cb(f"Removed FP16/FP32: {saved}")
            except Exception as _e:
                if progress_cb: progress_cb(f"WARN: cannot remove {saved}: {_e}")

    if progress_cb:
        progress_cb(f"Saved weights: {final_path}")
        progress_cb("DONE")
    return final_path

# ----------------------------
# CLI
# ----------------------------
def cli_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="run")
    ap.add_argument("--data", type=str, default="Datasets/dataset.txt")
    ap.add_argument("--models", type=str, default="Models")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--embd", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--quant", type=str, default=None, help="e.g. Q3_K_L, Q4_K_M, Q5_K_S, Q8_0, IQ4_NL ...")
    ap.add_argument("--quant-bin", type=str, default=None, help="path to llama-quantize if not in PATH")
    ap.add_argument("--no-keep-fp16", action="store_true", help="remove original FP16/FP32 after quantization")
    args = ap.parse_args()

    def _print(x): print(x, flush=True)

    # Map bfloat16 -> float32 for GGUF export compatibility
    dtype = args.dtype if args.dtype in ("float16", "float32") else "float32"

    train_once(
        run_name=args.run, data_path=args.data, models_dir=args.models,
        vocab_size=args.vocab, block_size=args.block,
        n_layer=args.layers, n_head=args.heads, n_embd=args.embd,
        lr=args.lr, batch_size=args.batch, epochs=args.epochs,
        weight_dtype=dtype, amp=True, progress_cb=_print,
        quant=args.quant, quant_bin=args.quant_bin, keep_fp16=(not args.no_keep_fp16)
    )

if __name__ == "__main__":
    cli_main()
