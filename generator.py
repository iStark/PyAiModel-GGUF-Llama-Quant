#!/usr/bin/env python3
# PyAiModel-TFormer-BPERoPESwiGLU — generator.py (Llama 3 Instruct, BOTH default, auto JSONL)
# Train LLaMA-style Transformer (RMSNorm + RoPE + SwiGLU, optional GQA) with SentencePiece (BPE mode)
# and export a fully compliant GGUF v3 using the official `gguf` library.
# Adds SFT with Llama 3 chat format (<|start_header_id|>/<|end_header_id|> ... <|eot_id|>) and permissive tokenizer for Unicode/bytes.
# Default task: BOTH (LM + SFT) with auto JSONL discovery in Datasets/.
# Author: Artur Strazewicz — 2025 — MIT

import argparse, json, os, math, time, shutil, subprocess, re
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
    n_kv_head: int = 8  # GQA: number of KV heads (<= n_head). If == n_head, it's MHA.
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
        x1 = x[..., ::2]; x2 = x[..., 1::2]
        xr = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x * cos + xr * sin

class LlamaAttention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        d = cfg.n_embd
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_kv_head
        assert self.n_head % self.n_kv == 0, "n_head must be divisible by n_kv_head for GQA"
        self.groups = self.n_head // self.n_kv
        self.head_dim = d // cfg.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = RoPE(self.head_dim, base=cfg.rope_theta)
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv,  self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv,  self.head_dim).transpose(1, 2)
        cos, sin = self.rope.cos_sin(T, x.device)
        q = RoPE.apply(q, cos, sin); k = RoPE.apply(k, cos, sin)
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
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
        self.up_proj   = nn.Linear(d, hidden, bias=False)
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
# Sanitize: удаление управляющих ASCII (кроме \n и \t)
# ----------------------------
_CLEAN_CC = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"               # ASCII control (без \n,\t)
    r"|[\uD800-\uDFFF]"                               # суррогаты UTF-16
    r"|[\uFDD0-\uFDEF]"                               # noncharacters U+FDD0..FDEF
    r"|[\uFFFE\uFFFF]"                                # noncharacters U+FFFE/FFFF
, flags=re.UNICODE)

def sanitize_text(s: str) -> str:
    """Удаляет мусорные управляющие символы из строки."""
    if not s:
        return ""
    return _CLEAN_CC.sub("", s)

# ----------------------------
# Tokenizer / Datasets — Llama 3 special tokens
# ----------------------------
L3_BEGIN = "<|begin_of_text|>"
L3_END   = "<|end_of_text|>"
L3_S     = "<|start_header_id|>"
L3_E     = "<|end_header_id|>"
L3_EOT   = "<|eot_id|>"

def train_sentencepiece_bpe(corpus: Path, out_dir: Path, vocab_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "tokenizer"
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=-1,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=False,               # не подставляем пробел в начале
        hard_vocab_limit=False,               # позволяем добавить служебные/байтовые юниты
        user_defined_symbols=[L3_BEGIN, L3_END, L3_S, L3_E, L3_EOT],  # Llama 3
    )
    return prefix.with_suffix(".model")

def encode_spm(sp: SentencePieceProcessor, text: str):
    clean = sanitize_text(text)
    return [sp.bos_id()] + sp.encode(clean, out_type=int) + [sp.eos_id()]

class TextDataset(Dataset):
    def __init__(self, ids: list[int], block: int):
        self.ids = ids
        self.block = block
    def __len__(self): return max(0, len(self.ids) - self.block - 1)
    def __getitem__(self, i):
        x = torch.tensor(self.ids[i:i + self.block], dtype=torch.long)
        y = torch.tensor(self.ids[i + 1:i + self.block + 1], dtype=torch.long)
        return x, y

def load_lm_ids(sp: SentencePieceProcessor, path: Path) -> list[int]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    text = sanitize_text(text)
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    ids_list: list[int] = []
    for ex in paragraphs:
        ids_list.extend(encode_spm(sp, ex))
    return ids_list or [sp.bos_id(), sp.eos_id()]

def render_llama3(system: str | None, user: str, assistant: str | None) -> str:
    parts = [L3_BEGIN]
    if system and system.strip():
        parts.append(f"{L3_S}system{L3_E}\n\n{sanitize_text(system.strip())}{L3_EOT}")
    usr_clean = sanitize_text(user.strip())
    parts.append(f"{L3_S}user{L3_E}\n\n{usr_clean}{L3_EOT}")
    if assistant is not None:
        as_clean = sanitize_text(assistant.strip())
        parts.append(f"{L3_S}assistant{L3_E}\n\n{as_clean}{L3_EOT}")
    return "".join(parts)

def load_sft_ids_llama3(sp: SentencePieceProcessor, jsonl_path: Path) -> list[int]:
    ids_list: list[int] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue
            text = ex.get("text")
            if not text:
                sys_ = ex.get("system", "")
                usr = ex.get("prompt") or ex.get("user") or ex.get("input") or ex.get("question")
                rsp = ex.get("response") or ex.get("assistant") or ex.get("answer")
                if usr is None or rsp is None:
                    msgs = ex.get("messages")
                    if isinstance(msgs, list) and msgs:
                        sys_msgs = [m.get("content","") for m in msgs if m.get("role")=="system"]
                        usr_msgs = [m.get("content","") for m in msgs if m.get("role")=="user"]
                        as_msgs  = [m.get("content","") for m in msgs if m.get("role")=="assistant"]
                        if not sys_ and sys_msgs: sys_ = sys_msgs[0]
                        if usr is None and usr_msgs: usr = usr_msgs[-1]
                        if rsp is None and as_msgs:  rsp = as_msgs[-1]
                if usr is None or rsp is None:
                    continue
                text = render_llama3(system=None, user=usr, assistant=rsp)
            else:
                text = sanitize_text(text)
            ids_list.extend(encode_spm(sp, text))
    return ids_list or [sp.bos_id(), sp.eos_id()]

# ----------------------------
# GGUF export (official)
# ----------------------------
def _size_label(n: int) -> str:
    if n >= 1_000_000_000:
        v = n / 1e9; s = "B"
    elif n >= 1_000_000:
        v = n / 1e6; s = "M"
    elif n >= 1_000:
        v = n / 1e3; s = "K"
    else:
        return str(n)
    x = f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{x}{s}"

def _torch_to_export_np(t: torch.Tensor, want: str):
    if want == "float16":
        return t.detach().to(torch.float16).contiguous().cpu().numpy()
    else:
        return t.detach().to(torch.float32).contiguous().cpu().numpy()

def _add_u32_list(writer: "gguf.GGUFWriter", key: str, values):
    vals = [int(x) for x in values]
    if hasattr(writer, "add_uint32_list"):
        return writer.add_uint32_list(key, vals)
    if hasattr(writer, "add_array"):
        return writer.add_array(key, vals)
    raise RuntimeError(f"Your gguf.GGUFWriter lacks add_uint32_list/add_array; cannot write key={key}")

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

    # --- TOKENIZER + BASIC METADATA ---
    sp = SentencePieceProcessor(model_file=str(spm_path))
    n_sp = sp.vocab_size()
    assert cfg.vocab_size == n_sp, f"vocab mismatch: cfg={cfg.vocab_size} vs spm={n_sp}"

    _bos = sp.bos_id()
    _eos = sp.eos_id()
    _unk = sp.unk_id()

    tokens = [sp.id_to_piece(i) for i in range(n_sp)]
    writer.add_tokenizer_model("llama")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    if hasattr(writer, "add_token_scores"):
        writer.add_token_scores([0.0] * len(tokens))

    writer.add_bos_token_id(int(_bos if _bos >= 0 else 1))
    writer.add_eos_token_id(int(_eos if _eos >= 0 else 2))
    writer.add_unk_token_id(int(_unk if _unk >= 0 else 0))

    # Stop tokens: EOS + <|eot_id|> if present
    stop_ids = []
    if _eos >= 0: stop_ids.append(int(_eos))
    try:
        eot_id = sp.piece_to_id(L3_EOT)
        if eot_id >= 0:
            stop_ids.append(int(eot_id))
    except Exception:
        pass
    if stop_ids:
        _add_u32_list(writer, "tokenizer.ggml.stop_token_ids", stop_ids)

    # FILE TYPE ↔ DTYPE
    out_dtype = "float16" if cfg.dtype == "float16" else "float32"
    writer.add_uint32("general.file_type", 1 if out_dtype == "float16" else 0)

    total_params = sum(p.numel() for p in model.parameters())
    writer.add_uint64("general.parameter_count", int(total_params))
    writer.add_string("general.size_label", _size_label(total_params))

    # === TENSORS ===
    sd = model.state_dict()
    def add(name: str, tensor: torch.Tensor, force_f32: bool = False):
        arr = _torch_to_export_np(tensor, "float32" if force_f32 else out_dtype)
        writer.add_tensor(name, arr)

    add("token_embd.weight", sd["tok_emb.weight"])
    add("output_norm.weight", sd["norm.weight"], force_f32=True)  # FP32 norms
    add("output.weight", sd["lm_head.weight"])

    for i in range(cfg.n_layer):
        p = f"blocks.{i}."
        add(f"blk.{i}.attn_norm.weight", sd[p + "attn_norm.weight"], force_f32=True)
        add(f"blk.{i}.attn_q.weight",     sd[p + "attn.q_proj.weight"])
        add(f"blk.{i}.attn_k.weight",     sd[p + "attn.k_proj.weight"])
        add(f"blk.{i}.attn_v.weight",     sd[p + "attn.v_proj.weight"])
        add(f"blk.{i}.attn_output.weight",sd[p + "attn.o_proj.weight"])
        add(f"blk.{i}.ffn_norm.weight",   sd[p + "ffn_norm.weight"], force_f32=True)
        add(f"blk.{i}.ffn_gate.weight",   sd[p + "mlp.gate_proj.weight"])
        add(f"blk.{i}.ffn_up.weight",     sd[p + "mlp.up_proj.weight"])
        add(f"blk.{i}.ffn_down.weight",   sd[p + "mlp.down_proj.weight"])

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
    if custom_path: cands.append(custom_path)
    if os.environ.get("LLAMA_QUANTIZE"): cands.append(os.environ["LLAMA_QUANTIZE"])
    cands += [str(Path("llama.cpp") / "build" / "bin" / exe),
              str(Path("llama.cpp") / "bin" / exe),
              exe]
    for c in cands:
        w = shutil.which(c)
        if w: return w
        if Path(c).exists(): return str(Path(c).resolve())
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
        raise FileNotFoundError("llama-quantize not found. Provide --quant-bin or set LLAMA_QUANTIZE env var.")
    cmd = [bin_path]
    if allow_requantize: cmd.append("--allow-requantize")
    cmd += [str(in_gguf), str(out_gguf), qtype]
    run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"llama-quantize failed ({run.returncode}):\n{run.stdout}")
    return out_gguf

# ----------------------------
# Train loop (LM / SFT / BOTH)
# ----------------------------
def build_loader(ids: list[int], block_size: int, batch_size: int, workers: int | None):
    if workers is None:
        workers = 0 if os.name == "nt" else 2
    ds = TextDataset(ids, block_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=int(workers),
        **({"persistent_workers": True, "prefetch_factor": 2} if workers > 0 else {})
    )
    return loader, len(ds)

def run_epoch(model, opt, scaler, loader, device, progress_cb, epoch_i, epoch_T, global_step, total_steps):
    model.train()
    pbar = tqdm(loader, desc=f"epoch {epoch_i}/{epoch_T}")
    for step, (xb, yb) in enumerate(pbar, 1):
        t0 = time.time()
        xb = xb.to(device); yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            _, loss = model(xb, yb)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        global_step += 1
        pct = 100.0 * global_step / max(1, total_steps)
        step_time = time.time() - t0
        if progress_cb:
            progress_cb(f"Progress:  {pct:.2f}% | epoch {epoch_i}/{epoch_T} | step {step}/{len(loader)} | loss {loss.item():.4f} | s_it {step_time:.2f}s/it")
    return global_step

def _auto_find_jsonl(base_dir: Path) -> str | None:
    """Try to find a JSONL dataset in priority order."""
    candidates = [
        "dataset_llama3_nosystem.jsonl",
        "dataset_llama3_text.jsonl",
        "dataset_chatml_nosystem.jsonl",
        "dataset_chatml_text.jsonl",
        "dataset.jsonl",
    ]
    for name in candidates:
        p = base_dir / name
        if p.exists():
            return str(p)
    # fallback: any .jsonl
    any_jsonl = sorted(base_dir.glob("*.jsonl"))
    return str(any_jsonl[0]) if any_jsonl else None

def train_once(run_name: str, data_path: str, models_dir: str,
               vocab_size: int, block_size: int,
               n_layer: int, n_head: int, n_embd: int,
               lr: float, batch_size: int, epochs: int,
               weight_dtype: str = "float16", amp: bool = True,
               progress_cb=None,
               quant: str | None = None,
               quant_bin: str | None = None,
               kv_heads: int | None = None,
               keep_fp16: bool = True,
               workers: int | None = None,
               # NEW:
               task: str = "both",                 # default BOTH
               data_jsonl: str | None = None,      # path to JSONL for SFT
               epochs_sft: int | None = None) -> str:

    """
    If task == "lm":     train on TXT (data_path) only
    If task == "sft":    train on JSONL (data_jsonl) only (Llama 3 chat)
    If task == "both":   LM first (epochs), then SFT (epochs_sft or 1)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if progress_cb:
        progress_cb(json.dumps({
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_name0": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "torch": torch.__version__,
        }))

    models_dir = Path(models_dir); models_dir.mkdir(parents=True, exist_ok=True)

    # === Tokenizer (SPM BPE) trained from TXT corpus ===
    corpus = Path(data_path)
    assert corpus.is_file(), f"Missing tokenizer corpus {data_path}"
    run_dir = models_dir / run_name
    if run_dir.exists(): shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    spm_path = train_sentencepiece_bpe(corpus, run_dir, vocab_size)
    sp = SentencePieceProcessor(model_file=str(spm_path))
    n_sp = sp.vocab_size()

    # === Resolve JSONL automatically if needed (for SFT or BOTH)
    if task in ("sft", "both") and not data_jsonl:
        auto = _auto_find_jsonl(Path("Datasets"))
        if not auto:
            raise FileNotFoundError("SFT requested but no JSONL found in Datasets/.")
        data_jsonl = auto
        if progress_cb:
            progress_cb(f'ENV {json.dumps({"auto_jsonl": data_jsonl})}')

    # === Build training ids (LM / SFT / BOTH)
    lm_ids: list[int] = []
    sft_ids: list[int] = []
    if task in ("lm", "both"):
        lm_ids = load_lm_ids(sp, corpus)
    if task in ("sft", "both"):
        sft_ids = load_sft_ids_llama3(sp, Path(data_jsonl))

    # === Model/Opt
    cfg = LlamaConfig(
        vocab_size=n_sp,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=(kv_heads if kv_heads is not None else n_head),
        n_embd=n_embd, block_size=block_size, dtype=weight_dtype
    )
    model = LlamaModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and amp))

    total_params = sum(p.numel() for p in model.parameters())
    if progress_cb:
        progress_cb(f"PARAMS {total_params} ({_size_label(total_params)}) | "
                    f"d_model={cfg.n_embd}, heads={cfg.n_head}, kv_heads={cfg.n_kv_head}, "
                    f"layers={cfg.n_layer}, ctx={cfg.block_size}, vocab={cfg.vocab_size}")
        progress_cb(f"Training started (device={device}, AMP={scaler.is_enabled()})")

    global_step = 0
    # ---- LM phase
    if task in ("lm", "both") and lm_ids:
        if progress_cb: progress_cb("PHASE:LM start")
        train_loader, _ = build_loader(lm_ids, block_size, batch_size, workers)
        total_steps = max(1, epochs * len(train_loader))
        for ep in range(1, epochs + 1):
            global_step = run_epoch(model, opt, scaler, train_loader, device, progress_cb, ep, epochs, global_step,
                                    total_steps)
        if progress_cb: progress_cb("PHASE:LM done")

    # ---- SFT phase
    if task in ("sft", "both") and sft_ids:
        if progress_cb: progress_cb(f"PHASE:SFT start (epochs={epochs_sft or 1})")
        epT = (epochs_sft if epochs_sft and epochs_sft > 0 else 1)
        train_loader_sft, _ = build_loader(sft_ids, block_size, batch_size, workers)
        total_steps = max(1, epT * len(train_loader_sft))
        for ep in range(1, epT + 1):
            global_step = run_epoch(model, opt, scaler, train_loader_sft, device, progress_cb, ep, epT, global_step,
                                    total_steps)
        if progress_cb: progress_cb("PHASE:SFT done")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_file = models_dir / f"{run_name}-{stamp}.gguf"

    # === Export GGUF
    saved = export_to_gguf(model, cfg, spm_path, out_file, run_name=run_name)

    # (optional) drop tokenizer nearby
    try:
        shutil.copy2(spm_path, out_file.with_suffix(".tokenizer.model"))
    except Exception:
        pass

    # === Optional quantization
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
    ap.add_argument("--data", type=str, default="Datasets/dataset.txt", help="TXT corpus for tokenizer + LM")
    ap.add_argument("--jsonl", type=str, default=None, help="JSONL for SFT (Llama 3 chat or fields to render)")
    ap.add_argument("--models", type=str, default="Models")
    ap.add_argument("--task", type=str, default="both", choices=["lm", "sft", "both"])  # BOTH by default
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=None, help="number of KV heads for GQA (<= n_head)")
    ap.add_argument("--embd", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2, help="LM epochs (or SFT if task=sft)")
    ap.add_argument("--epochs-sft", type=int, default=None, help="SFT epochs (only used if task=both)")
    ap.add_argument("--workers", type=int, default=None, help="DataLoader workers (override)")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--quant", type=str, default=None, help="e.g. Q3_K_L, Q4_K_M, Q5_K_S, Q8_0, IQ4_NL ...")
    ap.add_argument("--quant-bin", type=str, default=None, help="path to llama-quantize if not in PATH")
    ap.add_argument("--no-keep-fp16", action="store_true", help="remove original FP16/FP32 after quantization")
    args = ap.parse_args()

    def _print(x): print(x, flush=True)

    dtype = args.dtype if args.dtype in ("float16", "float32") else "float32"

    train_once(
        run_name=args.run,
        data_path=args.data,
        data_jsonl=args.jsonl,
        models_dir=args.models,
        vocab_size=args.vocab,
        block_size=args.block,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.embd,
        lr=args.lr,
        batch_size=args.batch,
        epochs=args.epochs,
        weight_dtype=dtype,
        amp=True,
        progress_cb=_print,
        quant=args.quant,
        quant_bin=args.quant_bin,
        kv_heads=args.kv_heads,
        keep_fp16=(not args.no_keep_fp16),
        workers=args.workers,
        task=args.task,
        epochs_sft=args.epochs_sft
    )

if __name__ == "__main__":
    cli_main()
