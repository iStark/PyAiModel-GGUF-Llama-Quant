#!/usr/bin/env python3
# PyAiModel-TFormer-BPERoPESwiGLU — chat.py
# Flask chat UI: loads our GGUF via `gguf`, reads sibling SentencePiece tokenizer,
# builds the same LLaMA-like PyTorch model and streams generation.

import os, json, math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, Response, render_template_string, stream_with_context

import gguf
from sentencepiece import SentencePieceProcessor

# ---------- Model (same as generator) ----------
@dataclass
class LlamaConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    block_size: int
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rms_eps: float = 1e-5

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
        self.inv = inv
    def cos_sin(self, T, device):
        t = torch.arange(T, device=device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None,None,:,:], emb.sin()[None,None,:,:]
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
        self.head_dim = d // cfg.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.rope = RoPE(self.head_dim, base=cfg.rope_theta)
    def forward(self, x):
        B,T,C = x.shape
        q = self.q_proj(x).view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B,T,self.n_head,self.head_dim).transpose(1,2)
        cos, sin = self.rope.cos_sin(T, x.device)
        q = RoPE.apply(q, cos, sin)
        k = RoPE.apply(k, cos, sin)
        att = (q @ k.transpose(-2,-1)) * self.scale
        mask = torch.tril(torch.ones(T,T, device=x.device, dtype=torch.bool)).view(1,1,T,T)
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.o_proj(y)
        return y

class LlamaMLP(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        d = cfg.n_embd
        hidden = int(4 * d * 2 // 3)
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj   = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

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
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
    @torch.no_grad()
    def forward(self, idx):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return self.lm_head(x)

# ---------- GGUF load (metadata only) ----------
def read_cfg_from_gguf(path: str) -> dict:
    r = gguf.GGUFReader(path)
    kv = {k.key: k.value for k in r.get_kv_data()}
    return {
        "vocab_size": int(kv["tokenizer.ggml.tokens_count"]) if "tokenizer.ggml.tokens_count" in kv else int(kv.get("vocab_size", 0)),
        "block_size": int(kv.get("context_length")),
        "n_embd": int(kv.get("embedding_length")),
        "n_layer": int(kv.get("block_count")),
        "n_head": int(kv.get("attention.head_count")),
        "n_kv_head": int(kv.get("attention.head_count_kv", kv.get("attention.head_count", 0))),
        "rope_theta": float(kv.get("rope.freq_base", 10000.0)),
    }

def load_torch_from_gguf(path: str, device: str = "cpu") -> tuple[LlamaModel, dict]:
    meta = read_cfg_from_gguf(path)
    cfg = LlamaConfig(
        vocab_size=meta["vocab_size"], n_layer=meta["n_layer"], n_head=meta["n_head"],
        n_kv_head=meta["n_kv_head"], n_embd=meta["n_embd"], block_size=meta["block_size"],
        rope_theta=meta["rope_theta"]
    )
    model = LlamaModel(cfg).to(device)
    # Map GGUF tensor names -> model.state_dict() keys (наш экспорт соответствует)
    name_map = {}
    name_map["token_embd.weight"] = "tok_emb.weight"
    name_map["output_norm.weight"] = "norm.weight"
    name_map["output.weight"] = "lm_head.weight"
    for i in range(cfg.n_layer):
        name_map[f"blk.{i}.attn_norm.weight"]   = f"blocks.{i}.attn_norm.weight"
        name_map[f"blk.{i}.attn_q.weight"]      = f"blocks.{i}.attn.q_proj.weight"
        name_map[f"blk.{i}.attn_k.weight"]      = f"blocks.{i}.attn.k_proj.weight"
        name_map[f"blk.{i}.attn_v.weight"]      = f"blocks.{i}.attn.v_proj.weight"
        name_map[f"blk.{i}.attn_output.weight"] = f"blocks.{i}.attn.o_proj.weight"
        name_map[f"blk.{i}.ffn_norm.weight"]    = f"blocks.{i}.ffn_norm.weight"
        name_map[f"blk.{i}.ffn_gate.weight"]    = f"blocks.{i}.mlp.gate_proj.weight"
        name_map[f"blk.{i}.ffn_up.weight"]      = f"blocks.{i}.mlp.up_proj.weight"
        name_map[f"blk.{i}.ffn_down.weight"]    = f"blocks.{i}.mlp.down_proj.weight"

    # load tensors
    reader = gguf.GGUFReader(path)
    tensors = {t.name: t for t in reader.tensors}
    sd = model.state_dict()
    with torch.no_grad():
        for gguf_name, pt_name in name_map.items():
            tinfo = tensors.get(gguf_name, None)
            if tinfo is None:
                raise KeyError(f"Missing tensor in GGUF: {gguf_name}")
            arr = tinfo.data()  # numpy memmap
            t = torch.from_numpy(np.array(arr, copy=False))
            if "uint16" in str(t.dtype):
                t = t.view(torch.bfloat16)
            # match dtypes
            t = t.to(sd[pt_name].dtype)
            if tuple(t.shape) != tuple(sd[pt_name].shape):
                raise ValueError(f"Shape mismatch for {gguf_name}: {tuple(t.shape)} vs {tuple(sd[pt_name].shape)}")
            sd[pt_name].copy_(t, non_blocking=True)
    return model, meta

# ---------- Sampling ----------
def top_k_logits(logits: torch.Tensor, k: int):
    if k <= 0: return logits
    v, ix = torch.topk(logits, k)
    out = torch.full_like(logits, float('-inf'))
    out.scatter_(1, ix, v)
    return out

@torch.no_grad()
def generate_stream(model: LlamaModel, sp: SentencePieceProcessor, device: str,
                    prompt: str, max_new: int=200, temperature: float=0.9, top_k: int=40):
    x = torch.tensor([sp.bos_id()] + sp.encode(prompt), dtype=torch.long, device=device)[None, ...]
    sent = ""
    for _ in range(max_new):
        x_cond = x if x.size(1) <= model.cfg.block_size else x[:, -model.cfg.block_size:]
        logits = model(x_cond)[:, -1, :]
        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1)
        else:
            logits = logits / temperature
            if top_k > 0:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(1)
        nid = int(next_id.item())
        x = torch.cat([x, next_id[:, None]], dim=1)
        if nid == sp.eos_id(): break
        txt = sp.decode(x[0].tolist())
        delta = txt[len(sent):]
        if delta:
            sent = txt
            yield delta

# ---------- Flask UI ----------
HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Transformer Chat</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Arial;background:#f7f7fb;margin:0}
.wrap{max-width:960px;margin:0 auto;padding:16px;display:flex;flex-direction:column;height:100vh}
.card{background:#fff;border:1px solid #eee;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
label{display:inline-block;margin:6px 12px 6px 0}
input,select,textarea{padding:8px;border:1px solid #ddd;border-radius:8px}
button{padding:10px 14px;border-radius:10px;border:1px solid #4f46e5;background:#4f46e5;color:#fff;cursor:pointer}
.row{display:flex;gap:8px;align-items:center}
.grow{flex:1}
.chat{flex:1;overflow-y:auto;background:#fff;border:1px solid #eee;border-radius:12px;padding:12px}
.msg{background:#fafafe;border:1px solid #eee;border-radius:10px;padding:10px 12px;margin:8px 0}
.msg.ai{border-left:4px solid #4f46e5}
.msg.user{border-left:4px solid #9ca3af}
.small{color:#666;font-size:12px;margin-left:4px}
:root {--mut:#556;}
footer{margin:0 auto;margin-bottom:2rem;font-size:.9em;color:var(--mut)}
footer div a{color:inherit}.links a{margin-right:.75rem}
</style></head>
<body><div class="wrap">
  <div class="card">
    <div class="row">
      <label>Model
        <select id="model">{% for m in models %}<option value="{{m}}">{{m}}</option>{% endfor %}</select>
      </label>
      <label>Max new <input id="max_new" type="number" value="200" min="1" max="2048"></label>
      <label>Temp <input id="temp" type="number" step="0.1" value="0.9" min="0.1" max="2.0"></label>
      <label>Top-k <input id="topk" type="number" value="40" min="0" max="500"></label>
      <span class="small" id="dev">device: —</span>
      <span class="small" id="meta"></span>
    </div>
  </div>
  <div class="chat" id="chat"></div>
  <div class="card">
    <div class="row">
      <textarea id="ta" class="grow" placeholder="Type..." rows="4"></textarea>
      <button id="send">Send</button>
    </div>
  </div>
  <footer>
    <div><strong>PyAiModel-TFormer-BPERoPESwiGLU</strong> — SentencePiece, RoPE, SwiGLU, GGUF.</div>
    <div>© <span id="year">2025</span>. MIT.</div>
  </footer>
</div>
<script>
(function(){
  const $ = (id)=>document.getElementById(id);
  const chat = $('chat');
  function add(role, text){
    const el = document.createElement('div');
    el.className = 'msg ' + (role==='ai'?'ai':'user');
    el.textContent = text;
    chat.appendChild(el); chat.scrollTop = chat.scrollHeight;
    return el;
  }
  async function startGen(prompt){
    const params = new URLSearchParams({
      model: $('model').value,
      max_new: $('max_new').value,
      temperature: $('temp').value,
      top_k: $('topk').value,
      prompt: prompt
    });
    const es = new EventSource('/gen?'+params.toString());
    let aiEl = add('ai', '');
    es.onmessage = (e)=>{
      const s = e.data || '';
      if (s.startsWith('DEV:')){ $('dev').textContent = 'device: ' + s.slice(4); return; }
      if (s.startsWith('META:')){ $('meta').textContent = ' | ' + s.slice(5); return; }
      if (s.startsWith('TOK:')){ aiEl.textContent += s.slice(4); chat.scrollTop = chat.scrollHeight; return; }
      if (s === 'DONE'){ es.close(); return; }
      if (s.startsWith('ERR:')){ aiEl.textContent += '\\n['+s.slice(4)+']'; es.close(); }
    };
    es.onerror = ()=>{ es.close(); };
  }
  $('send').onclick = ()=>{
    const t = $('ta').value.trim();
    if(!t) return;
    add('user', t); $('ta').value='';
    startGen(t);
  };
  $('ta').addEventListener('keydown', (e)=>{ if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)) $('send').click(); });
})();
</script>
</body></html>"""

app = Flask(__name__)
_CACHE = {}

def _sse(s: str) -> str: return f"data: {s}\n\n"

@app.route("/")
def index():
    models = [p.name for p in sorted(Path("Models").glob("*.gguf"))]
    return render_template_string(HTML, models=models)

@app.route("/gen")
def gen():
    try:
        model_name = request.args.get("model","")
        prompt = request.args.get("prompt","")
        max_new = int(request.args.get("max_new",200))
        temperature = float(request.args.get("temperature",0.9))
        top_k = int(request.args.get("top_k",40))
        if not model_name:
            return Response(_sse("ERR:select model")+_sse("DONE"), mimetype="text/event-stream")

        model_path = str(Path("Models") / model_name)
        spm_path = str(Path(model_path).with_suffix(".tokenizer.model"))

        def stream():
            if model_path not in _CACHE:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, meta = load_torch_from_gguf(model_path, device=device)
                sp = SentencePieceProcessor(model_file=spm_path)
                _CACHE[model_path] = (model, sp, device, meta)
            model, sp, device, meta = _CACHE[model_path]
            dev = device if device=="cpu" else f"cuda ({torch.cuda.get_device_name(0)})"
            yield _sse("DEV:"+dev)
            pretty = f'params≈{sum(p.numel() for p in model.parameters()):,} | d_model={meta["n_embd"]}, heads={meta["n_head"]}, layers={meta["n_layer"]}, ctx={meta["block_size"]}, vocab={meta["vocab_size"]}'
            yield _sse("META:"+pretty)
            try:
                for chunk in generate_stream(model, sp, device, prompt, max_new, temperature, top_k):
                    yield _sse("TOK:"+chunk)
            except Exception as e:
                yield _sse("ERR:"+f"{type(e).__name__}: {e}")
            yield _sse("DONE")
        return Response(stream_with_context(stream()), mimetype="text/event-stream")
    except Exception as e:
        return Response(_sse("ERR:"+f"{type(e).__name__}: {e}")+_sse("DONE"), mimetype="text/event-stream")

if __name__ == "__main__":
    os.makedirs("Models", exist_ok=True)
    os.makedirs("Datasets", exist_ok=True)
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
