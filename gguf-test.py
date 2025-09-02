#!/usr/bin/env python3
# gguf-test.py — Flask UI to validate a GGUF model for LM Studio / llama.cpp compatibility
# Looks for Models/*.gguf, lets you pick one, and streams validation results via SSE.
# Requirements: pip install flask gguf numpy

import os, json, math
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Lock
from typing import Dict, List, Tuple

import numpy as np
from flask import Flask, request, Response, render_template_string, stream_with_context

import gguf  # official library

app = Flask(__name__)
TEST_LOCK = Lock()

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>GGUF Validator</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Arial;background:#f7f7fb;margin:0}
.wrap{max-width:960px;margin:0 auto;padding:16px}
.card{background:#fff;border:1px solid #eee;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
label{display:inline-block;margin:6px 12px 6px 0}
select,button{padding:10px 14px;border-radius:10px;border:1px solid #4f46e5}
button{background:#4f46e5;color:#fff;cursor:pointer}
button[disabled]{opacity:.5;cursor:not-allowed}
pre{white-space:pre-wrap;background:#fafafe;border:1px solid #eee;border-radius:8px;padding:8px;max-height:50vh;overflow:auto}
.kv{display:grid;grid-template-columns:200px 1fr;gap:8px}
.ok{color:#0a7a2f} .warn{color:#9a6b00} .err{color:#b00020}
.small{color:#666;font-size:12px;margin-top:6px}
</style></head>
<body><div class="wrap">
  <h1>GGUF Validator (LLaMA / LM Studio)</h1>
  <div class="card">
    <div class="row">
      <label>Model
        <select id="model">
          {% for m in models %}<option value="{{m}}">{{m}}</option>{% endfor %}
        </select>
      </label>
      <button id="btn">Validate</button>
      <span class="small" id="env"></span>
    </div>
  </div>
  <div class="card">
    <h3>Live log</h3>
    <pre id="log"></pre>
  </div>
<script>
(function(){
  const $ = id => document.getElementById(id);
  function sse(){
    const mdl = $('model').value;
    if(!mdl) { alert('Pick a .gguf'); return; }
    $('btn').disabled = true;
    const es = new EventSource('/validate?model='+encodeURIComponent(mdl));
    const log = $('log');
    es.onmessage = (e)=>{
      const line = e.data || '';
      if(line.startsWith('ENV:')) { $('env').textContent = line.slice(4); return; }
      log.textContent += line + "\\n";
      log.scrollTop = log.scrollHeight;
      if(line === 'DONE') { es.close(); $('btn').disabled = false; }
    };
    es.onerror = ()=>{ es.close(); $('btn').disabled=false; alert('Stream error (see server logs).'); };
  }
  $('btn').onclick = sse;
})();
</script>
</div></body></html>
"""

def _sse(line: str) -> str:
    return f"data: {line}\n\n"

@app.route("/")
def index():
    models_dir = Path("Models")
    models_dir.mkdir(exist_ok=True)
    models = [p.name for p in sorted(models_dir.glob("*.gguf"))]
    return render_template_string(HTML, models=models)

# ---------------------------
# Core validation helpers
# ---------------------------
REQ_KEYS_LLaMA = [
    "general.architecture",
    "general.file_type",
    "llama.context_length",
    "llama.embedding_length",
    "llama.block_count",
    "llama.feed_forward_length",
    "llama.attention.head_count",
    "llama.attention.head_count_kv",
    "llama.attention.key_length",
    "llama.attention.value_length",
    "llama.attention.layer_norm_rms_epsilon",
    "llama.rope.dimension_count",
    "llama.rope.freq_base",
    # tokenizer (SentencePiece layout inside GGUF):
    "tokenizer.ggml.model",
    "tokenizer.ggml.tokens_count",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
]

def _to_int(v):
    try: return int(v)
    except: return None
def _to_float(v):
    try: return float(v)
    except: return None

def read_gguf(path: str):
    r = gguf.GGUFReader(path)

    # --- KV ---
    kv_map: Dict[str, object] = {}
    if hasattr(r, "get_kv_data"):
        # новые версии
        for kv in r.get_kv_data():
            kv_map[kv.key] = kv.value
    elif hasattr(r, "kv_data"):
        # старые версии
        if isinstance(r.kv_data, dict):
            kv_map = dict(r.kv_data)
        else:
            # иногда это список объектов
            for e in r.kv_data:
                k = getattr(e, "key", None) or getattr(e, "name", None)
                v = getattr(e, "value", None)
                if k: kv_map[k] = v

    # --- tensors ---
    tensors = []
    if hasattr(r, "get_tensors"):
        for t in r.get_tensors():
            tensors.append((t.name, list(t.shape), t.tensor_type, t.nbytes))
    elif hasattr(r, "tensors"):
        for t in r.tensors:
            name = getattr(t, "name", None)
            shape = getattr(t, "shape", None)
            dtype = getattr(t, "tensor_type", None)
            nbytes = getattr(t, "nbytes", None)
            if name is None and isinstance(t, (tuple, list)) and len(t) >= 4:
                name, shape, dtype, nbytes = t[0], t[1], t[2], t[3]
            tensors.append((name, list(shape) if shape is not None else [], dtype, nbytes))
    return kv_map, tensors

def expected_tensor_names(n_layer: int) -> List[str]:
    names = [
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
    ]
    for i in range(n_layer):
        base = f"blk.{i}"
        names += [
            f"{base}.attn_norm.weight",
            f"{base}.attn_q.weight",
            f"{base}.attn_k.weight",
            f"{base}.attn_v.weight",
            f"{base}.attn_output.weight",
            f"{base}.ffn_norm.weight",
            f"{base}.ffn_gate.weight",
            f"{base}.ffn_up.weight",
            f"{base}.ffn_down.weight",
        ]
    return names

def validate_llama_kv(kv: Dict[str, object]) -> Tuple[List[str], List[str]]:
    missing, type_warn = [], []
    for key in REQ_KEYS_LLaMA:
        if key not in kv:
            missing.append(key)

    # basic types / values
    arch = kv.get("general.architecture")
    if arch != "llama":
        type_warn.append(f"general.architecture is '{arch}', expected 'llama'")

    # ints
    i_keys = [
        "llama.context_length", "llama.embedding_length", "llama.block_count",
        "llama.feed_forward_length", "llama.attention.head_count", "llama.attention.head_count_kv",
        "llama.attention.key_length", "llama.attention.value_length", "llama.rope.dimension_count",
        "tokenizer.ggml.tokens_count", "general.file_type"
    ]
    for k in i_keys:
        if k in kv and _to_int(kv[k]) is None:
            type_warn.append(f"{k} should be u32")

    # floats
    f_keys = ["llama.attention.layer_norm_rms_epsilon", "llama.rope.freq_base"]
    for k in f_keys:
        if k in kv and _to_float(kv[k]) is None:
            type_warn.append(f"{k} should be f32")

    # cross-check head_dim
    if all(k in kv for k in ("llama.embedding_length", "llama.attention.head_count", "llama.attention.key_length")):
        d = _to_int(kv["llama.embedding_length"])
        h = _to_int(kv["llama.attention.head_count"])
        hd = _to_int(kv["llama.attention.key_length"])
        if d and h and hd and d // h != hd:
            type_warn.append(f"head_dim mismatch: embd/heads={d//h} != key_length={hd}")

    # tokenizer arrays length
    if all(k in kv for k in ("tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type", "tokenizer.ggml.tokens_count")):
        tn = len(kv["tokenizer.ggml.tokens"])
        sn = len(kv["tokenizer.ggml.scores"])
        pn = len(kv["tokenizer.ggml.token_type"])
        tc = _to_int(kv["tokenizer.ggml.tokens_count"])
        if not (tn == sn == pn):
            type_warn.append(f"token arrays mismatch: tokens={tn}, scores={sn}, token_type={pn}")
        if tc is not None and tc != tn:
            type_warn.append(f"tokens_count={tc} but tokens len={tn}")
    return missing, type_warn

def validate_tensors(kv: Dict[str, object], tensors: List[Tuple[str, List[int], int, int]]) -> Tuple[List[str], List[str]]:
    n_layer = _to_int(kv.get("llama.block_count", 0)) or 0
    n_embd  = _to_int(kv.get("llama.embedding_length", 0)) or 0
    n_head  = _to_int(kv.get("llama.attention.head_count", 0)) or 0
    vocab   = _to_int(kv.get("tokenizer.ggml.tokens_count", 0)) or 0

    expected = set(expected_tensor_names(n_layer))
    present  = {name for (name, _, _, _) in tensors}

    missing = sorted(list(expected - present))
    extra   = sorted(list(present - expected))

    # quick shape checks (header-level)
    shape_warn = []
    tmap = {name: shape for (name, shape, _, _) in tensors}
    # embeddings/output
    if "token_embd.weight" in tmap and n_embd and vocab:
        shp = tmap["token_embd.weight"]
        if tuple(shp) != (vocab, n_embd):
            shape_warn.append(f"token_embd.weight shape {shp} != ({vocab}, {n_embd})")
    if "output.weight" in tmap and n_embd and vocab:
        shp = tmap["output.weight"]
        if tuple(shp) != (n_embd, vocab):
            shape_warn.append(f"output.weight shape {shp} != ({n_embd}, {vocab})")

    # block weights (rough checks)
    for i in range(n_layer):
        base = f"blk.{i}"
        for key in ("attn_q", "attn_k", "attn_v", "attn_output"):
            nm = f"{base}.{key}.weight"
            if nm in tmap and n_embd:
                shp = tmap[nm]
                if shp[-1] != n_embd:
                    shape_warn.append(f"{nm} last-dim {shp} expected * x {n_embd}")
        for key in ("ffn_gate", "ffn_up"):
            nm = f"{base}.{key}.weight"
            if nm in tmap and n_embd:
                shp = tmap[nm]
                if shp[-1] != n_embd:
                    shape_warn.append(f"{nm} last-dim {shp} expected * x {n_embd}")
        nm = f"{base}.ffn_down.weight"
        if nm in tmap and n_embd:
            shp = tmap[nm]
            if shp[-1] != (4 * n_embd * 2 // 3):
                shape_warn.append(f"{nm} last-dim {shp} expected * x {4*n_embd*2//3}")

    # biases: LM Studio expects llama w/o biases; warn if any found
    bias_like = [n for n in present if n.endswith(".bias")]
    if bias_like:
        shape_warn.append(f"bias tensors present (unexpected for llama): {', '.join(sorted(bias_like)[:8])}{'…' if len(bias_like)>8 else ''}")

    # unexpected prefix
    wrong_prefix = [n for n in extra if n.startswith("blocks.")]
    for n in wrong_prefix:
        shape_warn.append(f"unexpected prefix 'blocks.' found: {n} (should be 'blk.')")

    return missing + extra, shape_warn

# ---------------------------
# Validate endpoint (SSE)
# ---------------------------
@app.route("/validate")
def validate_route():
    mdl = request.args.get("model", "")
    if not mdl:
        return Response(_sse("Select a model") + _sse("DONE"), mimetype="text/event-stream")

    path = str(Path("Models") / mdl)
    if not Path(path).is_file():
        return Response(_sse("Model not found") + _sse("DONE"), mimetype="text/event-stream")

    if not TEST_LOCK.acquire(blocking=False):
        return Response(_sse("Another validation is running") + _sse("DONE"), mimetype="text/event-stream")

    q: Queue[str] = Queue()

    def log(s: str): q.put(s)

    def worker():
        try:
            # env
            ver = getattr(gguf, "__version__", "unknown")
            log("ENV: gguf " + str(ver))

            log(f"Opening: {path}")
            try:
                kv, tensors = read_gguf(path)
            except Exception as e:
                log("ERROR: failed to read gguf: " + repr(e))
                return

            # show brief KV
            arch = kv.get("general.architecture")
            ctx  = kv.get("llama.context_length")
            d    = kv.get("llama.embedding_length")
            L    = kv.get("llama.block_count")
            H    = kv.get("llama.attention.head_count")
            log(f"KV: arch={arch}, ctx={ctx}, d_model={d}, layers={L}, heads={H}")

            # KV completeness
            missing, warns = validate_llama_kv(kv)
            if missing:
                log("Missing KV keys:")
                for k in missing:
                    log("  - " + k)
            else:
                log("KV keys: OK (required present)")

            for w in warns:
                log("WARN: " + w)

            # Tokenizer checks
            tcount = kv.get("tokenizer.ggml.tokens_count")
            tlen = len(kv.get("tokenizer.ggml.tokens", [])) if "tokenizer.ggml.tokens" in kv else None
            log(f"Tokenizer: model={kv.get('tokenizer.ggml.model')} tokens_count={tcount} tokens_len={tlen}")

            # Tensor presence / names
            bad_names, shape_warns = validate_tensors(kv, tensors)
            if bad_names:
                log("Tensor name issues (missing/extra):")
                for n in bad_names[:100]:
                    log("  - " + n)
                if len(bad_names) > 100:
                    log(f"  ... and {len(bad_names)-100} more")
            else:
                log("Tensor names: OK")

            for w in shape_warns[:100]:
                log("WARN: " + w)
            if len(shape_warns) > 100:
                log(f"WARN: ... and {len(shape_warns)-100} more")

            # Final verdict
            if not missing and not bad_names:
                log("RESULT: Looks compatible for LM Studio / llama.cpp.")
            else:
                log("RESULT: Issues found above. Fix missing KV and tensor names.")

        except Exception as e:
            log("ERROR: " + repr(e))
        finally:
            q.put("DONE")
            TEST_LOCK.release()

    Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            try:
                msg = q.get(timeout=1.0)
                yield _sse(msg)
                if msg == "DONE": break
            except Empty:
                pass

    return Response(stream_with_context(stream()), mimetype="text/event-stream")

if __name__ == "__main__":
    os.makedirs("Models", exist_ok=True)
    app.run("0.0.0.0", port=5050, debug=False, threaded=True)
