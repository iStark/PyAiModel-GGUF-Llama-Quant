#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyAiModel-TFormer-BPERoPESwiGLU — generator_frontend.py (LM/SFT/BOTH, ChatML, auto-JSONL)
Flask UI for training LLaMA-style model (SentencePiece + RoPE + SwiGLU),
exporting GGUF via official gguf lib, and optional quantization via llama-quantize.

Особенности:
- Task по умолчанию: BOTH (LM + SFT)
- SFT JSONL можно не выбирать — бекенд сам найдёт dataset в Datasets/ (автопоиск в generator.py)
- Подсказки по настройке LM Studio (ChatML)
"""

import time, os, json
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Lock

from flask import Flask, request, Response, render_template_string, stream_with_context

from generator import train_once  # training core + GGUF + quant

HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>BPE/SP Transformer Trainer</title>
<style>
:root{--pri:#4f46e5;}
*{box-sizing:border-box}
body{font-family:system-ui,Segoe UI,Roboto,Arial;background:#f7f7fb;margin:0}
.wrap{max-width:1100px;margin:0 auto;padding:18px}
h1{margin:8px 0 12px}
.card{background:#fff;border:1px solid #eee;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
label{display:inline-block;margin:6px 12px 6px 0}
input,select,textarea{padding:8px;border:1px solid #ddd;border-radius:8px;font:inherit}
button{padding:10px 14px;border-radius:10px;border:1px solid var(--pri);background:var(--pri);color:#fff;cursor:pointer}
button[disabled]{opacity:.5;cursor:not-allowed}
.bar{height:8px;background:#ececff;border-radius:999px;overflow:hidden}
.bar i{display:block;height:100%;width:0%;background:var(--pri)}
.grid{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:8px;margin-top:8px}
.kv{background:#fafafe;border:1px solid #eee;border-radius:8px;padding:8px}
.kv b{display:block;font-size:12px;color:#666;margin-bottom:4px}
.kv span{font-variant-numeric:tabular-nums}
pre{white-space:pre-wrap;max-height:420px;overflow:auto;background:#0b1020;color:#dbe2ff;padding:12px;border-radius:12px}
.small{color:#666;font-size:12px;margin-top:6px}
footer{margin:0 auto;margin-bottom:2rem;font-size:.9em;color:#666}
footer div a{color:inherit}
.links a{margin-right:.75rem}
.row{display:flex;gap:12px;flex-wrap:wrap}
.sep{height:1px;background:#eee;margin:12px 0}
.help{font-size:.95em;color:#333;line-height:1.45}
.help code{background:#f4f4ff;border:1px solid #e7e7ff;padding:1px 5px;border-radius:6px}
ol,ul{padding-left:18px}
kbd{background:#eee;border-radius:6px;padding:2px 6px;border:1px solid #ccc}
.copy{border:1px dashed #ccc;border-radius:10px;padding:10px;background:#fafbff}
textarea.template{width:100%;min-height:150px}
.mono{font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;}
</style></head>
<body><div class="wrap">
  <h1>LLaMA-style Transformer Trainer (CUDA) · GGUF · Quant · ChatML</h1>

  <div class="card">
    <form id="f" onsubmit="start();return false;">
      <div class="row">
        <label>Task
          <select id="task">
            <option value="lm">LM (.txt)</option>
            <option value="sft">SFT (.jsonl → ChatML)</option>
            <option value="both" selected>BOTH (LM + SFT)</option>
          </select>
        </label>

        <label>Tokenizer TXT (SPM corpus)
          <select id="txt">
            {% for f in txts %}<option value="{{f}}">{{f}}</option>{% endfor %}
          </select>
        </label>

        <label>SFT JSONL (optional)
          <select id="jsonl">
            <option value="">(auto-discover in /Datasets)</option>
            {% for f in jsonls %}<option value="{{f}}">{{f}}</option>{% endfor %}
          </select>
        </label>
      </div>

      <div class="sep"></div>

      <div class="row">
        <label>Vocab <input id="vocab" type="number" value="60000" min="2000" max="200000"></label>
        <label>D_model <input id="dmodel" type="number" value="576" min="128" max="4096"></label>
        <label>Heads <input id="heads" type="number" value="9" min="1" max="64"></label>
        <label>Layers <input id="layers" type="number" value="9" min="1" max="80"></label>
        <label>Seq <input id="seq" type="number" value="256" min="32" max="8192"></label>
        <label>Batch <input id="bs" type="number" value="64" min="1" max="4096"></label>
        <label>Epochs (LM) <input id="ep" type="number" value="4" min="1" max="100"></label>
        <label>Epochs SFT <input id="ep_sft" type="number" value="3" min="1" max="100"></label>
        <label>LR <input id="lr" step="0.0001" type="number" value="0.0003"></label>
        <label><input id="amp" type="checkbox" checked> Use AMP (Tensor Cores)</label>
      </div>

      <div class="sep"></div>

      <div class="row">
        <label>Quant
          <select id="quant">
            <option value="">(none, keep FP16/FP32)</option>
            <option>Q2_K</option>
            <option selected>Q3_K_L</option>
            <option>Q3_K_M</option>
            <option>Q4_0</option>
            <option>Q4_K_M</option>
            <option>Q5_K_S</option>
            <option>Q5_K_M</option>
            <option>Q8_0</option>
            <option>IQ2_M</option>
            <option>IQ3_M</option>
            <option>IQ4_NL</option>
            <option>IQ4_XS</option>
            <option>IQ5_NL</option>
          </select>
        </label>
        <label>llama-quantize path <input id="quant_bin" type="text" placeholder="auto from PATH"></label>
        <label><input id="del_fp16" type="checkbox"> Delete FP16/FP32 after quant</label>
      </div>
<div class="sep"></div>

      <div class="row">
        <h3 style="margin:0 0 6px 0;width:100%;">Multi-GPU (2-GPU pipeline)</h3>
        <label>
          <input id="mgpu_enable" type="checkbox">
          Enable 2-GPU pipeline
        </label>
        <label title="Сколько первых блоков уедет на малую видеокарту (например GTX 1060 3GB)">
          Pipeline split
          <input id="mgpu_split" type="number" value="1" min="0" max="999">
        </label>
        <label title="Индекс основной (большой) видеокарты, обычно 0 (RTX 3070 Ti)">
          Big GPU index
          <input id="mgpu_big" type="number" value="0" min="0" max="15">
        </label>
        <label title="Индекс малой видеокарты, обычно 1 (GTX 1060 3GB)">
          Small GPU index
          <input id="mgpu_small" type="number" value="1" min="0" max="15">
        </label>
      </div>
      <button id="btn">Start training</button>
    </form>
  </div>

  <div class="card help">
    <h3>LM Studio · Рекомендованные настройки под ChatML</h3>
    <p>В <b>Prompt format</b> выбери <b>Manual/Custom</b> и задай поля так:</p>
    <div class="copy mono">
      <b>Перед Системой</b>
      <pre>&lt;|im_start|&gt;system</pre>
      <b>После Системы</b>
      <pre>\n&lt;|im_end|&gt;\n</pre>
      <b>Перед Пользователем</b>
      <pre>&lt;|im_start|&gt;user</pre>
      <b>После Пользователя</b>
      <pre>\n&lt;|im_end|&gt;\n</pre>
      <b>Перед Ассистентом</b>
      <pre>&lt;|im_start|&gt;assistant</pre>
      <b>После Ассистента</b>
      <pre>(оставь пусто)</pre>

      <b>Stop sequences</b>:
      <pre>&lt;|im_end|&gt;
&lt;/s&gt;</pre>

      <b>Grammar (Regex)</b>:
      <pre>[\s\S]*</pre>
      <i>Это пермишсивный паттерн: допускает любой Unicode/байты (эмодзи и пр.).</i>
    </div>
    <p class="small">Токенайзер включает ChatML-токены и <code>byte_fallback=True</code> (см. generator.py).</p>
  </div>

  <div class="card">
    <h3>Progress</h3>
    <div class="bar"><i id="p"></i></div>
    <div class="grid" id="stats-grid">
      <div class="kv"><b>%</b><span id="s_pct">0.00%</span></div>
      <div class="kv"><b>Epoch</b><span id="s_epoch">0 / 0</span></div>
      <div class="kv"><b>Steps</b><span id="s_step">0 / 0</span></div>
      <div class="kv"><b>Loss</b><span id="s_loss">—</span></div>
      <div class="kv"><b>s/it</b><span id="s_spit">—</span></div>
      <div class="kv"><b>Elapsed</b><span id="s_elapsed">—</span></div>
      <div class="kv"><b>ETA</b><span id="s_eta">—</span></div>
      <div class="kv"><b>Device</b><span id="s_device">—</span></div>
      <div class="kv"><b>Params</b><span id="s_params">—</span></div>
      <div class="kv"><b>Out</b><span id="s_out">—</span></div>
    </div>
    <pre id="log"></pre>
  </div>

  <footer>
    <div><strong>PyAiModel TFormer</strong> — SentencePiece (BPE), RoPE, SwiGLU, CUDA AMP, GGUF export, llama-quantize. LM/SFT/BOTH. ChatML-friendly.</div>
    <div>© <span id="year">2025</span>. MIT.</div>
  </footer>
</div>
<script>
(function(){
  const $ = (id)=>document.getElementById(id);
  function val(id){ return $(id).value; }

  function parseProgress(line){
    const pct   = (line.match(/Progress:\s+([\d.]+)%/) || [,''])[1];
    const epoch = (line.match(/epoch\s+(\d+)\/(\d+)/) || [,,])[1];
    const epochT= (line.match(/epoch\s+(\d+)\/(\d+)/) || [,,,''])[2];
    const step  = (line.match(/step\s+(\d+)\/(\d+)/) || [,,])[1];
    const stepT = (line.match(/step\s+(\d+)\/(\d+)/) || [,,,''])[2];
    const loss  = (line.match(/loss\s+([\d.]+)/) || [,''])[1];
    const spit  = (line.match(/s_it\s+([\d.]+s\/it)/) || [,''])[1];
    const elapsed = (line.match(/elapsed\s+([\d.]+m)/) || [,''])[1];
    const eta     = (line.match(/ETA\s+([\d.]+m)/) || [,''])[1];
    if (pct){ $('p').style.width = pct + '%'; $('s_pct').textContent = pct + '%'; }
    if (epoch && epochT) $('s_epoch').textContent = epoch + ' / ' + epochT;
    if (step && stepT)   $('s_step').textContent  = step + ' / ' + stepT;
    if (loss)   $('s_loss').textContent  = loss;
    if (spit)   $('s_spit').textContent  = spit;
    if (elapsed)$('s_elapsed').textContent = elapsed;
    if (eta)    $('s_eta').textContent     = eta || '—';
  }

  let es = null;
  function setBusy(b){ $('btn').disabled = b; }

  function handleEnvLine(line){
    let payload = line.startsWith('ENV ') ? line.slice(4) : line;
    try{
      const env = JSON.parse(payload);
      let dev = env.device || '—';
      if (Array.isArray(env.cuda_names) && env.cuda_names.length) {
        dev += ' — ' + env.cuda_names.map((n,i)=>`GPU${i}: ${n}`).join(', ');
      }
      const amp = env.cuda_available ? 'True' : 'False';
      $('s_device').textContent = dev + ' | AMP: ' + amp;

      if (env.parameter_count || env.size_label) {
        const n = env.parameter_count ? env.parameter_count.toLocaleString('en-US') : '—';
        const sl = env.size_label ? ` (${env.size_label})` : '';
        $('s_params').textContent = n + sl;
      }
    }catch(_){}
  }

  window.start = function(){
    const task = val('task');
    const txt = val('txt');
    const jsonl = val('jsonl'); // optional (auto-discover on backend)

    if (!txt || txt.indexOf('.txt') === -1) {
      alert('Select tokenizer TXT (.txt)');
      return;
    }

    setBusy(true);
    const params = new URLSearchParams({
      task: task,
      txt: txt,
      jsonl: jsonl || '',
      vocab: val('vocab'),
      dmodel: val('dmodel'),
      heads:  val('heads'),
      layers: val('layers'),
      seq:    val('seq'),
      bs:     val('bs'),
      ep:     val('ep'),
      ep_sft: val('ep_sft'),
      lr:     val('lr'),
      amp:    $('amp').checked ? '1':'0',
      quant:  val('quant'),
      quant_bin: val('quant_bin') || '',
      del_fp16: $('del_fp16').checked ? '1':'0'
    });
     // --- Multi-GPU params ---
    const mgpuOn   = $('mgpu_enable').checked;
    const mgpuSplit= mgpuOn ? (parseInt($('mgpu_split').value||'0',10)||0) : 0;
    params.set('pipeline_split', String(mgpuSplit));
    params.set('big_dev',  String(parseInt($('mgpu_big').value||'0',10)||0));
    params.set('small_dev',String(parseInt($('mgpu_small').value||'1',10)||1));

    es  = new EventSource('/train?' + params.toString());
    const log = $('log');
    es.onmessage = function(e){
    
      const line = e.data || '';
  if (line.startsWith('ENV ')) {
    try {
      const env = JSON.parse(line.slice(4));
      if (env.auto_jsonl) {
        document.getElementById('s_out').textContent = 'JSONL: ' + env.auto_jsonl;
      }
    } catch(_) {}
    log.textContent += line + "\n"; log.scrollTop = log.scrollHeight; return;
  }
  if (line.startsWith('PHASE:')) {
    // просто выделим фазу в логе
    log.textContent += "\n=== " + line + " ===\n";
    log.scrollTop = log.scrollHeight;
    return;
  }
      if (line.startsWith('{') || line.startsWith('ENV ')) { handleEnvLine(line); log.textContent += line + "\n"; log.scrollTop = log.scrollHeight; return; }
      if (line.startsWith('PARAMS ')) { log.textContent += line + "\n"; log.scrollTop = log.scrollHeight; return; }
      if (line.startsWith('Saved weights: ')) { $('s_out').textContent = line.replace('Saved weights: ', ''); }
      if (line.startsWith('Training started') || line.startsWith('Quantizing ')) { log.textContent += line + "\n"; log.scrollTop = log.scrollHeight; return; }
      if (line.startsWith('Progress:')) parseProgress(line);
      if (line === 'DONE'){ try{es.close();}catch(_){} setBusy(false); return; }
      if (line.startsWith('ERROR:')){ log.textContent += line + "\n"; try{es.close();}catch(_){} setBusy(false); return; }
      if (line.startsWith('ERR:busy')){ alert('Training already running'); try{es.close();}catch(_){} setBusy(false); return; }
      log.textContent += line + "\n"; log.scrollTop = log.scrollHeight;
    };
    es.onerror = function(){
      try{es.close();}catch(_){}
      setBusy(false);
      alert('Stream interrupted. Check server logs.');
    };
  };
})();
</script>
</body></html>"""

app = Flask(__name__)
TRAIN_LOCK = Lock()


def _sse(line:str) -> str:
    return f"data: {line}\n\n"


@app.route("/")
def index():
    ds_dir = Path("Datasets"); ds_dir.mkdir(exist_ok=True)
    txts = sorted([p.name for p in ds_dir.glob("*.txt")]) or ["dataset.txt"]
    jsonls = sorted([p.name for p in ds_dir.glob("*.jsonl")])
    return render_template_string(HTML, txts=txts, jsonls=jsonls)


@app.route("/train")
def train_route():
    task = request.args.get("task", "both").strip()
    txt  = request.args.get("txt", "dataset.txt").strip()
    jsonl = request.args.get("jsonl", "").strip() or None

    vocab = int(request.args.get("vocab", 60000))
    dmodel = int(request.args.get("dmodel", 576))
    heads  = int(request.args.get("heads", 9))
    layers = int(request.args.get("layers", 9))
    seq    = int(request.args.get("seq", 256))
    bs     = int(request.args.get("bs", 64))
    ep     = int(request.args.get("ep", 4))
    ep_sft = request.args.get("ep_sft", "")
    ep_sft = int(ep_sft) if ep_sft.strip().isdigit() else None
    lr     = float(request.args.get("lr", 0.0003))
    amp    = request.args.get("amp", "1") == "1"

    quant  = request.args.get("quant", "").strip() or None
    quant_bin = request.args.get("quant_bin", "").strip() or None
    del_fp16 = request.args.get("del_fp16", "0") == "1"
    # --- Multi-GPU ---
    pipeline_split = int(request.args.get("pipeline_split", 0))
    big_dev = int(request.args.get("big_dev", 0))
    small_dev = int(request.args.get("small_dev", 1))
    data_path = str(Path("Datasets") / txt)
    data_jsonl = str(Path("Datasets") / jsonl) if jsonl else None
    models_dir = "Models"

    stamp = time.strftime('%Y%m%d-%H%M%S')
    base = Path(txt).stem if task == 'lm' or not jsonl else Path(jsonl).stem
    run_name = f"{base}-d{dmodel}h{heads}l{layers}-seq{seq}-bs{bs}-{stamp}"

    if not TRAIN_LOCK.acquire(blocking=False):
        return Response(_sse("ERR:busy") + _sse("DONE"), mimetype="text/event-stream")

    q: Queue[str] = Queue()
    def progress_cb(msg: str): q.put(msg)

    def worker():
        try:
            saved_path = train_once(
                run_name=run_name,
                data_path=data_path,
                models_dir=models_dir,
                vocab_size=vocab,
                block_size=seq,
                n_layer=layers,
                n_head=heads,
                n_embd=dmodel,
                lr=lr,
                batch_size=bs,
                epochs=ep,
                weight_dtype='float16',
                amp=True if amp else False,
                progress_cb=progress_cb,
                quant=quant,
                quant_bin=quant_bin,
                keep_fp16=(not del_fp16),
                # extended args
                task=task,
                data_jsonl=data_jsonl,
                epochs_sft=ep_sft,
                # Multi-GPU
                pipeline_split=pipeline_split,
                big_dev=big_dev,
                small_dev=small_dev
            )
            q.put(f"Saved weights: {saved_path}")
        except Exception as e:
            q.put(f"ERROR: {type(e).__name__}: {e}")
        finally:
            q.put("DONE")
            TRAIN_LOCK.release()

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
    os.makedirs("Datasets", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
