#!/usr/bin/env python3
# PyAiModel-TFormer-BPERoPESwiGLU — generator_frontend.py
# Flask UI for training LLaMA-style model (SentencePiece + RoPE + SwiGLU),
# exporting GGUF via official gguf lib, and optional quantization via llama-quantize.
# Uses SSE for live progress and a lock to avoid concurrent trainings.

import time, os, json
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Lock

from flask import Flask, request, Response, render_template_string, stream_with_context

from generator import train_once  # training core + GGUF + quant

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>BPE/SP Transformer Trainer</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Arial;background:#f7f7fb;margin:0}
.wrap{max-width:960px;margin:0 auto;padding:16px}
.card{background:#fff;border:1px solid #eee;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
label{display:inline-block;margin:6px 12px 6px 0}
input,select{padding:8px;border:1px solid #ddd;border-radius:8px}
button{padding:10px 14px;border-radius:10px;border:1px solid #4f46e5;background:#4f46e5;color:#fff;cursor:pointer}
button[disabled]{opacity:.5;cursor:not-allowed}
.bar{height:8px;background:#ececff;border-radius:999px;overflow:hidden}
.bar i{display:block;height:100%;width:0%;background:#4f46e5}
.grid{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:8px;margin-top:8px}
.kv{background:#fafafe;border:1px solid #eee;border-radius:8px;padding:8px}
.kv b{display:block;font-size:12px;color:#666;margin-bottom:4px}
.kv span{font-variant-numeric:tabular-nums}
pre{white-space:pre-wrap}
.small{color:#666;font-size:12px;margin-top:6px}
footer{margin:0 auto;margin-bottom:2rem;font-size:.9em;color:#666}
footer div a{color:inherit}
.links a{margin-right:.75rem}
</style></head>
<body><div class="wrap">
  <h1>LLaMA-style Transformer Trainer (CUDA) + GGUF + Quant</h1>
  <div class="card">
    <form id="f" onsubmit="start();return false;">
      <label>Dataset
        <select id="ds">
          {% for f in datasets %}<option value="{{f}}">{{f}}</option>{% endfor %}
        </select>
      </label>
      <label>Vocab <input id="vocab" type="number" value="32000" min="2000" max="200000"></label>
      <label>D_model <input id="dmodel" type="number" value="512" min="128" max="4096"></label>
      <label>Heads <input id="heads" type="number" value="8" min="1" max="64"></label>
      <label>Layers <input id="layers" type="number" value="8" min="1" max="80"></label>
      <label>Seq <input id="seq" type="number" value="256" min="32" max="8192"></label>
      <label>Batch <input id="bs" type="number" value="64" min="1" max="4096"></label>
      <label>Epochs <input id="ep" type="number" value="2" min="1" max="100"></label>
      <label>LR <input id="lr" step="0.0001" type="number" value="0.0003"></label>
      <label><input id="amp" type="checkbox" checked> Use AMP (Tensor Cores)</label>
      <br>
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
      <button id="btn">Start training</button>
    </form>
  </div>

  <div class="card">
    <h3>Progress</h3>
    <div class="bar"><i id="p"></i></div>
    <div class="grid" id="stats-grid">
      <div class="kv"><b>%</b><span id="s_pct">0.00%</span></div>
      <div class="kv"><b>Elapsed</b><span id="s_elapsed">0.00m</span></div>
      <div class="kv"><b>ETA</b><span id="s_eta">—</span></div>
      <div class="kv"><b>Epoch</b><span id="s_epoch">0 / 0</span></div>
      <div class="kv"><b>Steps</b><span id="s_step">0 / 0</span></div>
      <div class="kv"><b>Loss</b><span id="s_loss">—</span></div>
      <div class="kv"><b>s/it</b><span id="s_spit">—</span></div>
      <div class="kv"><b>Device</b><span id="s_device">—</span></div>
      <div class="kv"><b>Params</b><span id="s_params">—</span></div>
      <div class="kv"><b>Out</b><span id="s_out">—</span></div>
    </div>
    <pre id="log"></pre>
  </div>
  <footer>
    <div><strong>PyAiModel TFormer</strong> — SentencePiece (unigram), RoPE, SwiGLU, CUDA AMP, GGUF export, llama-quantize.</div>
    <div>© <span id="year">2025</span>. MIT.</div>
  </footer>
</div>
<script>
(function(){
  const $ = (id)=>document.getElementById(id);
  function val(id){ return $(id).value; }

  function parseProgress(line){
    const pct = (line.match(/Progress:\\s+([\\d.]+)%/) || [,''])[1];
    const epoch = (line.match(/epoch\\s+(\\d+)\\/(\\d+)/) || [,,])[1];
    const epochT = (line.match(/epoch\\s+(\\d+)\\/(\\d+)/) || [,,,''])[2];
    const step  = (line.match(/step\\s+(\\d+)\\/(\\d+)/) || [,,])[1];
    const stepT = (line.match(/step\\s+(\\d+)\\/(\\d+)/) || [,,,''])[2];
    const loss  = (line.match(/loss\\s+([\\d.]+)/) || [,''])[1];
    const spit  = (line.match(/s_it\\s+([\\d.]+s\\/it)/) || [,''])[1];
    const elapsed = (line.match(/elapsed\\s+([\\d.]+m)/) || [,''])[1];
    const eta     = (line.match(/ETA\\s+([\\d.]+m)/) || [,''])[1];
    if (pct) { $('p').style.width = pct + '%'; $('s_pct').textContent = pct + '%'; }
    if (epoch && epochT) $('s_epoch').textContent = epoch + ' / ' + epochT;
    if (step && stepT) $('s_step').textContent = step + ' / ' + stepT;
    if (loss) $('s_loss').textContent = loss;
    if (spit) $('s_spit').textContent = spit;
    if (elapsed) $('s_elapsed').textContent = elapsed;
    if (eta) $('s_eta').textContent = eta || '—';
  }

  let es = null;
  function setBusy(b){ $('btn').disabled = b; }

  function handleEnvLine(line){
    let payload = line.startsWith('ENV ') ? line.slice(4) : line;
    try{
      const env = JSON.parse(payload);
      let dev = env.device || '—';
      if (env.cuda_name0) dev += ` (${env.cuda_name0})`;
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
    const sel = $('ds');
    if (!sel || !sel.value || sel.value.indexOf('.txt') === -1) {
      alert('Select a dataset (.txt).');
      return;
    }
    setBusy(true);
    const params = new URLSearchParams({
      ds: sel.value,
      vocab: val('vocab'),
      dmodel: val('dmodel'),
      heads:  val('heads'),
      layers: val('layers'),
      seq:    val('seq'),
      bs:     val('bs'),
      ep:     val('ep'),
      lr:     val('lr'),
      amp:    $('amp').checked ? '1':'0',
      quant:  val('quant'),
      quant_bin: val('quant_bin') || '',
      del_fp16: $('del_fp16').checked ? '1':'0'
    });
    es  = new EventSource('/train?' + params.toString());
    const log = $('log');
    es.onmessage = function(e){
      const line = e.data || '';
      if (line.startsWith('{') || line.startsWith('ENV ')) {
        handleEnvLine(line);
        log.textContent += line + "\\n";
        log.scrollTop = log.scrollHeight;
        return;
      }
      if (line.startsWith('PARAMS ')) {
        const m = line.match(/^PARAMS\\s+(.+?)\\s+\\((.+?)\\)/);
        if (m) $('s_params').textContent = `${Number(m[1]).toLocaleString('en-US')} (${m[2]})`;
        log.textContent += line + "\\n";
        log.scrollTop = log.scrollHeight;
        return;
      }
      if (line.startsWith('Saved weights: ')) {
        $('s_out').textContent = line.replace('Saved weights: ', '');
      }
      if (line.startsWith('Training started') || line.startsWith('Quantizing ')) {
        log.textContent += line + "\\n";
        log.scrollTop = log.scrollHeight;
        return;
      }
      if (line.startsWith('Progress:')) parseProgress(line);
      if (line === 'DONE'){ es.close(); setBusy(false); return; }
      if (line.startsWith('ERR:busy')){ alert('Training already running'); es.close(); setBusy(false); return; }
      log.textContent += line + "\\n";
      log.scrollTop = log.scrollHeight;
    };
    es.onerror = function(){
      if (es){ es.close(); }
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
    ds_dir = Path("Datasets")
    ds_dir.mkdir(exist_ok=True)
    datasets = sorted([p.name for p in ds_dir.glob("*.txt")])
    return render_template_string(HTML, datasets=datasets)

@app.route("/train")
def train_route():
    ds = request.args.get("ds", "dataset.txt")
    vocab = int(request.args.get("vocab", 32000))
    dmodel = int(request.args.get("dmodel", 512))
    heads  = int(request.args.get("heads", 8))
    layers = int(request.args.get("layers", 8))
    seq    = int(request.args.get("seq", 256))
    bs     = int(request.args.get("bs", 64))
    ep     = int(request.args.get("ep", 2))
    lr     = float(request.args.get("lr", 0.0003))
    amp    = request.args.get("amp", "1") == "1"

    quant  = request.args.get("quant", "").strip() or None
    quant_bin = request.args.get("quant_bin", "").strip() or None
    del_fp16 = request.args.get("del_fp16", "0") == "1"

    data_path = str(Path("Datasets") / ds)
    models_dir = "Models"

    stamp = time.strftime('%Y%m%d-%H%M%S')
    base = Path(ds).stem
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
                keep_fp16=(not del_fp16)
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
