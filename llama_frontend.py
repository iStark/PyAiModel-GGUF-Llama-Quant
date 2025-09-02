#!/usr/bin/env python3
# llama_frontend.py — a tiny Flask UI to run llama-cli (llama.cpp) with your GGUF.
# Place in project root; run: python llama_frontend.py
# Open http://localhost:5001

import os
from pathlib import Path
from flask import Flask, request, Response, render_template_string, stream_with_context
from llama_runner import run_llama_cli

HTML = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>llama.cpp Runner UI</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Arial;background:#f7f7fb;margin:0}
.wrap{max-width:900px;margin:0 auto;padding:16px}
.card{background:#fff;border:1px solid #eee;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
label{display:block;margin:8px 0}
input,select,textarea{width:100%;padding:8px;border:1px solid #ddd;border-radius:8px}
button{padding:10px 14px;border-radius:10px;border:1px solid #4f46e5;background:#4f46e5;color:#fff;cursor:pointer}
pre{white-space:pre-wrap;background:#0b1020;color:#d7e0ff;padding:12px;border-radius:8px;max-height:480px;overflow:auto}
.grid{display:grid;grid-template-columns:1fr 1fr; gap:12px}
</style></head>
<body>
<div class="wrap">
  <h1>Run GGUF via llama-cli</h1>
  <div class="card">
    <form id="f" onsubmit="start();return false;">
      <div class="grid">
        <label>Model (.gguf)
          <select id="model">
            {% for f in models %}<option value="{{f}}">{{f}}</option>{% endfor %}
          </select>
        </label>
        <label>llama-cli path (optional)
          <input id="llama_bin" placeholder="auto from PATH or local folder">
        </label>
      </div>
      <label>Prompt<textarea id="prompt" rows="4">Hello! Summarize yourself in one sentence.</textarea></label>
      <div class="grid">
        <label>n_predict<input id="n_predict" type="number" value="256"></label>
        <label>ctx<input id="ctx" type="number" value="2048"></label>
      </div>
      <div class="grid">
        <label>gpu_layers (-1=auto, 0=CPU)<input id="gpu_layers" type="number" value="-1"></label>
        <label>temperature<input id="temp" type="number" step="0.01" value="0.7"></label>
      </div>
      <div class="grid">
        <label>top_k<input id="topk" type="number" value="40"></label>
        <label>top_p<input id="topp" type="number" step="0.01" value="0.95"></label>
      </div>
      <div class="grid">
        <label>repeat_penalty<input id="rp" type="number" step="0.01" value="1.1"></label>
        <label>seed (-1=random)<input id="seed" type="number" value="-1"></label>
      </div>
      <button>Run</button>
    </form>
  </div>

  <div class="card">
    <h3>Output</h3>
    <pre id="log"></pre>
  </div>
</div>
<script>
(function(){
  const $ = (id)=>document.getElementById(id);
  function val(id){ return $(id).value; }

  window.start = function(){
    const m = val('model');
    if (!m) { alert('Pick a .gguf model'); return; }
    const p = val('prompt');
    if (!p.trim()) { alert('Enter a prompt'); return; }

    const qs = new URLSearchParams({
      model: m,
      prompt: p,
      llama_bin: val('llama_bin'),
      n_predict: val('n_predict'),
      ctx: val('ctx'),
      gpu_layers: val('gpu_layers'),
      temp: val('temp'),
      top_k: val('topk'),
      top_p: val('topp'),
      repeat_penalty: val('rp'),
      seed: val('seed')
    });

    const log = $('log'); log.textContent = '';
    const es = new EventSource('/run?' + qs.toString());
    es.onmessage = (e)=>{
      log.textContent += (e.data || '') + "\\n";
      log.scrollTop = log.scrollHeight;
    };
    es.onerror = ()=> es.close();
  };
})();
</script>
</body></html>
"""

app = Flask(__name__)

def _sse(line:str) -> str:
    return f"data: {line}\n\n"

@app.route("/")
def index():
    # ищем *.gguf в ./Models и в текущей папке
    models = [p.name for p in Path("Models").glob("*.gguf")] + [p.name for p in Path(".").glob("*.gguf")]
    models = sorted(list(dict.fromkeys(models)))
    return render_template_string(HTML, models=models)

@app.route("/run")
def run_route():
    model = request.args.get("model", "")
    prompt = request.args.get("prompt", "Hello")
    llama_bin = request.args.get("llama_bin", "").strip() or "llama-cli"
    n_predict = int(request.args.get("n_predict", 256))
    ctx = int(request.args.get("ctx", 2048))
    gpu_layers = int(request.args.get("gpu_layers", -1))
    temp = float(request.args.get("temp", 0.7))
    top_k = int(request.args.get("top_k", 40))
    top_p = float(request.args.get("top_p", 0.95))
    rp = float(request.args.get("repeat_penalty", 1.1))
    seed = int(request.args.get("seed", -1))

    # модель ищем сначала в Models/, затем в текущей папке
    mp = Path("Models") / model
    if not mp.exists():
        mp = Path(model)

    def stream():
        try:
            for line in run_llama_cli(
                model_path=str(mp),
                prompt=prompt,
                llama_bin=llama_bin,
                n_predict=n_predict,
                ctx=ctx,
                gpu_layers=gpu_layers,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=rp,
                seed=seed,
                # пример: можно подсказать CUDA, какую карту брать
                env=None  # или {"CUDA_VISIBLE_DEVICES": "0"}
            ):
                yield _sse(line)
        except Exception as e:
            yield _sse(f"ERROR: {type(e).__name__}: {e}")
        finally:
            yield _sse("DONE")

    return Response(stream_with_context(stream()), mimetype="text/event-stream")

if __name__ == "__main__":
    os.makedirs("Models", exist_ok=True)
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
