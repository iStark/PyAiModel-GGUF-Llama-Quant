#!/usr/bin/env python3
# llama_runner.py — thin wrapper to run llama.cpp binaries (llama-cli/llama-run) from Python.
# Works on Windows (PowerShell/CMD) and Linux/macOS. Streams stdout line-by-line.

import os, sys, subprocess, shlex
from pathlib import Path
from typing import Iterable, Optional

def which(p: str) -> Optional[str]:
    from shutil import which as _which
    hit = _which(p)
    return hit if hit else (str(Path(p)) if Path(p).exists() else None)

def run_llama_cli(
    model_path: str,
    prompt: str,
    llama_bin: str = "llama-cli",
    n_predict: int = 256,
    ctx: int = 2048,
    gpu_layers: int = -1,          # -1: auto, 0: CPU-only
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.95,
    repeat_penalty: float = 1.1,
    seed: int = -1,                 # -1: random
    extra_args: Optional[list[str]] = None,
    env: Optional[dict] = None,
) -> Iterable[str]:
    """
    Yields lines from llama-cli as they arrive.
    Requires llama-cli.exe (Windows) or llama-cli in PATH.
    If compiled with cuBLAS, will use ggml-cuda.dll automatically.
    """
    bin_path = which(llama_bin)
    if not bin_path:
        raise FileNotFoundError(
            f"Cannot find {llama_bin}. Put it in PATH or pass full path via llama_bin."
        )
    model_path = str(Path(model_path).resolve())
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    cmd = [
        bin_path,
        "-m", model_path,
        "-p", prompt,
        "--n-predict", str(n_predict),
        "-c", str(ctx),
        "--temp", str(temperature),
        "--top-k", str(top_k),
        "--top-p", str(top_p),
        "--repeat-penalty", str(repeat_penalty),
        "--seed", str(seed),
    ]
    if gpu_layers is not None:
        cmd += ["-ngl", str(gpu_layers)]

    if extra_args:
        cmd += list(extra_args)

    # Окружение: можно подсказать CUDA рантайму, какие GPU использовать
    env_final = os.environ.copy()
    if env:
        env_final.update(env)

    # universal_newlines=True == text mode; bufsize=1 + line-buffered
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace",
        env=env_final
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            yield line.rstrip("\n")
    finally:
        proc.stdout.close()
        proc.wait()
