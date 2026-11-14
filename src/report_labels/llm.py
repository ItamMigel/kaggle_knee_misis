"""vLLM side of the report extractor (Kaggle T4×2): install, weights, batched extraction.

torch/vllm/huggingface_hub are imported inside the functions: the module must
import before `install_vllm` has put them in place (and locally, where vLLM is
not installed). Checked against the vLLM v0.31.0 sources: `LLM.chat(...,
use_tqdm=, chat_template_kwargs=)`, `SamplingParams(structured_outputs=
StructuredOutputsParams(json=...))`, `RequestOutput.prompt_token_ids /
.num_cached_tokens / .outputs[0].{text, token_ids, finish_reason}`. No parsing
here: raw model text goes to JSONL, validation is src/report_labels/validate.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from src.report_labels.rubric import EXTRACTION_SCHEMA, PROMPT_VERSION, build_messages

# Only what vLLM needs to load a checkpoint; README/LICENSE/.gitattributes stay behind.
if TYPE_CHECKING:  # pip may replace numpy/pandas: nothing heavy imported before install_vllm
    import pandas as pd

MODEL_FILE_PATTERNS = ("*.safetensors", "*.json", "tokenizer*", "vocab.json", "merges.txt")
# extract() owns these engine arguments; the script must not pass them.
_FIXED_ENGINE_KWARGS = {"model", "dtype", "enable_prefix_caching"}

_ENV_PROBE = r"""
import json, torch, transformers, vllm
print(json.dumps({
    "python": __import__("sys").version.split()[0],
    "vllm": vllm.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpus": [{"name": torch.cuda.get_device_name(i),
              "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
              "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)}
             for i in range(torch.cuda.device_count())],
}))
"""


def _run(cmd: list[str], what: str) -> str:
    """Run a command, return stdout; on failure print its output tail and raise."""
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[llm] {what} failed (exit {res.returncode}):\n"
              f"--- stdout tail ---\n{res.stdout[-3000:]}\n--- stderr tail ---\n{res.stderr[-3000:]}",
              flush=True)
        raise RuntimeError(f"{what} failed with exit code {res.returncode}: {' '.join(cmd)}")
    return res.stdout


def runtime_env() -> dict:
    """Versions of python/vllm/torch/transformers, CUDA and GPUs (fresh subprocess) + nvidia-smi driver."""
    env = json.loads(_run([sys.executable, "-c", _ENV_PROBE], "runtime probe").strip().splitlines()[-1])
    env["nvidia_smi"] = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        "nvidia-smi").strip().splitlines()
    return env


def install_vllm(version: str) -> str:
    """pip-install the pinned vLLM, drop torchaudio, verify torch+CUDA+vllm in a fresh process."""
    _run([sys.executable, "-m", "pip", "install", "--quiet", "--no-input", f"vllm=={version}"],
         f"pip install vllm=={version}")
    # torchaudio from the image is built against the old torch and breaks `import transformers`.
    _run([sys.executable, "-m", "pip", "uninstall", "-y", "--quiet", "torchaudio"],
         "pip uninstall torchaudio")
    env = runtime_env()
    if env["vllm"] != version:
        raise RuntimeError(f"asked for vllm=={version}, the fresh interpreter imports {env['vllm']}")
    if not env["cuda_available"] or not env["gpus"]:
        raise RuntimeError(f"vllm {version} installed but torch sees no CUDA device: {env}")
    print(f"[llm] vllm {env['vllm']} | torch {env['torch']} (cuda {env['torch_cuda']}) | "
          f"{', '.join(g['name'] + ' sm' + g['capability'] for g in env['gpus'])}", flush=True)
    return env["vllm"]


def download_model(repo_id: str, revision: str, dst: Path) -> Path:
    """HF snapshot of a pinned commit (weights/config/tokenizer only) into dst; returns dst."""
    from huggingface_hub import HfApi, snapshot_download

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"revision must be a full 40-hex commit sha, got {revision!r}")
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"{repo_id}@{revision}: hub resolved commit {info.sha}")
    wanted = [s for s in info.siblings if any(fnmatch(s.rfilename, p) for p in MODEL_FILE_PATTERNS)]
    need = sum(s.size for s in wanted)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dst).free
    if free < need * 1.05:
        raise RuntimeError(f"{repo_id}: needs {need / 2**30:.1f} GiB, {dst} has {free / 2**30:.1f} GiB free")
    t0 = time.monotonic()
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=dst,
                      allow_patterns=list(MODEL_FILE_PATTERNS))
    for s in wanted:
        got = dst / s.rfilename
        if not got.is_file() or got.stat().st_size != s.size:
            raise RuntimeError(f"{repo_id}@{revision}: {s.rfilename} missing or wrong size in {dst}")
    print(f"[llm] {repo_id}@{revision[:12]}: {len(wanted)} files, {need / 2**30:.2f} GiB "
          f"in {time.monotonic() - t0:.0f}s", flush=True)
    return dst


def extract(reports: pd.DataFrame, model_dir: Path, model_id: str, revision: str,
            engine_kwargs: dict, max_tokens: int, out_jsonl: Path, time_budget_s: float,
            batch_size: int, on_batch: Callable[[int, list[str]], None]) -> dict:
    """Batched vLLM extraction (fp16, prefix cache, non-thinking, JSON schema) → JSONL + stats.

    reports: StudyInstanceUID + Report. One JSONL line per study, appended and
    fsynced after every batch, then `on_batch(batch_index, lines)` (0-based).
    time_budget_s counts from the call (engine load included): no new batch
    starts if the slowest batch so far would overrun it; the rest is returned
    as skipped. A prompt that cannot fit max_model_len fails before any decode.
    """
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    t0 = time.monotonic()
    missing_cols = {"StudyInstanceUID", "Report"} - set(reports.columns)
    if missing_cols:
        raise ValueError(f"reports lacks columns {sorted(missing_cols)}")
    if reports["StudyInstanceUID"].duplicated().any():
        raise ValueError("reports has duplicated StudyInstanceUID")
    clash = _FIXED_ENGINE_KWARGS & set(engine_kwargs)
    if clash:
        raise ValueError(f"engine_kwargs must not set {sorted(clash)}: extract() fixes them")
    if "max_model_len" not in engine_kwargs:
        raise ValueError("engine_kwargs must set max_model_len explicitly")
    out_jsonl = Path(out_jsonl)
    if out_jsonl.exists():
        raise FileExistsError(f"{out_jsonl} already exists; refusing to append to an old run")

    uids = reports["StudyInstanceUID"].tolist()
    conversations = [build_messages(r) for r in reports["Report"]]
    llm = LLM(model=str(model_dir), dtype="float16", enable_prefix_caching=True, **engine_kwargs)
    load_s = time.monotonic() - t0

    # Same rendering as llm.chat; vLLM itself would raise mid-batch on an oversized prompt.
    tok = llm.get_tokenizer()
    max_len = engine_kwargs["max_model_len"]
    prompt_len = [
        len(tok.encode(tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False), add_special_tokens=False))
        for c in conversations
    ]
    too_long = [(u, n) for u, n in zip(uids, prompt_len) if n >= max_len]
    if too_long:
        raise ValueError(f"{len(too_long)} prompt(s) do not fit max_model_len={max_len} "
                         f"(raise it): {too_long[:10]}")
    n_capped = sum(n + max_tokens > max_len for n in prompt_len)
    print(f"[llm] engine up in {load_s:.0f}s | {len(uids)} prompts, tokens max {max(prompt_len)} "
          f"mean {sum(prompt_len) / len(prompt_len):.0f} | output cap below max_tokens for {n_capped}",
          flush=True)

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            structured_outputs=StructuredOutputsParams(json=EXTRACTION_SCHEMA))
    done, finish, out_tok, in_tok, cached, batch_s = 0, Counter(), [], [], [], []
    gen_t0 = time.monotonic()
    skipped: list[str] = []
    for b, start in enumerate(range(0, len(uids), batch_size)):
        elapsed = time.monotonic() - t0
        if batch_s and elapsed + max(batch_s) > time_budget_s:
            skipped = uids[start:]
            print(f"[llm] time budget: {elapsed:.0f}s used of {time_budget_s:.0f}s, slowest batch "
                  f"{max(batch_s):.0f}s → stop, {len(skipped)} studies skipped", flush=True)
            break
        bt0 = time.monotonic()
        outs = llm.chat(conversations[start: start + batch_size], params, use_tqdm=False,
                        chat_template_kwargs={"enable_thinking": False})
        lines = []
        for uid, o in zip(uids[start: start + batch_size], outs):
            c = o.outputs[0]
            lines.append(json.dumps({
                "StudyInstanceUID": uid, "raw": c.text, "finish_reason": c.finish_reason,
                "prompt_tokens": len(o.prompt_token_ids), "output_tokens": len(c.token_ids),
                "model": model_id, "revision": revision, "prompt_version": PROMPT_VERSION,
                "pass": "base",
            }, ensure_ascii=False))
            finish[c.finish_reason] += 1
            in_tok.append(len(o.prompt_token_ids))
            out_tok.append(len(c.token_ids))
            cached.append(o.num_cached_tokens)
        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()
            os.fsync(f.fileno())
        done += len(lines)
        batch_s.append(time.monotonic() - bt0)
        print(f"[llm] batch {b}: {len(lines)} done ({done}/{len(uids)}) in {batch_s[-1]:.0f}s, "
              f"finish {dict(finish)}", flush=True)
        on_batch(b, lines)

    gen_s = time.monotonic() - gen_t0
    return {
        "n_total": len(uids), "n_done": done, "n_skipped": len(skipped), "skipped_uids": skipped,
        "n_batches": len(batch_s), "batch_size": batch_size,
        "wall_s": round(time.monotonic() - t0, 1), "load_s": round(load_s, 1),
        "gen_s": round(gen_s, 1), "req_per_s": round(done / gen_s, 4) if done else 0.0,
        "output_tok_per_s": round(sum(out_tok) / gen_s, 1) if done else 0.0,
        "mean_prompt_tokens": round(sum(in_tok) / done, 1) if done else None,
        "mean_output_tokens": round(sum(out_tok) / done, 1) if done else None,
        "max_output_tokens": max(out_tok) if done else None,
        "mean_cached_prompt_tokens": (round(sum(cached) / done, 1)
                                      if done and None not in cached else None),
        "max_prompt_tokens_all": max(prompt_len), "n_output_capped_by_context": n_capped,
        "finish_reasons": dict(finish), "max_tokens": max_tokens,
        "engine_kwargs": engine_kwargs, "model": model_id, "revision": revision,
        "prompt_version": PROMPT_VERSION,
    }
