"""exp-0003 extract.py — Kaggle kernel body: rubric evidence extraction from reports with vLLM.

Launch (T4×2, internet on for pip + HF weights; S4/P only for the pilot):

    uv run --project kaggle python kaggle/run_kernel.py --exp exp-0003 --script extract \
        --accelerator t4 --internet \
        [--ds stevenleehans/rsna-knee-llm-report-labels,pilkwang/rsna-knee-llm-labels]

Kernel output files cannot be downloaded, so everything that matters also goes
through the log via src/log_transport.py: selection.csv and env.json up front,
raw.part-NNNN.jsonl after every batch (survives a kill mid-run), stats.json at
the end, then the optional probe.bin. Decode locally with decode_log.py.
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

# Only stdlib at module level: pip install below may replace numpy/torch, and vLLM
# worker processes re-import this file (spawn) — the work lives in main().
import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402

from src.log_transport import emit_file  # noqa: E402
from src.report_labels import llm  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

COMP = "rsna-knee-abnormality-detection"
S4_DS, S4_FILE = "stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv"
P_DS, P_FILE = "pilkwang/rsna-knee-llm-labels", "report_labels_v2.csv"

# HF commit shas (HF API, 2026-10-17; all three last modified 2025-05-21):
#   Qwen/Qwen3-32B-AWQ  0499c3ac83fdef8810b907a23894ba91e95eddd8
#   Qwen/Qwen3-14B-AWQ  31c69efc29464b6bb0aee1398b5a7b50a99340c3
#   Qwen/Qwen3-4B-AWQ   74d4bd2bd4bff9cafc9345221320bffb08b406a3
MODEL_REPO = "Qwen/Qwen3-14B-AWQ"
MODEL_REVISION = "31c69efc29464b6bb0aee1398b5a7b50a99340c3"
VLLM_VERSION = "0.31.0"

SUBSET = "pilot_gold"     # "smoke" | "pilot" | "pilot_gold" (pilot + the 58 gold studies) | "shard"
SMOKE_N = 8               # one non-gold report per language, first 8 of pilot.LANGUAGES
PILOT = {"n_per_language": 10, "n_bilateral": 20, "n_disagreement": 6}  # 90 + 20 + 48 = 158
SHARD = (0, 1)            # (k, n): part k of n over all 4,407 studies, gold included

# T4 is sm75: fp16 only (extract() fixes dtype), AWQ via the non-Marlin kernel
# (Marlin needs sm80). Compact JSON: xgrammar's whitespace switch is engine-level.
ENGINE_KWARGS = {
    "tensor_parallel_size": 2,
    "quantization": "awq",
    "max_model_len": 6144,  # longest corpus prompt 4,377 tok (Qwen3 tokenizer) + MAX_TOKENS 1600
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 48,     # 14B TP2 on T4: ~5 GiB weights/GPU leave ~8 GiB KV/GPU ≈ 100k tokens
    "structured_outputs_config": {"backend": "xgrammar", "disable_any_whitespace": True},
}
MAX_TOKENS = 1600         # truncated ("length") outputs become review cases downstream
BATCH_SIZE = 192          # studies per llm.chat call = per flushed/emitted part (4x max_num_seqs)
TIME_BUDGET_S = 4000      # from engine start; kernel limit is 9 h
SEED = 20261017           # pilot/smoke order (sha256(seed:uid)); decoding is greedy
LOG_PROBE_BYTES = 0  # smoke only: random blob emitted last to measure the log limit
CHUNK_CHARS = 3800        # base64 chars per log line (lines stay < 4,000 chars)
FAIL_FAST = True          # stop after part 0 if it fails the validity gate below
FAIL_FAST_MIN_CELL_VALIDITY = 0.80  # part 0 also fails on any record-level error


def emit_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    emit_file(path, path.name, CHUNK_CHARS)


def main() -> None:
    if LOG_PROBE_BYTES and SUBSET != "smoke":
        raise SystemExit("LOG_PROBE_BYTES > 0 is for the smoke run only")
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    ctx = get_ctx()
    out = ctx.output_dir
    vllm_version = llm.install_vllm(VLLM_VERSION)

    import pandas as pd

    from src.report_labels.pilot import detect_language, select_pilot, select_shard, select_smoke
    from src.report_labels.rubric import TARGETS

    train = pd.read_csv(ctx.mounts[COMP] / "train.csv")
    if SUBSET == "smoke":
        selection = select_smoke(train, SMOKE_N, SEED)
    elif SUBSET in ("pilot", "pilot_gold"):
        s4 = pd.read_csv(ctx.mounts[S4_DS] / S4_FILE)
        p = pd.read_csv(ctx.mounts[P_DS] / P_FILE)
        selection = select_pilot(train, s4, p, seed=SEED, **PILOT)
        if SUBSET == "pilot_gold":  # one-time gold58 go/no-go diagnostic, lead 17.10
            # Label NaN mask only, as compare.split_gold (not imported: it pulls sklearn/scipy,
            # which may not survive the numpy that the vLLM install brings).
            gold = train.loc[train[list(TARGETS)].notna().all(axis=1), ["StudyInstanceUID", "Report"]]
            selection = pd.concat([selection, pd.DataFrame({
                "StudyInstanceUID": gold["StudyInstanceUID"].to_numpy(), "reason": "gold",
                "language": gold["Report"].map(detect_language).to_numpy()})], ignore_index=True)
    elif SUBSET == "shard":
        selection = select_shard(train, *SHARD)
    else:
        raise SystemExit(f"unknown SUBSET={SUBSET!r}")
    selection.to_csv(out / "selection.csv", index=False)
    emit_file(out / "selection.csv", "selection.csv", CHUNK_CHARS)
    print(f"[extract] {SUBSET}: {len(selection)} studies | "
          f"{selection['reason'].value_counts().to_dict()}", flush=True)

    emit_json(llm.runtime_env(), out / "env.json")
    model_dir = llm.download_model(MODEL_REPO, MODEL_REVISION,
                                   Path(tempfile.mkdtemp(prefix="hf-model-")))

    reports = (train.set_index("StudyInstanceUID")
               .loc[selection["StudyInstanceUID"], ["Report"]].reset_index())

    def on_batch(index: int, lines: list[str]) -> None:
        part = out / f"raw.part-{index:04d}.jsonl"
        part.write_text("\n".join(lines) + "\n", encoding="utf-8")
        emit_file(part, part.name, CHUNK_CHARS)
        if index != 0:
            return
        from src.report_labels.validate import check_batch

        check = check_batch(lines, dict(zip(reports["StudyInstanceUID"], reports["Report"])))
        print(f"[extract] part 0 check: {json.dumps(check, ensure_ascii=False)}", flush=True)
        if FAIL_FAST and (check["n_record_invalid"] > 0
                          or check["cell_valid_share"] < FAIL_FAST_MIN_CELL_VALIDITY):
            # RuntimeError, not SystemExit: SystemExit bypasses ctx's excepthook, and the
            # record of a deliberately stopped run would say status "ok".
            raise RuntimeError(
                f"FAIL_FAST: part 0 failed the validity gate: {check['n_record_invalid']} record-level "
                f"invalid of {check['n']}, cell_valid_share {check['cell_valid_share']:.3f} "
                f"(min {FAIL_FAST_MIN_CELL_VALIDITY}). raw.part-0000.jsonl is already in the log: decode it "
                f"with decode_log.py, fix the prompt/schema and launch a NEW kernel.")

    stats = llm.extract(reports, model_dir, MODEL_REPO, MODEL_REVISION, ENGINE_KWARGS, MAX_TOKENS,
                        out / "raw.jsonl", TIME_BUDGET_S, BATCH_SIZE, on_batch)
    stats.update({"subset": SUBSET, "shard": list(SHARD) if SUBSET == "shard" else None,
                  "vllm": vllm_version, "seed": SEED})
    emit_json(stats, out / "stats.json")
    print(f"[extract] done {stats['n_done']}/{stats['n_total']}, skipped {stats['n_skipped']} | "
          f"{stats['req_per_s']} req/s, {stats['output_tok_per_s']} out tok/s | "
          f"finish {stats['finish_reasons']}", flush=True)

    if LOG_PROBE_BYTES:
        probe = out / "probe.bin"
        probe.write_bytes(os.urandom(LOG_PROBE_BYTES))
        emit_file(probe, probe.name, CHUNK_CHARS)
        print(f"[extract] probe.bin emitted: {LOG_PROBE_BYTES} bytes", flush=True)


if __name__ == "__main__":
    main()
