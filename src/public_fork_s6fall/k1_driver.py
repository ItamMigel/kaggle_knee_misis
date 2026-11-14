"""S6fall's component B alone, through K1's own functions: prepare_inputs -> probe_b_stack -> run_b (throughput measurement).

Run as its own process: python -u k1_driver.py <K1 script> <budget seconds> <out json>. K1_KAGGLE_ROOT and K1_TMP must be
set: K1 reads them at import, and without them it would read the real /kaggle root. The K1 file is imported, never edited;
its main() (component A, blend, hang guard, submission.csv) is not called. Stdlib only (runs inside kernels).
"""

import importlib.util
import json
import os
import sys
import time


def main():
    k1_path, budget_s, out_json = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    for key in ("K1_KAGGLE_ROOT", "K1_TMP"):
        if not os.environ.get(key):
            raise SystemExit("k1_driver: %s is not set; K1 would read the real /kaggle root" % key)
    spec = importlib.util.spec_from_file_location("k1_s6fall", k1_path)
    k1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k1)
    k1.OUT.mkdir(parents=True, exist_ok=True)
    k1.TMP.mkdir(parents=True, exist_ok=True)
    seconds = {}
    t = time.time()
    k1._RECEIPT["env"] = k1.probe_env()          # run_b reads n_gpu from here
    seconds["env"] = time.time() - t
    t = time.time()
    inputs = k1.prepare_inputs()                 # sha checks of the model, anatomy.pt and every code file; I.cost from test_series.csv
    seconds["inputs"] = time.time() - t
    t = time.time()
    k1.probe_b_stack(inputs)
    seconds["b_probe"] = time.time() - t
    t = time.time()
    B, _, info, failed = k1.run_b(inputs, "B", inputs.test_ids, "test", budget_s)
    seconds["b"] = time.time() - t
    solo_sha = None
    if not failed:   # the submission.csv K1 v2 writes from this B: rank(B) (W_A 0, so A's values never enter without a B failure)
        k1.WORK = k1.TMP / "solo"                # write_submission writes WORK/submission.csv: keep it off /kaggle/working
        k1.WORK.mkdir(parents=True, exist_ok=True)
        solo_sha = k1.sha256_file(k1.write_submission(inputs.test_ids, inputs.sub_cols, k1.rank_blend(B, B)))
    out = {"seconds": seconds, "studies": len(inputs.test_ids), "parts": info["parts"], "failed_after_retry": failed,
           "retry": None if info["retry"] is None else len(info["retry"]["ids"]), "fail_cap": info["fail_cap"],
           "probs_csv": str(k1.OUT / "B" / "B_probs.csv"), "probs_sha256": info["probs_sha256"], "k1_solo_sha256": solo_sha,
           "receipts": info["receipts"]}
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("K1_DRIVER_DONE", json.dumps({k: round(v, 1) for k, v in seconds.items()}), flush=True)


if __name__ == "__main__":
    main()
