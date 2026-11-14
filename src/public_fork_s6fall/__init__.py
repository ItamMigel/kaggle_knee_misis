"""Port of public notebook nartaa/rsna-knee-s6fall-solo v2 (LB 0.950; v1 scored the same files from a private bundle).

`vendored/` holds the kernel source byte-exact (`.txt` suffix, so the index builder skips it; sha256 in sha256.json);
`pipeline.py` (`run_s6fall`) runs it unchanged as its own process; `k1_driver.py` runs only its component B over a
fake Kaggle root (throughput measurement on train studies).
"""
