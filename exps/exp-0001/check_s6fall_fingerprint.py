"""Fingerprint of a submit_s6fall.py run vs the scored S6fall v2 commit (3 public test studies).

Usage: python exps/exp-0001/check_s6fall_fingerprint.py exps/exp-0001/runs/<run_id>/log.txt
Passes when the log's K1_DONE submission_sha256 equals the v2 commit's and K1 confirmed component A against the 0.949
commit csv (K1 hard-fails otherwise, so a K1_DONE line implies it). Exit 1 with the reason otherwise.
"""

import json
import re
import sys

V2_SUBMISSION_SHA256 = "00031c5337550d13acaf2e8dbf8743e260169d93da64296d479f2c0fb0c0d6b0"   # nartaa/rsna-knee-s6fall-solo v2 log
V2_STAGES = ("env", "inputs", "B_probe", "A", "B", "blend")


def main():
    text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    done = re.findall(r"K1_DONE (\{.*\})", text)
    if len(done) != 1:
        sys.exit("FAIL: %d K1_DONE lines (K1 hard error or truncated log?)" % len(done))
    d = json.loads(done[0])
    problems = []
    if d.get("mode") != "submit" or d.get("rows") != 3:
        problems.append("mode/rows %s/%s, expected submit/3" % (d.get("mode"), d.get("rows")))
    if tuple(d.get("stages", {})) != V2_STAGES:
        problems.append("stages %s, expected %s" % (list(d.get("stages", {})), list(V2_STAGES)))
    if d.get("submission_sha256") != V2_SUBMISSION_SHA256:
        problems.append("submission sha256 %s != v2 %s" % (d.get("submission_sha256"), V2_SUBMISSION_SHA256))
    if "public_csv_matches_0949_commit" in text and "differs from the 0.949 commit csv" in text:
        problems.append("component A differs from the 0.949 commit csv")
    if problems:
        sys.exit("FAIL: " + "; ".join(problems))
    print("PASS: S6fall submission.csv sha256 == v2 commit (%s...), stages %s" % (V2_SUBMISSION_SHA256[:16], d["stages"]))


if __name__ == "__main__":
    main()
