"""Build the approximate local cache for training or inference (no downloads)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing study/series/DICOM")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--studies", type=Path, help="Optional JSON array of study UIDs")
    parser.add_argument("--anchors", type=Path, help="JSON: study -> series -> joint_lps [x,y,z] mm")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--mm-per-px", type=float, default=0.6)
    parser.add_argument("--skip-invalid-series", action="store_true",
                        help="Record invalid series and continue; never skip a whole study")
    args = parser.parse_args()
    from src.local_preprocessing import Config, prepare_cache
    studies = json.loads(args.studies.read_text()) if args.studies else None
    anchors = json.loads(args.anchors.read_text()) if args.anchors else None
    index = prepare_cache(args.data, args.output, Config(size=args.size, mm_per_px=args.mm_per_px),
                          studies, anchors, args.skip_invalid_series)
    print(json.dumps({"cache": str(args.output), "studies": len(index["done"]),
                      "slice_rows": sum(v[1] for v in index["done"].values()), "approximate": True}))


if __name__ == "__main__":
    main()
