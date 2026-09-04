"""Build a deterministic multilingual curriculum JSONL for TinyGPT.

This creates repeated practice examples from a small hand-reviewed seed. It is
intentionally finite and transparent; it is not a substitute for a licensed
corpus of complete English, Arabic, or French.

Usage:
    python scripts/build_curriculum.py --copies 20
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "curated" / "seed_dialogues.jsonl"
OUT = ROOT / "data" / "curated" / "curriculum.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=1)
    args = parser.parse_args()
    if args.copies < 1:
        raise SystemExit("--copies must be at least 1")
    rows = [json.loads(line) for line in SEED.read_text().splitlines()
            if line.strip()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for copy in range(args.copies):
            for row in rows:
                item = dict(row)
                item["round"] = copy + 1
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows) * args.copies} curriculum records to {OUT}")


if __name__ == "__main__":
    main()
