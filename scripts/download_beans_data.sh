#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${TOKENIZER_IMAGE_DIR:-$ROOT_DIR/data/images/beans}"
N_IMAGES="${TOKENIZER_N_IMAGES:-64}"
SPLIT="${BEANS_SPLIT:-train}"

mkdir -p "$OUT_DIR"

uv run python - "$OUT_DIR" "$N_IMAGES" "$SPLIT" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

from datasets import load_dataset


out_dir = Path(sys.argv[1]).expanduser()
n_images = int(sys.argv[2])
split = sys.argv[3]

out_dir.mkdir(parents=True, exist_ok=True)
dataset = load_dataset("beans", split=f"{split}[:{n_images}]")
label_names = dataset.features["labels"].names

rows = []
for dataset_index, item in enumerate(dataset):
    label = label_names[int(item["labels"])]
    filename = f"{dataset_index:04d}_{label}.jpg"
    image_path = out_dir / filename
    item["image"].convert("RGB").save(image_path, quality=95)
    rows.append(
        {
            "dataset": "beans",
            "dataset_split": split,
            "dataset_index": dataset_index,
            "path": filename,
            "label": label,
        }
    )

metadata_path = out_dir / "metadata.csv"
with metadata_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} images to {out_dir}")
print(f"Wrote metadata to {metadata_path}")
PY
