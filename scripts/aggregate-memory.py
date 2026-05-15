#!/usr/bin/env python3
"""Aggregate per-phase memory JSONL files into a single markdown report.

Usage:
    python3 scripts/aggregate-memory.py measurements/<date>/ > MEMORY_BASELINE_<date>.md

Reads every *.jsonl in the given directory, treats each file's stem as a
variant label, and emits two markdown tables: a per-phase grid and a
peaks-across-the-run summary.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

PHASE_ORDER = [
    "before-download",
    "after-loadArrays-mimi",
    "after-update-mimi",
    "after-loadArrays-moshi",
    "after-unflatten-moshi",
    "after-quantize-moshi",
    "after-update-moshi",
    "after-eval-moshi",
    "after-loadVocab",
    "after-warmup-mimi",
    "after-warmup-moshi",
    "after-step-100",
]


def fmt_bytes(n: int | None) -> str:
    if n is None or n < 0:
        return "—"
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


def load_run(path: str) -> tuple[list[dict], dict[str, list[dict]]]:
    snaps: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            snaps.append(json.loads(line))
    by_label: dict[str, list[dict]] = {}
    for s in snaps:
        by_label.setdefault(s["label"], []).append(s)
    return snaps, by_label


def main(directory: str) -> None:
    paths = sorted(glob.glob(f"{directory.rstrip('/')}/*.jsonl"))
    if not paths:
        print(f"No .jsonl files under {directory}", file=sys.stderr)
        sys.exit(1)

    runs = [(Path(p).stem, *load_run(p)) for p in paths]

    print("## Per-phase resident memory\n")
    header = ["Phase"] + [name for name, _, _ in runs]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")
    for label in PHASE_ORDER:
        row = [label]
        for _name, _snaps, by_label in runs:
            xs = by_label.get(label, [])
            row.append(fmt_bytes(max(s["residentBytes"] for s in xs)) if xs else "—")
        print("| " + " | ".join(row) + " |")
    print()

    print("## Peaks across the whole run\n")
    print(
        "| Variant | Peak resident | Peak MLX active | Peak MLX cache | "
        "MLX peak | KV cache @ step 100 |"
    )
    print("| " + " | ".join("---" for _ in range(6)) + " |")
    for name, snaps, by_label in runs:
        if not snaps:
            print(f"| {name} | (no data) | | | | |")
            continue
        peak_r = max(s["residentBytes"] for s in snaps)
        peak_a = max(s["mlxActiveBytes"] for s in snaps)
        peak_c = max(s["mlxCacheBytes"] for s in snaps)
        peak_p = max(s["mlxPeakBytes"] for s in snaps)
        kv_entries = by_label.get("after-step-100", [])
        kv = kv_entries[0].get("kvCacheBytes") if kv_entries else None
        print(
            f"| {name} | {fmt_bytes(peak_r)} | {fmt_bytes(peak_a)} | "
            f"{fmt_bytes(peak_c)} | {fmt_bytes(peak_p)} | {fmt_bytes(kv)} |"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
