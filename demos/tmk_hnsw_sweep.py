#!/usr/bin/env python3
"""Run a small parameter sweep for demo_tmk_recall."""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import time
from pathlib import Path


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_output(text: str) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            if "." in value:
                out[key] = float(value)
            else:
                out[key] = int(value)
        except ValueError:
            out[key] = value
    return out


def write_reports(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return
    rows = sorted(
        rows,
        key=lambda r: (
            -float(r.get("recall@100", 0.0)),
            float(r.get("search_seconds", 0.0)),
            float(r.get("build_seconds", 0.0)),
        ),
    )
    csv_path = out_dir / "tmk_hnsw_sweep.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "offset_step",
        "M",
        "efConstruction",
        "efSearch",
        "recall@100",
        "query_hit_rate@100",
        "build_seconds",
        "search_seconds",
        "elapsed_seconds",
        "videos",
        "queries",
    ]
    fieldnames = [x for x in preferred if x in fieldnames] + [
        x for x in fieldnames if x not in preferred
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# TMK HNSW Sweep",
        "",
        "| rank | offset_step | M | efConstruction | efSearch | recall@100 | hit_rate@100 | build_s | search_s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        md_lines.append(
            f"| {rank} | {row['offset_step']} | {row['M']} | "
            f"{row['efConstruction']} | {row['efSearch']} | "
            f"{float(row.get('recall@100', 0.0)):.6f} | "
            f"{float(row.get('query_hit_rate@100', 0.0)):.6f} | "
            f"{float(row.get('build_seconds', 0.0)):.3f} | "
            f"{float(row.get('search_seconds', 0.0)):.3f} |"
        )
    (out_dir / "REPORT.md").write_text("\n".join(md_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/demos/demo_tmk_recall"))
    parser.add_argument("--signatures", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--periods", default="0,64,128,256,512,1024,2048,4096")
    parser.add_argument("--offset-min", type=int, default=-1200)
    parser.add_argument("--offset-max", type=int, default=1200)
    parser.add_argument("--offset-steps", default="40,20,10")
    parser.add_argument("--Ms", default="16,32")
    parser.add_argument("--ef-constructions", default="20,100,200")
    parser.add_argument("--ef-searches", default="64,128,256")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max-vectors", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    combos = list(
        itertools.product(
            parse_ints(args.offset_steps),
            parse_ints(args.Ms),
            parse_ints(args.ef_constructions),
            parse_ints(args.ef_searches),
        )
    )
    for idx, (offset_step, m, efc, efs) in enumerate(combos, 1):
        cmd = [
            str(args.binary),
            "--signatures",
            str(args.signatures),
            "--videos",
            str(args.videos),
            "--gt",
            str(args.gt),
            "--periods",
            args.periods,
            "--offset-min",
            str(args.offset_min),
            "--offset-max",
            str(args.offset_max),
            "--offset-step",
            str(offset_step),
            "--k",
            str(args.k),
            "--M",
            str(m),
            "--efConstruction",
            str(efc),
            "--efSearch",
            str(efs),
        ]
        if args.max_vectors:
            cmd += ["--max-vectors", str(args.max_vectors)]
        if args.max_queries:
            cmd += ["--max-queries", str(args.max_queries)]

        print(
            f"[{idx}/{len(combos)}] offset_step={offset_step} M={m} "
            f"efConstruction={efc} efSearch={efs}",
            flush=True,
        )
        started = time.time()
        proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
        elapsed = time.time() - started
        parsed = parse_output(proc.stdout)
        row = {
            "offset_step": offset_step,
            "M": m,
            "efConstruction": efc,
            "efSearch": efs,
            "elapsed_seconds": elapsed,
            **parsed,
        }
        rows.append(row)
        write_reports(rows, args.out_dir)
        print(
            f"  recall@{args.k}={row.get(f'recall@{args.k}')} "
            f"build={row.get('build_seconds')} search={row.get('search_seconds')}",
            flush=True,
        )

    write_reports(rows, args.out_dir)
    print(f"wrote {args.out_dir / 'tmk_hnsw_sweep.csv'}")


if __name__ == "__main__":
    main()
