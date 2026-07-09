#!/usr/bin/env python3
"""Evaluate exact TMK ranks for pair-level GT.

This is an upper-bound diagnostic for HNSW + TMK: if exact TMK cannot rank the
GT target highly, graph-search parameter sweeps cannot fix the metric issue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_signatures(signatures: Path) -> np.ndarray:
    if signatures.suffix == ".npy":
        return np.load(signatures, mmap_mode="r")
    raise ValueError("only .npy signatures are supported")


def summarize(df: pd.DataFrame, ks: list[int]) -> dict:
    out: dict[str, float | int | None] = {"queries": int(len(df))}
    valid = df[df["target_rank"].notna()].copy()
    out["valid_queries"] = int(len(valid))
    if valid.empty:
        return out
    for k in ks:
        out[f"recall@{k}"] = float((valid["target_rank"] <= k).mean())
    out.update(
        {
            "rank_p50": float(valid["target_rank"].quantile(0.50)),
            "rank_p90": float(valid["target_rank"].quantile(0.90)),
            "rank_p99": float(valid["target_rank"].quantile(0.99)),
            "rank_max": int(valid["target_rank"].max()),
            "target_score_mean": float(valid["target_score"].mean()),
            "top1_score_mean": float(valid["top1_score"].mean()),
        }
    )
    if "coverage_min" in valid:
        out["coverage_min_p50"] = float(valid["coverage_min"].quantile(0.50))
        out["coverage_min_mean"] = float(valid["coverage_min"].mean())
    return out


def aggregate_table(df: pd.DataFrame, group_col: str, ks: list[int]) -> pd.DataFrame:
    rows = []
    valid = df[df["target_rank"].notna()].copy()
    for key, group in valid.groupby(group_col, dropna=False, observed=True):
        row = {group_col: str(key), "queries": len(group)}
        for k in ks:
            row[f"recall@{k}"] = float((group["target_rank"] <= k).mean())
        row.update(
            {
                "rank_p50": float(group["target_rank"].quantile(0.50)),
                "rank_p90": float(group["target_rank"].quantile(0.90)),
                "rank_mean": float(group["target_rank"].mean()),
                "target_score_mean": float(group["target_score"].mean()),
                "top1_score_mean": float(group["top1_score"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(
    out_dir: Path,
    args: argparse.Namespace,
    summary: dict,
    by_class: pd.DataFrame,
    by_direction: pd.DataFrame,
    by_coverage: pd.DataFrame,
) -> None:
    def markdown_table(table: pd.DataFrame) -> str:
        if table.empty:
            return "(empty)"
        cols = list(table.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for row in table.itertuples(index=False):
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f"{value:.6g}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        "# Exact TMK Rank Eval",
        "",
        "## Config",
        "",
        f"- signatures: `{args.signatures}`",
        f"- videos: `{args.videos}`",
        f"- gt: `{args.gt}`",
        f"- periods: `{args.periods}`",
        f"- offset_min: `{args.offset_min}`",
        f"- offset_max: `{args.offset_max}`",
        f"- offset_step: `{args.offset_step}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")

    def add_table(title: str, table: pd.DataFrame) -> None:
        lines.extend(["", f"## {title}", ""])
        lines.append(markdown_table(table))

    add_table("By Class", by_class)
    add_table("By Direction", by_direction)
    add_table("By Coverage", by_coverage)
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signatures", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--periods", default="0,64,128,256,512,1024,2048,4096")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--offset-min", type=int, default=-1200)
    parser.add_argument("--offset-max", type=int, default=1200)
    parser.add_argument("--offset-step", type=int, default=40)
    parser.add_argument("--ks", default="10,20,100")
    parser.add_argument("--sample-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ks = parse_ints(args.ks)
    periods = np.asarray(parse_floats(args.periods), dtype=np.float32)
    offsets = np.arange(args.offset_min, args.offset_max + args.offset_step, args.offset_step, dtype=np.float32)
    omegas = np.zeros_like(periods)
    non_dc = periods > 0
    omegas[non_dc] = (2.0 * np.pi / periods[non_dc]).astype(np.float32)
    cos_table = np.cos(omegas[:, None] * offsets[None, :]).astype(np.float32)
    sin_table = np.sin(omegas[:, None] * offsets[None, :]).astype(np.float32)

    x = load_signatures(args.signatures)
    n, flat_dim = x.shape
    f_count = len(periods)
    expected_dim = 2 * f_count * args.embedding_dim
    if flat_dim != expected_dim:
        raise ValueError(f"signature dim mismatch: got {flat_dim}, expected {expected_dim}")

    xb = np.asarray(x[:, : f_count * args.embedding_dim].reshape(n, f_count, args.embedding_dim), dtype=np.float32)
    xa = np.asarray(x[:, f_count * args.embedding_dim :].reshape(n, f_count, args.embedding_dim), dtype=np.float32)
    videos = pd.read_csv(args.videos)
    gt = pd.read_csv(args.gt)
    if args.sample_pairs > 0 and len(gt) > args.sample_pairs:
        gt = gt.sample(n=args.sample_pairs, random_state=args.seed).reset_index(drop=True)

    id_to_idx = {video_id: i for i, video_id in enumerate(videos["video_id"].tolist())}
    known_positive = set()
    for row in gt.itertuples(index=False):
        ia = id_to_idx.get(row.video_id_a)
        ib = id_to_idx.get(row.video_id_b)
        if ia is not None and ib is not None:
            known_positive.add((ia, ib))
            known_positive.add((ib, ia))

    query_rows = []
    for pair_id, row in enumerate(gt.itertuples(index=False)):
        for direction, qid, tid, qcov, tcov in [
            ("a_to_b", row.video_id_a, row.video_id_b, getattr(row, "coverage_a", np.nan), getattr(row, "coverage_b", np.nan)),
            ("b_to_a", row.video_id_b, row.video_id_a, getattr(row, "coverage_b", np.nan), getattr(row, "coverage_a", np.nan)),
        ]:
            query_rows.append(
                {
                    "pair_id": pair_id,
                    "class": row[0],
                    "query_id": qid,
                    "target_id": tid,
                    "direction": direction,
                    "coverage_query": qcov,
                    "coverage_target": tcov,
                    "coverage_min": np.nanmin([qcov, tcov]) if not (pd.isna(qcov) and pd.isna(tcov)) else np.nan,
                    "aligned_frames": getattr(row, "aligned_frames", np.nan),
                }
            )

    rows = []
    for qi, item in enumerate(query_rows, 1):
        qidx = id_to_idx.get(item["query_id"])
        tidx = id_to_idx.get(item["target_id"])
        row = dict(item)
        if qidx is None or tidx is None:
            row.update({"target_rank": np.nan, "target_score": np.nan, "top1_id": "", "top1_score": np.nan})
            rows.append(row)
            continue

        q_b = xb[qidx]
        q_a = xa[qidx]
        real = np.einsum("nfd,fd->nf", xb, q_b, optimize=True) + np.einsum("nfd,fd->nf", xa, q_a, optimize=True)
        imag = np.einsum("nfd,fd->nf", xa, q_b, optimize=True) - np.einsum("nfd,fd->nf", xb, q_a, optimize=True)
        scores = (real @ cos_table + imag @ sin_table).max(axis=1)
        scores[qidx] = -np.inf
        target_score = float(scores[tidx])
        target_rank = int(1 + np.count_nonzero(scores > target_score))
        top1_idx = int(np.argmax(scores))

        row.update(
            {
                "target_rank": target_rank,
                "target_score": target_score,
                "top1_id": videos.iloc[top1_idx]["video_id"],
                "top1_score": float(scores[top1_idx]),
                "top1_is_known_positive": int((qidx, top1_idx) in known_positive),
            }
        )
        for k in ks:
            row[f"recall@{k}"] = int(target_rank <= k)
        rows.append(row)
        if qi % 100 == 0 or qi == len(query_rows):
            print(f"exact ranked {qi}/{len(query_rows)}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / "exact_rank.csv", index=False)
    summary = summarize(result, ks)
    summary["top1_known_positive_rate"] = float(result["top1_is_known_positive"].mean())
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    by_class = aggregate_table(result, "class", ks).sort_values("queries", ascending=False)
    by_direction = aggregate_table(result, "direction", ks)
    bins = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.40, 1.0]
    result["coverage_bin"] = pd.cut(result["coverage_min"], bins=bins, include_lowest=True)
    by_coverage = aggregate_table(result, "coverage_bin", ks)
    by_class.to_csv(args.out_dir / "by_class.csv", index=False)
    by_direction.to_csv(args.out_dir / "by_direction.csv", index=False)
    by_coverage.to_csv(args.out_dir / "by_coverage.csv", index=False)
    write_markdown(args.out_dir, args, summary, by_class, by_direction, by_coverage)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
