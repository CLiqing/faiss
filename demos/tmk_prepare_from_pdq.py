#!/usr/bin/env python3
"""Prepare flat TMK signatures from PDQ hex frame artifacts.

Input layout matches sweep_signed_bit_w.py:

  artifact_dir/features_fps8/videos.parquet
  artifact_dir/features_fps8/frames.parquet

Output:

  tmk_flat.f32       raw float32 matrix in row-major order
  tmk_flat.npy       NumPy copy of the same matrix
  tmk_videos.csv     video order matching signature rows
  tmk_meta.json      periods/layout/config metadata
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_periods(text: str, include_dc: bool) -> list[float]:
    periods = [float(x.strip()) for x in text.split(",") if x.strip()]
    if include_dc:
        return [0.0] + periods
    return periods


def period_weights(periods: list[float], scheme: str) -> np.ndarray:
    periods_np = np.asarray(periods, dtype=np.float32)
    weights = np.ones_like(periods_np, dtype=np.float32)
    non_dc = periods_np > 0
    if not np.any(non_dc) or scheme == "uniform":
        return weights
    p = periods_np[non_dc]
    center = float(np.exp(np.mean(np.log(p))))
    if scheme.startswith("short"):
        alpha = float(scheme.replace("short", "")) / 100.0
        weights[non_dc] = (center / p) ** alpha
    elif scheme.startswith("long"):
        alpha = float(scheme.replace("long", "")) / 100.0
        weights[non_dc] = (p / center) ** alpha
    elif scheme.startswith("mid"):
        target = float(scheme.replace("mid", ""))
        sigma = 1.15
        weights[non_dc] = np.exp(-0.5 * (np.log(p / target) / sigma) ** 2)
    else:
        raise ValueError(f"unknown weight scheme: {scheme}")
    rms = float(np.sqrt(np.mean(weights[non_dc] * weights[non_dc])))
    if rms > 0:
        weights[non_dc] /= rms
    return weights


def hexes_to_signed_float(hexes: pd.Series) -> np.ndarray:
    blob = bytes.fromhex("".join(hexes.astype(str).tolist()))
    raw = np.frombuffer(blob, dtype=np.uint8).reshape(len(hexes), 32)
    return np.unpackbits(raw, axis=1).astype(np.float32) * 2.0 - 1.0


def compute_signatures(
    artifact_dir: Path,
    periods: list[float],
    weights: np.ndarray,
    normalize: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    features = artifact_dir / "features_fps8"
    videos = pd.read_parquet(features / "videos.parquet").sort_values("video_id").reset_index(drop=True)
    frames = pd.read_parquet(features / "frames.parquet", columns=["video_id", "t", "pdq_hex"])
    frames = frames.sort_values(["video_id", "t"], kind="mergesort")

    embedding_dim = 256
    w = np.asarray([0.0 if p == 0.0 else 2.0 * math.pi / p for p in periods], dtype=np.float32)
    video_to_row = {video_id: i for i, video_id in enumerate(videos["video_id"].tolist())}
    a = np.zeros((len(videos), len(periods), embedding_dim), dtype=np.float32)
    b = np.zeros_like(a)

    for count, (video_id, group) in enumerate(frames.groupby("video_id", sort=False), 1):
        row = video_to_row.get(video_id)
        if row is None:
            continue
        x = hexes_to_signed_float(group["pdq_hex"])
        t = group["t"].to_numpy(dtype=np.float32)
        wt = t[:, None] * w[None, :]
        sin = np.sin(wt).astype(np.float32)
        cos = np.cos(wt).astype(np.float32)
        dc = w == 0.0
        if np.any(dc):
            sin[:, dc] = 0.0
            cos[:, dc] = 1.0
        a[row] = sin.T @ x
        b[row] = cos.T @ x
        if count % 500 == 0 or count == len(videos):
            print(f"computed signatures {count}/{len(videos)}", flush=True)

    a *= weights[None, :, None]
    b *= weights[None, :, None]

    if normalize:
        norms = np.sqrt(np.sum(a * a, axis=(1, 2)) + np.sum(b * b, axis=(1, 2)))
        valid = norms > 0
        a[valid] /= norms[valid, None, None]
        b[valid] /= norms[valid, None, None]

    flat = np.concatenate([b, a], axis=1).reshape(len(videos), -1).astype(np.float32)
    return videos, flat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--periods", default="64,128,256,512,1024,2048,4096")
    parser.add_argument("--no-dc", action="store_true")
    parser.add_argument("--weight-scheme", default="long025")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    periods = parse_periods(args.periods, include_dc=not args.no_dc)
    weights = period_weights(periods, args.weight_scheme)
    videos, flat = compute_signatures(
        args.artifact_dir,
        periods,
        weights=weights,
        normalize=not args.no_normalize,
    )

    flat.tofile(args.out_dir / "tmk_flat.f32")
    np.save(args.out_dir / "tmk_flat.npy", flat)
    videos.to_csv(args.out_dir / "tmk_videos.csv", index=False)
    meta = {
        "layout": "[b(period0),...,b(periodF-1),a(period0),...,a(periodF-1)]",
        "dtype": "float32",
        "shape": list(flat.shape),
        "embedding_dim": 256,
        "periods": periods,
        "weight_scheme": args.weight_scheme,
        "weights": [float(x) for x in weights.tolist()],
        "normalize": not args.no_normalize,
        "includes_dc": not args.no_dc,
    }
    (args.out_dir / "tmk_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
