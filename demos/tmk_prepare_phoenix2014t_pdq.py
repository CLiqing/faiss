#!/usr/bin/env python3
"""Prepare PDQ frame artifacts from RWTH-PHOENIX-Weather 2014T.

The output layout matches tmk_prepare_from_pdq.py:

  artifact_dir/features_fps8/videos.parquet
  artifact_dir/features_fps8/frames.parquet
  artifact_dir/gt/phoenix_synthetic_gt_pairs.csv

The script computes one 256-bit PDQ hash per sampled frame and optionally creates
synthetic contain/shifted positives at the PDQ sequence level. This gives us a
dedup-style benchmark from a temporally structured sign-language corpus.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pdqhash
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}


@dataclass
class SourceVideo:
    video_id: str
    paths: list[Path]


def bits_to_hex(bits: np.ndarray) -> str:
    bits = np.asarray(bits, dtype=np.uint8).reshape(256)
    return np.packbits(bits).tobytes().hex()


def image_to_pdq_hex(path: Path) -> str:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bits, _quality = pdqhash.compute(arr)
    return bits_to_hex(bits)


def natural_key(path: Path) -> tuple:
    parts: list[int | str] = []
    stem = path.stem
    cur = ""
    is_digit = False
    for ch in stem:
        if ch.isdigit() != is_digit and cur:
            parts.append(int(cur) if is_digit else cur)
            cur = ch
            is_digit = ch.isdigit()
        else:
            cur += ch
            is_digit = ch.isdigit()
    if cur:
        parts.append(int(cur) if is_digit else cur)
    return tuple(parts) + (path.name,)


def find_feature_root(root: Path) -> Path:
    candidates = [
        root / "PHOENIX-2014-T" / "features" / "fullFrame-210x260px",
        root / "features" / "fullFrame-210x260px",
        root / "fullFrame-210x260px",
        root,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find PHOENIX feature root under {root}")


def discover_videos(root: Path, max_sources: int) -> list[SourceVideo]:
    feature_root = find_feature_root(root)
    videos: list[SourceVideo] = []
    for dirpath, _dirnames, filenames in os.walk(feature_root):
        image_names = [
            name for name in filenames if Path(name).suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_names:
            continue
        frame_dir = Path(dirpath)
        rel = frame_dir.relative_to(feature_root)
        video_id = str(rel).replace(os.sep, "/")
        paths = [frame_dir / name for name in image_names]
        paths.sort(key=natural_key)
        videos.append(SourceVideo(video_id=video_id, paths=paths))
    videos.sort(key=lambda item: item.video_id)
    if max_sources > 0:
        videos = videos[:max_sources]
    return videos


def sampled_pdq_sequence(video: SourceVideo, stride: int, max_frames: int) -> list[str]:
    selected = video.paths[::stride]
    if max_frames > 0:
        selected = selected[:max_frames]
    return [image_to_pdq_hex(path) for path in selected]


def add_video(
    video_rows: list[dict],
    frame_rows: list[dict],
    video_id: str,
    pdq_hexes: list[str],
    source_video_id: str,
    variant: str,
) -> None:
    if not pdq_hexes:
        return
    video_rows.append(
        {
            "video_id": video_id,
            "source_video_id": source_video_id,
            "variant": variant,
            "n_frames": len(pdq_hexes),
        }
    )
    for t, pdq_hex in enumerate(pdq_hexes):
        frame_rows.append({"video_id": video_id, "t": t, "pdq_hex": pdq_hex})


def middle_slice(seq: list[str], target_len: int) -> list[str]:
    if len(seq) <= target_len:
        return list(seq)
    start = max(0, (len(seq) - target_len) // 2)
    return list(seq[start : start + target_len])


def head_slice(seq: list[str], target_len: int) -> list[str]:
    return list(seq[: min(len(seq), target_len)])


def tail_slice(seq: list[str], target_len: int) -> list[str]:
    if target_len <= 0:
        return []
    return list(seq[max(0, len(seq) - target_len) :])


def build_artifact(
    source_videos: list[SourceVideo],
    out_dir: Path,
    fps: float,
    source_fps: float,
    max_frames_per_source: int,
    min_frames: int,
    synthetic_count: int,
    synthetic_segment_frames: int,
    seed: int,
) -> None:
    stride = max(1, round(source_fps / fps))
    rng = random.Random(seed)

    hashed: list[tuple[str, list[str]]] = []
    for idx, source in enumerate(source_videos, 1):
        seq = sampled_pdq_sequence(source, stride=stride, max_frames=max_frames_per_source)
        if len(seq) < min_frames:
            continue
        hashed.append((source.video_id, seq))
        if idx % 100 == 0 or idx == len(source_videos):
            print(
                f"hashed source videos {idx}/{len(source_videos)} "
                f"kept={len(hashed)}",
                flush=True,
            )

    if len(hashed) < 3:
        raise RuntimeError("need at least 3 source videos after filtering")

    video_rows: list[dict] = []
    frame_rows: list[dict] = []
    gt_rows: list[dict] = []

    for source_id, seq in hashed:
        add_video(
            video_rows,
            frame_rows,
            video_id=f"{source_id}::full",
            pdq_hexes=seq,
            source_video_id=source_id,
            variant="full",
        )

    eligible = [(source_id, seq) for source_id, seq in hashed if len(seq) >= min_frames]
    if synthetic_count > 0:
        eligible = eligible[: min(synthetic_count, len(eligible))]

    all_indices = list(range(len(hashed)))
    for idx, (source_id, seq) in enumerate(eligible):
        segment_len = min(synthetic_segment_frames, max(min_frames, len(seq) // 2))
        needle = middle_slice(seq, segment_len)
        if len(needle) < min_frames:
            continue

        other_indices = [i for i in all_indices if hashed[i][0] != source_id]
        prefix_id = rng.choice(other_indices)
        suffix_id = rng.choice([i for i in other_indices if i != prefix_id])
        prefix = tail_slice(hashed[prefix_id][1], max(min_frames // 2, segment_len // 3))
        suffix = head_slice(hashed[suffix_id][1], max(min_frames // 2, segment_len // 3))
        haystack = prefix + needle + suffix

        needle_id = f"{source_id}::needle"
        haystack_id = f"{source_id}::haystack"
        shifted_id = f"{source_id}::shifted_full"
        add_video(video_rows, frame_rows, needle_id, needle, source_id, "needle")
        add_video(video_rows, frame_rows, haystack_id, haystack, source_id, "haystack")
        add_video(video_rows, frame_rows, shifted_id, prefix + seq, source_id, "shifted_full")
        gt_rows.append(
            {
                "video_id_a": needle_id,
                "video_id_b": haystack_id,
                "class": "synthetic_contain",
            }
        )
        gt_rows.append(
            {
                "video_id_a": f"{source_id}::full",
                "video_id_b": shifted_id,
                "class": "synthetic_shifted",
            }
        )
        if (idx + 1) % 100 == 0 or idx + 1 == len(eligible):
            print(
                f"built synthetic positives {idx + 1}/{len(eligible)}",
                flush=True,
            )

    features_dir = out_dir / "features_fps8"
    gt_dir = out_dir / "gt"
    features_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    videos = pd.DataFrame(video_rows).sort_values("video_id").reset_index(drop=True)
    frames = pd.DataFrame(frame_rows).sort_values(["video_id", "t"]).reset_index(drop=True)
    videos.to_parquet(features_dir / "videos.parquet", index=False)
    frames.to_parquet(features_dir / "frames.parquet", index=False)

    gt_path = gt_dir / "phoenix_synthetic_gt_pairs.csv"
    with gt_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id_a", "video_id_b", "class"])
        writer.writeheader()
        writer.writerows(gt_rows)

    print(f"videos={len(videos)}")
    print(f"frames={len(frames)}")
    print(f"gt_pairs={len(gt_rows)}")
    print(f"stride={stride}")
    print(f"wrote={out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--source-fps", type=float, default=25.0)
    parser.add_argument("--max-sources", type=int, default=1000)
    parser.add_argument("--max-frames-per-source", type=int, default=512)
    parser.add_argument("--min-frames", type=int, default=32)
    parser.add_argument("--synthetic-count", type=int, default=500)
    parser.add_argument("--synthetic-segment-frames", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    source_videos = discover_videos(args.root, max_sources=args.max_sources)
    print(f"discovered_source_videos={len(source_videos)}")
    build_artifact(
        source_videos=source_videos,
        out_dir=args.out_dir,
        fps=args.fps,
        source_fps=args.source_fps,
        max_frames_per_source=args.max_frames_per_source,
        min_frames=args.min_frames,
        synthetic_count=args.synthetic_count,
        synthetic_segment_frames=args.synthetic_segment_frames,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
