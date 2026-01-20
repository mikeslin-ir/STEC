#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""General demo: choose a sampler, sample frames, then compute STEC.

Run:
  python demo_stec.py --video <path> --sampler {stacfp|uniform|random|katna} -k 8 --out results/demo
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Make imports robust: add repo root to sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from samplers.FrameSampler_Uniform import FrameSampler_Uniform
from samplers.FrameSampler_STACFP import FrameSampler_STACFP
from samplers.FrameSampler_Random import FrameSampler_Random
from samplers.FrameSampler_Kanta import FrameSampler_Kanta

try:
    from metrics.stec_metric import STECMetric, STECConfig
except ModuleNotFoundError:
    from stec_metric import STECMetric, STECConfig


FRAME_RE = re.compile(r"frame_(\d+)\.jpg$")


def parse_frame_index_from_filename(path: str) -> Optional[int]:
    m = FRAME_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def get_video_stats(video_path: str) -> Tuple[int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 0:
        fps = 30.0
    return total_frames, fps


def load_frames_from_folder(folder: str) -> Tuple[List[np.ndarray], List[str]]:
    paths = sorted(glob.glob(os.path.join(folder, "frame_*.jpg")))
    if not paths:
        raise RuntimeError(f"No frames found in: {folder}")

    frames_bgr: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        frames_bgr.append(img)
    return frames_bgr, paths


def read_meta_indices(frames_dir: str) -> Optional[Tuple[List[int], str, Dict[str, Any]]]:
    """Prefer sampler-written meta.json; fall back to STACFP's stpac_meta.json."""
    meta_path = os.path.join(frames_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        idxs = meta.get("selected_frame_indices_saved") or meta.get("selected_frame_indices")
        if isinstance(idxs, list) and idxs:
            mode = meta.get("index_mode", "unknown")
            return [int(x) for x in idxs], str(mode), meta

    stpac_path = os.path.join(frames_dir, "stpac_meta.json")
    if os.path.isfile(stpac_path):
        with open(stpac_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        idxs = meta.get("selected_frame_indices")
        if isinstance(idxs, list) and idxs:
            return [int(x) for x in idxs], "true", meta

    return None


def infer_indices_from_filenames(paths: List[str]) -> List[int]:
    idxs: List[int] = []
    for p in paths:
        idx = parse_frame_index_from_filename(p)
        if idx is None:
            idxs.append(len(idxs))
        else:
            idxs.append(idx)
    return idxs


def save_timeline_plot(out_path: str, total_frames: int, indices: List[int], fps: float, title: str) -> None:
    xs = np.array(indices, dtype=np.float32)
    plt.figure(figsize=(10, 2.2))
    plt.hlines(0, 0, max(total_frames - 1, 1), linewidth=3)
    plt.scatter(xs, np.zeros_like(xs), s=80)
    plt.yticks([])
    plt.xlim(0, max(total_frames - 1, 1))
    plt.xlabel(f"frame index  (sec ≈ idx/{fps:.2f})")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_frame_grid(out_path: str, frames_bgr: List[np.ndarray], indices: List[int], fps: float, max_cols: int = 4) -> None:
    k = len(frames_bgr)
    cols = min(max_cols, max(1, k))
    rows = int(np.ceil(k / cols))
    plt.figure(figsize=(3.2 * cols, 3.0 * rows))

    for i in range(k):
        rgb = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB)
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(rgb)
        if i < len(indices):
            idx = indices[i]
            ax.set_title(f"idx={idx}  t={idx / fps:.2f}s", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_sampler(args: argparse.Namespace):
    name = args.sampler.lower()
    if name == "uniform":
        return FrameSampler_Uniform(no_of_frames=args.k)
    if name == "random":
        return FrameSampler_Random(num_frames=args.k, seed=args.seed)
    if name == "katna":
        return FrameSampler_Kanta(no_of_frames=args.k, use_uniform_on_fail=True)
    if name == "stacfp":
        return FrameSampler_STACFP(
            frame_step=args.frame_step,
            color_space=args.color_space,
            hist_bins=tuple(args.stacfp_hist_bins),
            gamma_time=args.gamma_time,
            k_min=args.k_min,
            k_max=args.k_max,
            resize_shorter_to=args.resize_shorter_to,
            random_state=args.seed,
        )
    raise ValueError(f"Unknown sampler: {args.sampler}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--sampler", default="stacfp", choices=["stacfp", "uniform", "random", "katna"])
    ap.add_argument("-k", "--k", type=int, default=8, help="k for uniform/random/katna")
    ap.add_argument("--out", default="results/demo", help="Output root dir")
    ap.add_argument("--seed", type=int, default=42, help="Seed for random sampler")

    # STACFP knobs
    ap.add_argument("--frame_step", type=int, default=10)
    ap.add_argument("--color_space", type=str, default="HSV", choices=["HSV", "Lab", "LAB"])
    ap.add_argument("--stacfp_hist_bins", type=int, nargs=3, default=[8, 8, 8])
    ap.add_argument("--gamma_time", type=float, default=1.2)
    ap.add_argument("--k_min", type=int, default=5)
    ap.add_argument("--k_max", type=int, default=30)
    ap.add_argument("--resize_shorter_to", type=int, default=256)
    ap.add_argument("--save_silhouette", action="store_true")

    # STEC knobs
    ap.add_argument("--time_bins", type=int, default=8)
    ap.add_argument("--entropy_radius", type=int, default=6)
    ap.add_argument("--spatial_mode", type=str, default="laplacian", choices=["laplacian", "sobel"])
    args = ap.parse_args()

    sampler_name = args.sampler.lower()
    out_dir = os.path.join(args.out, sampler_name)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    total_frames, fps = get_video_stats(args.video)

    sampler = build_sampler(args)

    # 1) Sample frames
    if sampler_name == "stacfp":
        sampler.extract_frames(args.video, frames_dir, is_saved_silhouette=args.save_silhouette)
    else:
        sampler.extract_frames(args.video, frames_dir)

    # 2) Load frames
    frames_bgr, paths = load_frames_from_folder(frames_dir)

    # 3) Indices: prefer meta.json (random/katna), or stpac_meta.json (stacfp)
    meta_read = read_meta_indices(frames_dir)
    if meta_read is not None:
        indices, index_mode, meta = meta_read
    else:
        indices = infer_indices_from_filenames(paths)
        index_mode = "filename"
        meta = {}

    # Align lengths if any decoding failed
    m = min(len(frames_bgr), len(indices))
    frames_bgr = frames_bgr[:m]
    indices = indices[:m]

    # If indices are not strictly increasing, keep current order but allow STEC heuristic for T
    use_rank_fallback = sampler_name in ["katna"] or index_mode in ["rank", "filename"]

    # 4) Compute STEC
    cfg = STECConfig(
        spatial_mode=args.spatial_mode,
        entropy_radius=args.entropy_radius,
        time_bins=args.time_bins,
        hist_bins=(8, 8, 8),
        use_span_penalty=True,
        use_rank_fallback=use_rank_fallback,
    )
    metric = STECMetric(cfg)
    stec_out = metric.compute(
        frames_bgr=frames_bgr,
        frame_indices=indices,
        total_frames=total_frames if total_frames > 0 else None,
        return_debug=True,
    )

    print("==== DEMO: sampler -> STEC ====")
    print(f"sampler: {sampler_name} (index_mode={index_mode})")
    print(f"video: {args.video}")
    print(f"total_frames: {total_frames}, fps: {fps:.3f}")
    print(f"num_selected: {len(frames_bgr)}")
    print(f"S: {stec_out['S']:.6f}")
    print(f"T: {stec_out['T']:.6f}")
    print(f"R: {stec_out['R']:.6f}")
    print(f"STEC: {stec_out['STEC']:.6f}")

    # 5) Save outputs
    with open(os.path.join(out_dir, "stec_output.json"), "w", encoding="utf-8") as f:
        json.dump(stec_out, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "selected_indices.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "sampler": sampler_name,
                "index_mode": index_mode,
                "total_frames": total_frames,
                "fps": fps,
                "k_saved": len(indices),
                "indices": indices,
                "seed": args.seed,
                "meta_hint": {k: meta.get(k) for k in ["fallback_used", "seed", "requested_k", "saved_count"] if k in meta},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_timeline_plot(
        os.path.join(out_dir, "timeline.png"),
        max(total_frames, 1),
        indices,
        fps,
        title=f"Selected frames timeline ({sampler_name}, index_mode={index_mode})",
    )
    save_frame_grid(os.path.join(out_dir, "frames_grid.png"), frames_bgr, indices, fps)

    print("\nSaved:")
    print(f"  {os.path.join(out_dir, 'stec_output.json')}")
    print(f"  {os.path.join(out_dir, 'selected_indices.json')}")
    print(f"  {os.path.join(out_dir, 'timeline.png')}")
    print(f"  {os.path.join(out_dir, 'frames_grid.png')}")
    print(f"  frames folder: {frames_dir}")


if __name__ == "__main__":
    main()
