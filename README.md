# STEC: Spatio-Temporal Entropy Coverage Metric

STEC is a **task-agnostic metric** for evaluating sparse video frame sampling.
It explicitly captures the trade-off between:
- **S (Spatial information)**: how informative each sampled frame is
- **T (Temporal coverage)**: how well sampled frames cover the full video timeline
- **R (Non-redundancy)**: how non-redundant the sampled frames are

This repository contains:
- A reference implementation of **STEC** (`metrics/`)
- Several frame samplers (`samplers/`) including **Uniform, Random, Katna, STACFP**
- Batch scripts for dataset-level evaluation (`scripts/`)
- A **single-video demo** you can run in minutes (`demo_stec.py`)

---

## Quick start (reviewer-friendly)

### 1) Install

```bash
pip install numpy opencv-python matplotlib scikit-image scikit-learn
```

Sampler-specific dependency:
- **Katna** sampler requires Katna:

```bash
pip install Katna
```

### 2) Run a single-video demo

From repo root:

```bash
python demo_stec.py --video <PATH_TO_VIDEO> --sampler stacfp --out results/demo --save_silhouette
```

Other samplers:

```bash
python demo_stec.py --video <PATH_TO_VIDEO> --sampler uniform -k 8 --out results/demo
python demo_stec.py --video <PATH_TO_VIDEO> --sampler random  -k 8 --seed 123 --out results/demo
python demo_stec.py --video <PATH_TO_VIDEO> --sampler katna   -k 8 --out results/demo
```

The demo writes (under `results/demo/<sampler>/`):
- `stec_output.json`
- `selected_indices.json`
- `timeline.png`
- `frames_grid.png`
- `frames/` (saved sampled frames)

---

## Important note: true frame indices + meta.json

STEC's **T component** depends on *where* frames are located in the original video.
To make evaluation reproducible and correct, samplers write metadata:

- `samplers/FrameSampler_Random.py` writes `frames/meta.json` containing `selected_frame_indices_saved` (true indices)
- `samplers/FrameSampler_Kanta.py` writes `frames/meta.json`
  - if Katna succeeds: `index_mode = "rank"` (Katna API does not expose original indices)
  - if Katna falls back to OpenCV uniform sampling: `index_mode = "approx"` with best-effort true indices
- `samplers/FrameSampler_STACFP.py` writes `frames/stpac_meta.json` and also encodes true indices in filenames

The demo (`demo_stec.py`) prefers reading these meta files before falling back to filename-based inference.

---

## Repository structure

```
STEC/
├── demo_stec.py            # single-video demo (sampler -> STEC)
├── metrics/
│   └── stec_metric.py      # STEC implementation
├── samplers/
│   ├── FrameSampler_Uniform.py
│   ├── FrameSampler_Random.py
│   ├── FrameSampler_Kanta.py
│   └── FrameSampler_STACFP.py
├── scripts/
│   ├── extract_frames_all.py
│   ├── compute_stec_all.py
│   └── summary_from_json.py
└── results/
```

---

## Batch evaluation (dataset-level)

### 1) Extract frames for a list of videos

In `scripts/extract_frames_all.py`, call:

```python
extract_all(video_list, video_dir, out_root, num_frames=16)
```

This will create:

```
<out_root>/
  uniform/<video_id>/frame_*.jpg
  random/<video_id>/frame_*.jpg (+ meta.json)
  katna/<video_id>/frame_*.jpg (+ meta.json)
  stacfp/<video_id>/frame_*.jpg (+ stpac_meta.json)
```

### 2) Compute STEC for all videos

`scripts/compute_stec_all.py` loads frames and computes STEC for each sampler, writing a JSON dictionary keyed by video id.

---

## Recommended citation

If you use this code, please cite the STEC paper.
```
@article{Lin2026STEC,
  title={STEC: A Reference-Free Spatio-Temporal Entropy Coverage Metric for Evaluating Sampled Video Frames},
  author={Lin, Shih-Yao},
  journal={arXiv preprint arXiv:2601.13974},
  year={2026}
}
```
---

## License
This project is released under the Apache License 2.0.
See the `LICENSE` file for full license text.
