import os
import re
import cv2
import numpy as np
from skimage.filters.rank import entropy as sk_entropy
from skimage.morphology import disk
from sklearn.preprocessing import normalize


# -------------------------
# Utils: parse frame index
# -------------------------
_FRAME_RE = re.compile(r"frame_(\d+)\.jpg$")

def parse_frame_index(fname: str):
    m = _FRAME_RE.search(fname)
    return int(m.group(1)) if m else None


# -------------------------
# Spatial entropy (Laplacian/Sobel)
# -------------------------
def spatial_complexity_map(gray: np.ndarray, mode: str = "laplacian"):
    mode = mode.lower()
    if mode == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return np.abs(lap)
    elif mode == "sobel":
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sx * sx + sy * sy)
    else:
        raise ValueError(f"Unknown spatial mode: {mode}")


def spatial_entropy(frame_bgr: np.ndarray, radius: int = 6, mode: str = "laplacian"):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    comp = spatial_complexity_map(gray, mode=mode)
    comp = cv2.normalize(comp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ent_map = sk_entropy(comp, disk(radius))
    return float(ent_map.mean())


# -------------------------
# Redundancy (HSV 3D hist cosine sim)
# -------------------------
def hsv_hist_3d(frame_bgr: np.ndarray, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256]).flatten().astype(np.float32)
    hist += 1e-6
    hist = normalize(hist.reshape(1, -1), norm="l2").reshape(-1)
    return hist

def cosine_sim(a: np.ndarray, b: np.ndarray):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# -------------------------
# Temporal coverage (normalized entropy over bins)
# -------------------------
def temporal_coverage(frame_positions_norm: np.ndarray, num_bins: int = 8):
    """
    frame_positions_norm: values in [0,1], one per selected frame
    """
    if len(frame_positions_norm) == 0:
        return 0.0
    num_bins = int(max(2, num_bins))
    bins = np.clip((frame_positions_norm * num_bins).astype(int), 0, num_bins - 1)
    counts = np.bincount(bins, minlength=num_bins).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    # entropy with numerical stability
    p = np.clip(p, 1e-12, 1.0)
    H = -np.sum(p * np.log(p))
    return float(H / np.log(num_bins))


# -------------------------
# Main STEC
# -------------------------
def stec_score_from_frames(
    frames_bgr: list,
    frame_indices: list,
    total_frames: int | None,
    spatial_mode: str = "laplacian",
    entropy_radius: int = 6,
    time_bins: int = 8,
    hist_bins=(8, 8, 8),
):
    """
    frames_bgr: list of sampled frames in chronological order
    frame_indices: original indices if known; else monotonic order indices
    total_frames: total frames in video if known; else None
    """
    K = len(frames_bgr)
    if K == 0:
        return {"S": 0.0, "T": 0.0, "R": 0.0, "STEC": 0.0}

    # 1) Spatial information
    ents = [spatial_entropy(f, radius=entropy_radius, mode=spatial_mode) for f in frames_bgr]
    S = float(np.mean(ents))

    # ---- Temporal coverage with span penalty ----
    if total_frames is None or total_frames <= 1:
        pos = np.linspace(0.0, 1.0, K, dtype=np.float32)
        span = 1.0
    else:
        idx = np.array(frame_indices, dtype=np.float32)
        if (idx.max() <= 2.0 * max(K - 1, 1)) and (total_frames >= 10 * K):
            # rank-based fallback
            pos = np.linspace(0.0, 1.0, K, dtype=np.float32)
            span = float(pos.max() - pos.min()) 
        else:
            pos = np.clip((idx - 1.0) / float(total_frames - 1), 0.0, 1.0)
            span = float((idx.max() - idx.min()) / max(total_frames - 1, 1))

    T = temporal_coverage(pos, num_bins=time_bins) * span



    # 3) Redundancy penalty via adjacent similarity (chronological)
    hists = [hsv_hist_3d(f, bins=hist_bins) for f in frames_bgr]
    if K == 1:
        mean_sim = 0.0
    else:
        sims = [cosine_sim(hists[i], hists[i + 1]) for i in range(K - 1)]
        mean_sim = float(np.mean(sims))
    R = float(np.clip(1.0 - mean_sim, 0.0, 1.0))

    STEC = float(S * T * R)
    return {"S": S, "T": T, "R": R, "STEC": STEC}


def load_sampled_frames_from_folder(folder: str):
    """
    Loads jpgs sorted by filename. Extracts frame index from name when available.
    Works for:
      - STACFP: frame_{ORIG_IDX:06d}.jpg  :contentReference[oaicite:2]{index=2}
      - Katna (renamed): frame_000000.jpg,... but these are NOT original indices :contentReference[oaicite:3]{index=3}
    """
    files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
    frames, idxs = [], []
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        if img is None:
            continue
        frames.append(img)
        idx = parse_frame_index(f)
        idxs.append(idx if idx is not None else len(idxs))
    return frames, idxs
