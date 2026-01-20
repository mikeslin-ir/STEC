import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.filters.rank import entropy as sk_entropy
from skimage.morphology import disk
from sklearn.preprocessing import normalize


# -------------------------
# Utils: parse frame index
# -------------------------
_FRAME_RE = re.compile(r"frame_(\d+)\.jpg$")  # same behavior as your current code :contentReference[oaicite:5]{index=5}


def parse_frame_index(fname: str) -> Optional[int]:
    """Parse `frame_XXXXXX.jpg` -> int(XXXXXX). Return None if not matched."""
    m = _FRAME_RE.search(fname)
    return int(m.group(1)) if m else None


# -------------------------
# STEC Metric (Class API)
# -------------------------
@dataclass(frozen=True)
class STECConfig:
    spatial_mode: str = "laplacian"   # {"laplacian","sobel"} :contentReference[oaicite:6]{index=6}
    entropy_radius: int = 6          # :contentReference[oaicite:7]{index=7}
    time_bins: int = 8               # :contentReference[oaicite:8]{index=8}
    hist_bins: Tuple[int, int, int] = (8, 8, 8)  # :contentReference[oaicite:9]{index=9}
    use_span_penalty: bool = True
    # If True, we keep your rank-based fallback heuristic when indices look like 0..K-1 but total_frames is huge :contentReference[oaicite:10]{index=10}
    use_rank_fallback: bool = True


class STECMetric:
    """
    STEC = S * T * R

    - S: spatial information (mean local entropy over spatial complexity map)
    - T: temporal coverage (entropy over temporal bins) optionally multiplied by span penalty
    - R: non-redundancy = 1 - mean adjacent cosine similarity of HSV 3D histograms

    This class preserves the behavior of your current functional implementation:
    stec_score_from_frames(...) :contentReference[oaicite:11]{index=11}
    """

    def __init__(self, config: Optional[STECConfig] = None):
        self.config = config or STECConfig()

    # -------------------------
    # Spatial entropy
    # -------------------------
    @staticmethod
    def spatial_complexity_map(gray: np.ndarray, mode: str = "laplacian") -> np.ndarray:
        mode = mode.lower()
        if mode == "laplacian":
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            return np.abs(lap)
        if mode == "sobel":
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(sx * sx + sy * sy)
        raise ValueError(f"Unknown spatial mode: {mode}")

    def spatial_entropy(self, frame_bgr: np.ndarray) -> float:
        """Mean local entropy over a normalized complexity map."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        comp = self.spatial_complexity_map(gray, mode=self.config.spatial_mode)
        comp_u8 = cv2.normalize(comp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ent_map = sk_entropy(comp_u8, disk(int(self.config.entropy_radius)))
        return float(ent_map.mean())

    # -------------------------
    # Redundancy (HSV 3D hist)
    # -------------------------
    @staticmethod
    def hsv_hist_3d(frame_bgr: np.ndarray, bins: Tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            list(bins),
            [0, 180, 0, 256, 0, 256],
        ).flatten().astype(np.float32)
        hist += 1e-6
        hist = normalize(hist.reshape(1, -1), norm="l2").reshape(-1)
        return hist

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # -------------------------
    # Temporal coverage
    # -------------------------
    @staticmethod
    def temporal_coverage(frame_positions_norm: np.ndarray, num_bins: int = 8) -> float:
        """
        Normalized entropy over temporal bins.
        frame_positions_norm: values in [0,1], one per selected frame :contentReference[oaicite:12]{index=12}
        """
        if len(frame_positions_norm) == 0:
            return 0.0
        num_bins = int(max(2, num_bins))
        bins = np.clip((frame_positions_norm * num_bins).astype(int), 0, num_bins - 1)
        counts = np.bincount(bins, minlength=num_bins).astype(np.float64)
        p = counts / max(counts.sum(), 1.0)
        p = np.clip(p, 1e-12, 1.0)
        H = -np.sum(p * np.log(p))
        return float(H / np.log(num_bins))

    # -------------------------
    # Main compute API
    # -------------------------
    def compute(
        self,
        frames_bgr: Sequence[np.ndarray],
        frame_indices: Sequence[int],
        total_frames: Optional[int],
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Args:
            frames_bgr: sampled frames in chronological order (BGR)
            frame_indices: original indices if known; else monotonic order indices
            total_frames: total frames in the full video (optional)
            return_debug: if True, return extra intermediate values

        Returns:
            dict with keys: S, T, R, STEC (+ optional debug)
        """
        K = len(frames_bgr)
        if K == 0:
            out = {"S": 0.0, "T": 0.0, "R": 0.0, "STEC": 0.0}
            if return_debug:
                out["debug"] = {"K": 0}
            return out

        # 1) Spatial information (mean entropy) :contentReference[oaicite:13]{index=13}
        ents = [self.spatial_entropy(f) for f in frames_bgr]
        S = float(np.mean(ents))

        # 2) Temporal coverage with optional span penalty :contentReference[oaicite:14]{index=14}
        pos, span = self._compute_positions_and_span(frame_indices, total_frames, K)
        T_raw = self.temporal_coverage(pos, num_bins=int(self.config.time_bins))
        T = float(T_raw * span) if self.config.use_span_penalty else float(T_raw)

        # 3) Non-redundancy via adjacent similarity (chronological) :contentReference[oaicite:15]{index=15}
        hists = [self.hsv_hist_3d(f, bins=self.config.hist_bins) for f in frames_bgr]
        if K == 1:
            mean_sim = 0.0
            sims: List[float] = []
        else:
            sims = [self.cosine_sim(hists[i], hists[i + 1]) for i in range(K - 1)]
            mean_sim = float(np.mean(sims))
        R = float(np.clip(1.0 - mean_sim, 0.0, 1.0))

        STEC = float(S * T * R)

        out: Dict[str, Any] = {"S": S, "T": T, "R": R, "STEC": STEC}
        if return_debug:
            out["debug"] = {
                "K": K,
                "spatial_entropies": ents,
                "frame_positions_norm": pos.tolist(),
                "span": float(span),
                "temporal_coverage_raw": float(T_raw),
                "adjacent_sims": sims,
                "mean_adjacent_sim": float(mean_sim),
            }
        return out

    def compute_from_folder(
        self,
        folder: str,
        total_frames: Optional[int] = None,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """Load jpg frames from folder then compute STEC."""
        frames, idxs = load_sampled_frames_from_folder(folder)
        return self.compute(frames_bgr=frames, frame_indices=idxs, total_frames=total_frames, return_debug=return_debug)

    # -------------------------
    # Internal helpers
    # -------------------------
    def _compute_positions_and_span(
        self,
        frame_indices: Sequence[int],
        total_frames: Optional[int],
        K: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Mirrors your existing logic:
        - If total_frames missing -> linspace
        - Else compute normalized positions from indices
        - Includes the rank-based fallback heuristic when indices look "too small" relative to total_frames :contentReference[oaicite:16]{index=16}
        """
        if total_frames is None or total_frames <= 1:
            pos = np.linspace(0.0, 1.0, K, dtype=np.float32)
            return pos, 1.0

        idx = np.array(list(frame_indices), dtype=np.float32)
        if idx.size == 0:
            pos = np.linspace(0.0, 1.0, K, dtype=np.float32)
            return pos, 1.0

        if self.config.use_rank_fallback:
            if (idx.max() <= 2.0 * max(K - 1, 1)) and (total_frames >= 10 * K):
                pos = np.linspace(0.0, 1.0, K, dtype=np.float32)
                span = float(pos.max() - pos.min())
                return pos, span

        pos = np.clip((idx - 1.0) / float(total_frames - 1), 0.0, 1.0)
        span = float((idx.max() - idx.min()) / max(total_frames - 1, 1))
        return pos.astype(np.float32), span


# -------------------------
# Backward-compatible functional API
# -------------------------
def spatial_complexity_map(gray: np.ndarray, mode: str = "laplacian") -> np.ndarray:
    return STECMetric.spatial_complexity_map(gray, mode=mode)


def spatial_entropy(frame_bgr: np.ndarray, radius: int = 6, mode: str = "laplacian") -> float:
    # Keep original signature; delegate to class instance.
    metric = STECMetric(STECConfig(spatial_mode=mode, entropy_radius=radius))
    return metric.spatial_entropy(frame_bgr)


def hsv_hist_3d(frame_bgr: np.ndarray, bins=(8, 8, 8)) -> np.ndarray:
    return STECMetric.hsv_hist_3d(frame_bgr, bins=bins)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return STECMetric.cosine_sim(a, b)


def temporal_coverage(frame_positions_norm: np.ndarray, num_bins: int = 8) -> float:
    return STECMetric.temporal_coverage(frame_positions_norm, num_bins=num_bins)


def stec_score_from_frames(
    frames_bgr: list,
    frame_indices: list,
    total_frames: int | None,
    spatial_mode: str = "laplacian",
    entropy_radius: int = 6,
    time_bins: int = 8,
    hist_bins=(8, 8, 8),
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper around STECMetric.compute().
    Preserves behavior of your current implementation :contentReference[oaicite:17]{index=17}
    """
    cfg = STECConfig(
        spatial_mode=spatial_mode,
        entropy_radius=entropy_radius,
        time_bins=time_bins,
        hist_bins=tuple(hist_bins),
        use_span_penalty=True,
        use_rank_fallback=True,
    )
    metric = STECMetric(cfg)
    return metric.compute(frames_bgr=frames_bgr, frame_indices=frame_indices, total_frames=total_frames, return_debug=False)


# -------------------------
# I/O: load sampled frames from folder
# -------------------------
def load_sampled_frames_from_folder(folder: str) -> Tuple[List[np.ndarray], List[int]]:
    """
    Loads jpgs sorted by filename. Extracts frame index from name when available.
    Works for:
      - STACFP: frame_{ORIG_IDX:06d}.jpg
      - Katna (renamed): frame_000000.jpg,... but these are NOT original indices

    Same behavior as your current function :contentReference[oaicite:18]{index=18}
    """
    files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
    frames: List[np.ndarray] = []
    idxs: List[int] = []
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        if img is None:
            continue
        frames.append(img)
        idx = parse_frame_index(f)
        idxs.append(idx if idx is not None else len(idxs))
    return frames, idxs
