# metrics/stec_metric.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np
from skimage.filters.rank import entropy as sk_entropy
from skimage.morphology import disk
from sklearn.preprocessing import normalize

# ============================================================
# Utils: parse frame index (kept for backward compatibility)
# ============================================================
_FRAME_RE = re.compile(r"frame_(\d+)\.jpg$")


def parse_frame_index(fname: str) -> Optional[int]:
    """Parse `frame_XXXXXX.jpg` -> int(XXXXXX). Return None if not matched."""
    m = _FRAME_RE.search(fname)
    return int(m.group(1)) if m else None


# ============================================================
# Backward-compatible config (matches demo import)
# ============================================================
@dataclass(frozen=True)
class STECConfig:
    spatial_mode: str = "laplacian"   # {"laplacian","sobel"}
    entropy_radius: int = 6
    time_bins: int = 8
    hist_bins: Tuple[int, int, int] = (8, 8, 8)
    use_span_penalty: bool = True
    use_rank_fallback: bool = True


# ============================================================
# Shared interfaces / Context
# ============================================================
class ScoreComponent(Protocol):
    name: str

    def compute(self, ctx: "STECContext") -> Tuple[float, Dict[str, Any]]:
        ...


@dataclass
class STECContext:
    frames_bgr: Sequence[np.ndarray]
    frame_indices: Sequence[int]
    total_frames: Optional[int]

    temporal_pos_norm: Optional[np.ndarray] = None
    span: Optional[float] = None
    hsv_hists: Optional[List[np.ndarray]] = None
    spatial_entropies: Optional[List[float]] = None


# ============================================================
# Temporal components (paper Eq.2–5)
# ============================================================
@dataclass(frozen=True)
class TemporalPositions:
    """Compute normalized temporal positions τ_j = (i_j-1)/(N-1) (Eq.2)."""
    name: str = "temporal_positions"
    use_rank_fallback: bool = True

    def compute(self, ctx: STECContext) -> Tuple[float, Dict[str, Any]]:
        K = len(ctx.frames_bgr)
        N = ctx.total_frames

        if K == 0:
            ctx.temporal_pos_norm = np.array([], dtype=np.float32)
            ctx.span = 0.0
            return 0.0, {"K": 0}

        # If N missing -> evenly spaced fallback
        if N is None or N <= 1:
            pos = np.linspace(0.0, 1.0, K, dtype=np.float32)
            ctx.temporal_pos_norm = pos
            ctx.span = float(pos.max() - pos.min()) if K > 0 else 0.0
            return 1.0, {"fallback": "no_total_frames", "pos": pos.tolist(), "span": ctx.span}

        idx = np.asarray(list(ctx.frame_indices), dtype=np.float32)
        if idx.size == 0:
            pos = np.linspace(0.0, 1.0, K, dtype=np.float32)
            ctx.temporal_pos_norm = pos
            ctx.span = float(pos.max() - pos.min()) if K > 0 else 0.0
            return 1.0, {"fallback": "empty_indices", "pos": pos.tolist(), "span": ctx.span}

        # rank-based fallback (Katna / renamed frames: 0..K-1 while total_frames huge)
        if self.use_rank_fallback and (idx.max() <= 2.0 * max(K - 1, 1)) and (N >= 10 * K):
            pos = np.linspace(0.0, 1.0, K, dtype=np.float32)
            ctx.temporal_pos_norm = pos
            ctx.span = float(pos.max() - pos.min())
            return 1.0, {"fallback": "rank_fallback", "pos": pos.tolist(), "span": ctx.span}

        # Normal case
        pos = np.clip((idx - 1.0) / float(N - 1), 0.0, 1.0).astype(np.float32)
        span = float((idx.max() - idx.min()) / max(float(N - 1), 1.0))
        ctx.temporal_pos_norm = pos
        ctx.span = span
        return 1.0, {"pos": pos.tolist(), "span": span, "N": int(N)}


@dataclass(frozen=True)
class TemporalEntropy:
    """E_t(S): normalized Shannon entropy over temporal bins (Eq.3)."""
    name: str = "temporal_entropy"
    time_bins: int = 8

    def compute(self, ctx: STECContext) -> Tuple[float, Dict[str, Any]]:
        pos = ctx.temporal_pos_norm
        if pos is None:
            raise ValueError("Temporal positions not computed. Add TemporalPositions first.")
        if len(pos) == 0:
            return 0.0, {"reason": "no_frames"}

        B = int(max(2, self.time_bins))
        bins = np.clip((pos * B).astype(int), 0, B - 1)
        counts = np.bincount(bins, minlength=B).astype(np.float64)

        p = counts / max(counts.sum(), 1.0)
        mask = p > 0
        H = -np.sum(p[mask] * np.log(p[mask]))
        Et = float(H / np.log(B))
        return Et, {"B": B, "counts": counts.tolist(), "Et": Et}


# ============================================================
# Redundancy component (paper Eq.6)
# ============================================================
@dataclass(frozen=True)
class Redundancy:
    """R(S) = 1 - mean adjacent cosine similarity of HSV hist (Eq.6)."""
    name: str = "non_redundancy"
    hist_bins: Tuple[int, int, int] = (8, 8, 8)

    @staticmethod
    def _hsv_hist_3d(frame_bgr: np.ndarray, bins: Tuple[int, int, int]) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, list(bins),
            [0, 180, 0, 256, 0, 256],
        ).flatten().astype(np.float32)
        hist += 1e-6
        hist = normalize(hist.reshape(1, -1), norm="l2").reshape(-1)
        return hist

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def compute(self, ctx: STECContext) -> Tuple[float, Dict[str, Any]]:
        K = len(ctx.frames_bgr)
        if K <= 1:
            return (1.0 if K == 1 else 0.0), {"K": K, "adjacent_sims": [], "mean_sim": 0.0, "R": (1.0 if K == 1 else 0.0)}

        if ctx.hsv_hists is None:
            ctx.hsv_hists = [self._hsv_hist_3d(f, self.hist_bins) for f in ctx.frames_bgr]

        sims = [self._cosine(ctx.hsv_hists[i], ctx.hsv_hists[i + 1]) for i in range(K - 1)]
        mean_sim = float(np.mean(sims))
        R = float(np.clip(1.0 - mean_sim, 0.0, 1.0))
        return R, {"adjacent_sims": sims, "mean_sim": mean_sim, "R": R}


# ============================================================
# Spatial component (paper Eq.1)
# ============================================================
@dataclass(frozen=True)
class SpatialEntropy:
    """E_s(f): mean local entropy of Laplacian/Sobel response (Eq.1)."""
    name: str = "spatial_entropy"
    entropy_radius: int = 6
    spatial_mode: str = "laplacian"  # laplacian or sobel

    @staticmethod
    def _complexity_map(gray: np.ndarray, mode: str) -> np.ndarray:
        mode = mode.lower()
        if mode == "laplacian":
            return np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        if mode == "sobel":
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(sx * sx + sy * sy)
        raise ValueError(f"Unknown spatial mode: {mode}")

    def _entropy_one(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        comp = self._complexity_map(gray, self.spatial_mode)
        comp_u8 = cv2.normalize(comp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ent_map = sk_entropy(comp_u8, disk(int(self.entropy_radius)))
        return float(ent_map.mean())

    def compute(self, ctx: STECContext) -> Tuple[float, Dict[str, Any]]:
        K = len(ctx.frames_bgr)
        if K == 0:
            return 0.0, {"reason": "no_frames"}

        if ctx.spatial_entropies is None:
            ctx.spatial_entropies = [self._entropy_one(f) for f in ctx.frames_bgr]

        S = float(np.mean(ctx.spatial_entropies))
        return S, {"per_frame": ctx.spatial_entropies, "S": S}


# ============================================================
# Final STEC composer (paper Eq.7)
# ============================================================
@dataclass(frozen=True)
class STECComposer:
    temporal_positions: TemporalPositions
    temporal_entropy: TemporalEntropy
    spatial_entropy: SpatialEntropy
    redundancy: Redundancy
    use_span_penalty: bool = True  # T = Et*Ct if True else T = Et

    def compute(
        self,
        frames_bgr: Sequence[np.ndarray],
        frame_indices: Sequence[int],
        total_frames: Optional[int],
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        ctx = STECContext(frames_bgr=frames_bgr, frame_indices=frame_indices, total_frames=total_frames)

        _, dbg_pos = self.temporal_positions.compute(ctx)
        Et, dbg_et = self.temporal_entropy.compute(ctx)
        Ct = float(ctx.span or 0.0)

        S, dbg_s = self.spatial_entropy.compute(ctx)
        R, dbg_r = self.redundancy.compute(ctx)

        T = float(Et * Ct) if self.use_span_penalty else float(Et)
        STEC = float(S * T * R)

        out: Dict[str, Any] = {"S": S, "T": T, "R": R, "STEC": STEC}
        # Keep old-style debug keys + add Et/Ct for paper alignment
        if return_debug:
            out["debug"] = {
                "K": len(frames_bgr),
                "spatial_entropies": dbg_s.get("per_frame", []),
                "frame_positions_norm": (ctx.temporal_pos_norm.tolist() if ctx.temporal_pos_norm is not None else []),
                "span": Ct,                       # old key
                "temporal_coverage_raw": Et,       # old key (was T_raw) -> now Et
                "Et": float(Et),
                "Ct": float(Ct),
                "adjacent_sims": dbg_r.get("adjacent_sims", []),
                "mean_adjacent_sim": float(dbg_r.get("mean_sim", 0.0)),
                # optional deeper debug
                "pos_debug": dbg_pos,
                "Et_debug": dbg_et,
            }
        return out


# ============================================================
# Backward-compatible STECMetric (matches demo import)
# ============================================================
class STECMetric:
    """
    Backward-compatible wrapper around STECComposer.
    Keeps the same compute() signature and output keys:
      {S, T, R, STEC} (+ debug if requested)
    """

    def __init__(self, config: Optional[STECConfig] = None):
        self.config = config or STECConfig()
        self._composer = STECComposer(
            temporal_positions=TemporalPositions(use_rank_fallback=self.config.use_rank_fallback),
            temporal_entropy=TemporalEntropy(time_bins=int(self.config.time_bins)),
            spatial_entropy=SpatialEntropy(
                entropy_radius=int(self.config.entropy_radius),
                spatial_mode=str(self.config.spatial_mode),
            ),
            redundancy=Redundancy(hist_bins=tuple(self.config.hist_bins)),
            use_span_penalty=bool(self.config.use_span_penalty),
        )

    def compute(
        self,
        frames_bgr: Sequence[np.ndarray],
        frame_indices: Sequence[int],
        total_frames: Optional[int],
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        return self._composer.compute(frames_bgr, frame_indices, total_frames, return_debug=return_debug)

    def compute_from_folder(
        self,
        folder: str,
        total_frames: Optional[int] = None,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        frames, idxs = load_sampled_frames_from_folder(folder)
        return self.compute(frames_bgr=frames, frame_indices=idxs, total_frames=total_frames, return_debug=return_debug)


# ============================================================
# Backward-compatible functional API (keep names)
# ============================================================
def spatial_complexity_map(gray: np.ndarray, mode: str = "laplacian") -> np.ndarray:
    return SpatialEntropy._complexity_map(gray, mode)


def spatial_entropy(frame_bgr: np.ndarray, radius: int = 6, mode: str = "laplacian") -> float:
    se = SpatialEntropy(entropy_radius=radius, spatial_mode=mode)
    ctx = STECContext(frames_bgr=[frame_bgr], frame_indices=[1], total_frames=1)
    val, _ = se.compute(ctx)
    # val is mean over one frame -> same as per-frame entropy
    return float(se._entropy_one(frame_bgr))


def hsv_hist_3d(frame_bgr: np.ndarray, bins=(8, 8, 8)) -> np.ndarray:
    return Redundancy._hsv_hist_3d(frame_bgr, tuple(bins))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return Redundancy._cosine(a, b)


def temporal_coverage(frame_positions_norm: np.ndarray, num_bins: int = 8) -> float:
    # kept name; it returns temporal entropy (Et) like before.
    te = TemporalEntropy(time_bins=num_bins)
    ctx = STECContext(frames_bgr=[np.zeros((1, 1, 3), dtype=np.uint8)] * max(1, len(frame_positions_norm)),
                      frame_indices=list(range(1, max(1, len(frame_positions_norm)) + 1)),
                      total_frames=max(2, len(frame_positions_norm) + 1))
    ctx.temporal_pos_norm = np.asarray(frame_positions_norm, dtype=np.float32)
    Et, _ = te.compute(ctx)
    return float(Et)


def stec_score_from_frames(
    frames_bgr: list,
    frame_indices: list,
    total_frames: Optional[int],
    spatial_mode: str = "laplacian",
    entropy_radius: int = 6,
    time_bins: int = 8,
    hist_bins=(8, 8, 8),
) -> Dict[str, Any]:
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


# ============================================================
# I/O: load frames from folder (kept)
# ============================================================
def load_sampled_frames_from_folder(folder: str) -> Tuple[List[np.ndarray], List[int]]:
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
