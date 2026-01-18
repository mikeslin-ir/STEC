# utils/FrameSampler_STACFP.py
import os
import cv2
import json
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


class FrameSampler_STACFP:
    """
    Frame-wise Spatial-Temporal Adaptive Clustering Frame Proposal.

    - Features: 3D color histogram (HSV or Lab) + normalized time * gamma_time
    - Clustering: KMeans with K chosen via silhouette score
    - Output: one representative frame per cluster, saved as frame_{ORIG_FRAME_IDX:06d}.jpg
    - Metadata: strict RFC8259 JSON (no NaN/Inf; no numpy dtypes) written to stpac_meta.json
    """

    def __init__(
        self,
        frame_step: int = 10,                         # sample every Nth frame for feature extraction
        color_space: str = "HSV",                     # "HSV" or "Lab"
        hist_bins: Tuple[int, int, int] = (8, 8, 8),  # bins for each channel
        gamma_time: float = 1.0,                      # weight for normalized time feature
        k_min: int = 5,
        k_max: int = 30,
        resize_shorter_to: Optional[int] = 256,       # resize to speed up hist (None disables)
        random_state: int = 42,
        image_quality: int = 95                       # JPEG quality
    ):
        self.frame_step = max(1, int(frame_step))
        self.color_space = color_space.upper()
        assert self.color_space in {"HSV", "LAB"}, "color_space must be 'HSV' or 'Lab'"
        self.hist_bins = tuple(int(b) for b in hist_bins)
        self.gamma_time = float(gamma_time)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.resize_shorter_to = None if resize_shorter_to is None else int(resize_shorter_to)
        self.random_state = int(random_state)
        self.image_quality = int(np.clip(image_quality, 70, 100))

    # --------------------------- Public API ---------------------------

    def extract_frames(self, source_video_path: str, output_folder: str, is_saved_silhouette: bool = False) -> List[str]:
        """
        Run STPAC on a video and save one representative frame per cluster.

        Args:
            source_video_path: path to input video
            output_folder: folder to save sampled frames (created if missing)

        Returns:
            List of saved file paths (chronological by original frame index)
        """
        if not os.path.isfile(source_video_path):
            raise FileNotFoundError(f"Video not found: {source_video_path}")

        os.makedirs(output_folder, exist_ok=True)

        X, frame_idxs, times_sec, meta = self._extract_features(source_video_path)

        n = int(X.shape[0])
        if n == 0:
            raise RuntimeError("No frames extracted for features (check video/cv2).")
        if n == 1:
            saved = self._save_specific_frames(source_video_path, [int(frame_idxs[0])], output_folder)
            self._write_meta(output_folder, self._build_meta(source_video_path, meta, n, 1, [int(frame_idxs[0])], saved))
            return saved

        best_k, labels, kmeans, k_candidates, sil_scores = self._select_k_by_silhouette(X)
        reps_local = self._representatives_by_centroid(X, labels, kmeans.cluster_centers_)
        selected_frame_idxs = sorted(int(frame_idxs[i]) for i in reps_local)

        if is_saved_silhouette:
            import matplotlib.pyplot as plt

            # Ensure folder exists
            output_dir = os.path.join(output_folder, "_Silhouette_score_results")
            os.makedirs(output_dir, exist_ok=True)
            saved_res_img = os.path.join(output_dir, "silhouette_scores.png")
            print("Silhouette score: ", saved_res_img)

            plt.plot(k_candidates, sil_scores, marker='o')
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Silhouette score")
            plt.title("Silhouette score vs k")

            # Save instead of showing
            plt.savefig(saved_res_img, dpi=300, bbox_inches="tight")
            plt.close()  # close the figure to free memory

        saved_paths = self._save_specific_frames(source_video_path, selected_frame_idxs, output_folder)

        meta_obj = self._build_meta(
            source_video_path=source_video_path,
            meta=meta,
            sampled_n=n,
            chosen_k=int(best_k),
            selected_frame_indices=selected_frame_idxs,
            saved_paths=saved_paths,
        )
        self._write_meta(output_folder, meta_obj)

        print(f"[SUCCESS] Saved {len(saved_paths)} frames → {output_folder}")
        return saved_paths

    # --------------------------- Core steps ---------------------------

    def _extract_features(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isclose(fps, 0.0):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        frame_idxs, times_sec, hists = [], [], []
        read_idx = 0

        while True:
            ok = cap.grab()
            if not ok:
                break
            if read_idx % self.frame_step != 0:
                read_idx += 1
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                read_idx += 1
                continue

            frame = self._maybe_resize(frame)
            hist = self._color_histogram_3d(frame)
            hists.append(hist)

            frame_idxs.append(read_idx)
            times_sec.append(read_idx / fps)

            read_idx += 1

        cap.release()

        if len(hists) == 0:
            return np.empty((0, 0)), np.array([]), np.array([]), {"fps": float(fps), "total_frames": int(total_frames)}

        H = np.vstack(hists).astype(np.float32)   # [N, D_hist]
        H = normalize(H, norm="l2")

        duration = (total_frames - 1) / fps if total_frames > 1 else (max(times_sec) if times_sec else 1.0)
        t_norm = np.array(times_sec, dtype=np.float32) / max(duration, 1e-8)
        t_feat = (self.gamma_time * t_norm)[:, None]    # [N, 1]

        X = np.hstack([H, t_feat])                      # [N, D_hist + 1]
        meta = {"fps": float(fps), "total_frames": int(total_frames)}
        return X, np.array(frame_idxs, dtype=int), np.array(times_sec, dtype=np.float32), meta

    def _select_k_by_silhouette(self, X: np.ndarray):
        n = X.shape[0]
        k_candidates = [k for k in range(max(2, self.k_min), min(self.k_max, n - 1) + 1)]
        sil_scores = []

        if not k_candidates:
            # Fallback for tiny n
            k = 2 if n >= 2 else 1
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(X) if k >= 2 else None
            labels = km.labels_ if km is not None else np.zeros(n, dtype=int)
            return k, labels, km

        best = {"k": None, "score": -np.inf, "labels": None, "model": None}
        for k in k_candidates:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(X)
            try:
                score = silhouette_score(X, labels, metric="euclidean")
            except Exception:
                continue
            if score > best["score"]:
                best = {"k": k, "score": float(score), "labels": labels, "model": km}

            sil_scores.append(score)


        if best["k"] is None:
            # Degenerate case (e.g., identical points): pick a reasonable k
            k = min(max(2, self.k_min), max(2, n))
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(X)
            labels = km.labels_
            print("[INFO] Silhouette degenerate; falling back to K=", k)
            return k, labels, km

        print(f"[INFO] Chosen K={best['k']} (silhouette={best['score']:.4f})")
        return best["k"], best["labels"], best["model"], k_candidates, sil_scores

    def _representatives_by_centroid(self, X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> List[int]:
        reps = []
        for k in range(centers.shape[0]):
            idxs = np.where(labels == k)[0]
            if len(idxs) == 0:
                continue
            d = np.linalg.norm(X[idxs] - centers[k][None, :], axis=1)
            reps.append(int(idxs[np.argmin(d)]))
        return reps

    # --------------------------- I/O helpers ---------------------------

    def _save_specific_frames(
        self,
        video_path: str,
        frame_indices: List[int],
        output_folder: str
    ) -> List[str]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot reopen video for saving: {video_path}")

        saved_paths = []
        for fidx in sorted(int(i) for i in frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[WARN] Failed to read frame {fidx} for saving.")
                continue
            out_name = f"frame_{int(fidx):06d}.jpg"   # keep original frame number
            out_path = os.path.join(output_folder, out_name)
            ok = cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.image_quality])
            if ok:
                saved_paths.append(out_path)
                print(f"[SAVE] {out_name}")
            else:
                print(f"[WARN] Failed to write {out_name}")

        cap.release()
        return saved_paths

    # --------------------------- Metadata helpers ---------------------------

    def _build_meta(
        self,
        source_video_path: str,
        meta: Dict,
        sampled_n: int,
        chosen_k: int,
        selected_frame_indices: List[int],
        saved_paths: List[str],
    ) -> Dict:
        fps = float(meta.get("fps", 30.0))
        file_names = [os.path.basename(p) for p in saved_paths]
        frames_meta = [
            {
                "frame_index": int(idx),
                "timestamp_sec": float(idx / max(fps, 1e-8)),
                "file": fn,
            }
            for idx, fn in zip(selected_frame_indices, file_names)
        ]

        obj = {
            "schema": "stpac_meta@1.0",
            "video_path": str(source_video_path),
            "fps": fps,
            "total_frames": int(meta.get("total_frames", 0)),
            "sampled_frames": int(sampled_n),
            "frame_step": int(self.frame_step),
            "color_space": str(self.color_space),
            "hist_bins": list(self.hist_bins),
            "gamma_time": float(self.gamma_time),
            "k_min": int(self.k_min),
            "k_max": int(self.k_max),
            "chosen_k": int(chosen_k),
            "resize_shorter_to": (None if self.resize_shorter_to is None else int(self.resize_shorter_to)),
            "random_state": int(self.random_state),
            # Back-compat fields
            "selected_frame_indices": [int(i) for i in selected_frame_indices],
            "saved_files": file_names,
            # Explicit per-frame records
            "frames": frames_meta,
        }
        return obj

    def _to_python(self, obj):
        """Recursively convert numpy/scikit types to pure Python; map non-finite floats to None."""
        import numpy as _np
        if isinstance(obj, dict):
            return {str(k): self._to_python(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_python(x) for x in obj]
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, (_np.integer, )):
            return int(obj)
        if isinstance(obj, (_np.floating, )):
            x = float(obj)
            return x if _np.isfinite(x) else None
        return obj

    def _write_meta(self, folder: str, info: Dict):
        """Write JSON with strict encoding (no NaN/Inf) and only plain Python types."""
        meta_path = os.path.join(folder, "stpac_meta.json")
        clean = self._to_python(info)
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(clean, f, indent=2, ensure_ascii=False, allow_nan=False)
            print(f"[META] Wrote {meta_path}")
        except Exception as e:
            print(f"[WARN] Failed to write metadata: {e}")

    # --------------------------- Feature utils ---------------------------

    def _maybe_resize(self, img: np.ndarray) -> np.ndarray:
        if self.resize_shorter_to is None:
            return img
        h, w = img.shape[:2]
        s = min(h, w)
        if s <= self.resize_shorter_to:
            return img
        scale = self.resize_shorter_to / float(s)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _color_histogram_3d(self, img: np.ndarray) -> np.ndarray:
        if self.color_space == "HSV":
            cs = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            ranges = [0, 180, 0, 256, 0, 256]  # H, S, V
        else:  # "LAB"
            cs = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            ranges = [0, 256, 0, 256, 0, 256]  # L, a, b

        hist = cv2.calcHist(
            [cs],
            channels=[0, 1, 2],
            mask=None,
            histSize=list(self.hist_bins),
            ranges=ranges
        ).flatten().astype(np.float32)
        hist += 1e-6  # stability
        return hist


# --------------------------- Usage Example ---------------------------
if __name__ == "__main__":
    sampler = FrameSampler_STACFP(
        frame_step=10,
        color_space="HSV",
        hist_bins=(8, 8, 8),
        gamma_time=1.2,     # increase for stronger temporal grouping
        k_min=2,
        k_max=30,
        resize_shorter_to=256,
        random_state=42
    )
    saved = sampler.extract_frames(
        source_video_path="./tests/data/pos_video.mp4",
        output_folder="stpacfp_selectedframes",
        is_saved_silhouette=True
    )
    print(saved)
