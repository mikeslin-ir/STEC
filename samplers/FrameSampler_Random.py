import os
import cv2
import numpy as np
import json
from typing import Any, Dict, List


class FrameSampler_Random:
    def __init__(self, num_frames: int = 16, seed: int = 42):
        self.num_frames = num_frames
        self.seed = seed

    def extract_frames(self, video_path: str, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return

        rng = np.random.default_rng(self.seed)
        k = min(self.num_frames, total)
        indices = sorted(rng.choice(total, size=k, replace=False))

        saved = 0
        saved_indices: List[int] = []
        records: List[Dict[str, Any]] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            out = os.path.join(output_folder, f"frame_{saved:06d}.jpg")
            cv2.imwrite(out, frame)
            saved_indices.append(int(idx))
            records.append({
                "rank": int(saved),
                "frame_index": int(idx),
                "filename": os.path.basename(out),
            })
            saved += 1

        cap.release()

        # Write meta.json so downstream metrics can recover true indices.
        meta = {
            "sampler": "random",
            "seed": int(self.seed),
            "requested_k": int(self.num_frames),
            "total_frames": int(total),
            "index_mode": "true",
            "selected_frame_indices": [int(x) for x in indices],
            "selected_frame_indices_saved": saved_indices,
            "saved_count": int(saved),
            "records": records,
        }
        try:
            with open(os.path.join(output_folder, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            # Meta is best-effort; do not fail frame extraction.
            pass
