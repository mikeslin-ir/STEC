import os
import cv2
import numpy as np


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
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            out = os.path.join(output_folder, f"frame_{saved:06d}.jpg")
            cv2.imwrite(out, frame)
            saved += 1

        cap.release()
