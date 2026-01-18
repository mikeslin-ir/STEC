# utils/FrameSampler_Kanta.py
import os
import glob
import math
from typing import List, Optional

import cv2
import numpy as np
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter


class FrameSampler_Kanta:
    def __init__(self, no_of_frames: int = 30, use_uniform_on_fail: bool = True):
        """
        Initialize the frame sampler.
        Args:
            no_of_frames: desired number of frames to extract/save.
            use_uniform_on_fail: if True, fallback to uniform sampling when Katna fails.
        """
        self.no_of_frames = max(int(no_of_frames), 1)
        self.use_uniform_on_fail = use_uniform_on_fail
        self.vd = Video()

    # --------------------------- Public API ---------------------------

    def extract_frames(self, source_video_path: str, output_folder: str):
        """
        Try Katna keyframe extraction with fallbacks; if all fail or produce 0 frames,
        fallback to uniform sampling (OpenCV).
        """
        os.makedirs(output_folder, exist_ok=True)
        self.cleanup_partial_output(output_folder)  # always start clean

        # 1) Katna with fallback counts
        fallback_list = [self.no_of_frames, 10, 5, 1]
        katna_success = False

        for fallback_frames in fallback_list:
            print(f"[INFO] Katna: trying {fallback_frames} keyframes → {source_video_path}")
            diskwriter = KeyFrameDiskWriter(location=output_folder, file_ext=".jpg")
            try:
                self.vd.extract_video_keyframes(
                    no_of_frames=fallback_frames,
                    file_path=source_video_path,
                    writer=diskwriter
                )
                # Validate output
                self.rename_keyframes(output_folder)
                num = self.count_jpg(output_folder)
                if num > 0:
                    print(f"[SUCCESS] Katna extracted {num} frames")
                    katna_success = True
                    break
                else:
                    print("[WARNING] Katna returned 0 frames; retrying...")
                    self.cleanup_partial_output(output_folder)
            except Exception as e:
                print(f"[WARNING] Katna failed @ {fallback_frames} frames: {e}")
                self.cleanup_partial_output(output_folder)

        # 2) Uniform fallback if Katna not successful or produced 0 frames
        if not katna_success:
            if self.use_uniform_on_fail:
                print("[INFO] Falling back to uniform sampling...")
                ok, saved = self.uniform_sample_frames(
                    source_video_path=source_video_path,
                    output_folder=output_folder,
                    num_frames=self.no_of_frames
                )
                if not ok or saved == 0:
                    self.cleanup_partial_output(output_folder)
                    raise RuntimeError(
                        f"All attempts failed (Katna + uniform) for: {source_video_path}"
                    )
                print(f"[SUCCESS] Uniform sampling saved {saved} frames")
            else:
                raise RuntimeError(
                    f"Katna extraction failed and uniform fallback is disabled for: {source_video_path}"
                )

    # --------------------------- Helpers ---------------------------

    def count_jpg(self, folder_path: str) -> int:
        return len(glob.glob(os.path.join(folder_path, "*.jpg")))

    def rename_keyframes(self, folder_path: str):
        """
        Rename keyframes to frame_000000.jpg, frame_000001.jpg, ...
        Accept any '*.jpg' names that have a trailing integer or any order: we sort by mtime if needed.
        """
        files = glob.glob(os.path.join(folder_path, "*.jpg"))
        if not files:
            return

        def safe_index(path: str) -> int:
            # Try to parse trailing integer before .jpg (e.g., video_12.jpg → 12)
            stem = os.path.splitext(os.path.basename(path))[0]
            maybe = stem.split("_")[-1]
            return int(maybe) if maybe.isdigit() else math.inf  # non-numeric go to end

        files_idx = [f for f in files if safe_index(f) != math.inf]
        files_etc = [f for f in files if safe_index(f) == math.inf]

        files_idx.sort(key=safe_index)
        files_etc.sort(key=lambda p: os.path.getmtime(p))
        ordered = files_idx + files_etc

        for new_index, old_path in enumerate(ordered):
            new_name = f"frame_{new_index:06d}.jpg"
            new_path = os.path.join(folder_path, new_name)
            if os.path.abspath(old_path) != os.path.abspath(new_path):
                print(f"[RENAME] {os.path.basename(old_path)} → {new_name}")
                os.rename(old_path, new_path)

    def cleanup_partial_output(self, folder_path: str):
        jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
        for f in jpg_files:
            try:
                os.remove(f)
            except Exception:
                pass
        if jpg_files:
            print(f"[CLEANUP] Removed {len(jpg_files)} partially written .jpg files")

    # --------------------------- Uniform Fallback ---------------------------

    def uniform_sample_frames(
        self,
        source_video_path: str,
        output_folder: str,
        num_frames: Optional[int] = None,
        prefer_exact: bool = False,
    ) -> (bool, int):
        """
        Uniformly sample frames with OpenCV.
        Args:
            source_video_path: path to video
            output_folder: where to save 'frame_XXXXXX.jpg'
            num_frames: desired number of frames (defaults to self.no_of_frames)
            prefer_exact: if True, try harder to get exactly N frames (may be slower)
        Returns:
            (ok, saved_count)
        """
        os.makedirs(output_folder, exist_ok=True)
        self.cleanup_partial_output(output_folder)

        n = max(int(num_frames or self.no_of_frames), 1)

        cap = cv2.VideoCapture(source_video_path)
        if not cap.isOpened():
            print(f"[ERROR] OpenCV cannot open video: {source_video_path}")
            return False, 0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            # Some containers don't expose frame count; fallback to time-based stepping
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (fps + 1e-6)
            print(f"[WARNING] Unknown frame count (reported {total}). fps≈{fps:.2f}, duration≈{duration:.2f}s")
            # fall back to naive sequential read grabbing ~n frames
            return self._uniform_by_stream(cap, output_folder, n)

        # Build monotonically increasing indices in [0, total-1]
        if n >= total:
            indices = np.arange(total, dtype=np.int64)  # save all frames
        else:
            indices = np.linspace(0, total - 1, num=n, dtype=np.int64)

        saved = 0
        for i, idx in enumerate(indices.tolist()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                if prefer_exact:
                    # attempt small local search for nearest decodable frame
                    ok, frame = self._seek_nearby(cap, idx, total, window=3)
                if not ok or frame is None:
                    print(f"[WARN] Failed to read frame @ {idx}")
                    continue

            out_name = os.path.join(output_folder, f"frame_{saved:06d}.jpg")
            if not cv2.imwrite(out_name, frame):
                print(f"[WARN] Failed to write {out_name}")
                continue
            saved += 1

        cap.release()
        return (saved > 0), saved

    def _seek_nearby(self, cap: cv2.VideoCapture, idx: int, total: int, window: int = 3):
        """Try a few frames around idx to recover a readable frame."""
        for delta in range(1, window + 1):
            for sign in (-1, 1):
                j = int(np.clip(idx + sign * delta, 0, total - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, j)
                ok, frame = cap.read()
                if ok and frame is not None:
                    return True, frame
        return False, None

    def _uniform_by_stream(self, cap: cv2.VideoCapture, output_folder: str, num: int):
        """
        Fallback when CAP_PROP_FRAME_COUNT is unreliable:
        read sequentially and pick ~uniformly spaced frames.
        """
        # First pass: grab frames and store positions
        frames: List[np.ndarray] = []
        pos_list: List[int] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
            pos_list.append(idx)
            idx += 1

        total = len(frames)
        if total == 0:
            cap.release()
            return False, 0

        if num >= total:
            pick = np.arange(total, dtype=np.int64)
        else:
            pick = np.linspace(0, total - 1, num=num, dtype=np.int64)

        saved = 0
        for k, p in enumerate(pick.tolist()):
            out_name = os.path.join(output_folder, f"frame_{saved:06d}.jpg")
            if cv2.imwrite(out_name, frames[p]):
                saved += 1

        cap.release()
        return (saved > 0), saved


# --------------------------- Usage Example ---------------------------
if __name__ == "__main__":
    sampler = FrameSampler_Kanta(no_of_frames=30, use_uniform_on_fail=True)
    sampler.extract_frames(
        source_video_path="./tests/data/pos_video.mp4",
        output_folder="selectedframes"
    )
