# utils/FrameSampler_Uniform.py
import os
import glob
import cv2


class FrameSampler_Uniform:
    def __init__(self, no_of_frames: int = 30):
        """
        Uniformly sample `no_of_frames` frames from a video.
        If the video has fewer frames, it will return as many as available.
        """
        self.no_of_frames = no_of_frames

    def extract_frames(self, source_video_path: str, output_folder: str) -> None:
        """
        Uniformly extract frames from the given video and save to the output folder.

        Args:
            source_video_path: Path to the input video file.
            output_folder: Folder where the extracted frames will be saved.
        """
        os.makedirs(output_folder, exist_ok=True)
        self._cleanup_partial_output(output_folder)

        cap = cv2.VideoCapture(source_video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {source_video_path}")
            return

        # Try to get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # If frame count is unknown/unreliable, count manually
        if total_frames <= 0:
            total_frames = self._count_frames_manually(cap)
            # reopen for random access
            cap.release()
            cap = cv2.VideoCapture(source_video_path)
            if not cap.isOpened():
                print(f"[ERROR] Cannot reopen video after counting: {source_video_path}")
                return

        if total_frames == 0:
            print(f"[ERROR] Video appears to have 0 frames: {source_video_path}")
            cap.release()
            return

        k = min(self.no_of_frames, total_frames)
        indices = self._uniform_indices(total_frames, k)

        print(f"[INFO] Total frames: {total_frames}, sampling: {k} frames")
        saved = 0
        for i, idx in enumerate(indices):
            # Some codecs don’t seek perfectly, but this works well in practice
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[WARNING] Failed to read frame at index {idx}")
                continue

            out_name = os.path.join(output_folder, f"frame_{i:06d}.jpg")
            ok = cv2.imwrite(out_name, frame)
            if not ok:
                print(f"[WARNING] Failed to write {out_name}")
                continue
            saved += 1

        cap.release()

        if saved == 0:
            print(f"[ERROR] No frames were saved for: {source_video_path}")
            self._cleanup_partial_output(output_folder)
        else:
            print(f"[SUCCESS] Saved {saved} frames to: {output_folder}")

    # ---------------- Internals ----------------

    @staticmethod
    def _uniform_indices(total_frames: int, k: int) -> list[int]:
        """
        Evenly spaced indices in [0, total_frames-1], length k.
        """
        if k <= 1:
            return [0]
        if k >= total_frames:
            return list(range(total_frames))
        # Avoid numpy to keep deps minimal
        last = total_frames - 1
        # Round to nearest int; ensure strictly non-decreasing and unique
        idxs = []
        prev = -1
        for i in range(k):
            pos = round(i * (last / (k - 1)))
            if pos == prev and pos < last:
                pos += 1  # nudge forward if rounding collided
            idxs.append(pos)
            prev = pos
        # Clip just in case
        idxs = [min(max(0, x), last) for x in idxs]
        return idxs

    @staticmethod
    def _count_frames_manually(cap: cv2.VideoCapture) -> int:
        """
        Fallback when CAP_PROP_FRAME_COUNT is not reliable.
        Sequentially read frames to count.
        """
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            count += 1
        return count

    @staticmethod
    def _cleanup_partial_output(folder_path: str) -> None:
        """
        Remove previously written .jpg files so a retry starts clean.
        """
        jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
        for f in jpg_files:
            try:
                os.remove(f)
            except OSError:
                pass
        if jpg_files:
            print(f"[CLEANUP] Removed {len(jpg_files)} existing .jpg files")


# Usage Example
if __name__ == "__main__":
    sampler = FrameSampler_Uniform(no_of_frames=30)
    sampler.extract_frames(
        source_video_path="./tests/data/pos_video.mp4",
        output_folder="selectedframes_uniform"
    )
