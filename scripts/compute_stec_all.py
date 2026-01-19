import os, json, csv
import cv2
from metrics.stec_metric import (
    load_sampled_frames_from_folder,
    stec_score_from_frames,
)


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n if n > 0 else None


def compute_all(video_list, video_dir, frames_root, out_json):
    results = {}

    for vid in video_list:
        vpath = os.path.join(video_dir, f"{vid}.mp4")
        total = get_total_frames(vpath)
        results[vid] = {}

        for sampler in ["uniform", "random", "katna", "stacfp"]:
            folder = os.path.join(frames_root, sampler, vid)
            if not os.path.isdir(folder):
                continue
            frames, idxs = load_sampled_frames_from_folder(folder)
            res = stec_score_from_frames(frames, idxs, total_frames=total)
            results[vid][sampler] = res

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
