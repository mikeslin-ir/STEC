import os
from samplers.FrameSampler_Uniform import FrameSampler_Uniform
from samplers.FrameSampler_Random import FrameSampler_Random
from samplers.FrameSampler_Kanta import FrameSampler_Kanta
from samplers.FrameSampler_STACFP import FrameSampler_STACFP


def build_samplers(num_frames: int = 16):
    """Factory to create fresh sampler instances.

    NOTE: Keep these arguments aligned with each sampler's __init__ signature.
    """
    return {
        "uniform": FrameSampler_Uniform(no_of_frames=num_frames),
        "random": FrameSampler_Random(num_frames=num_frames),
        "katna": FrameSampler_Kanta(no_of_frames=num_frames),
        "stacfp": FrameSampler_STACFP(frame_step=10),
    }


def extract_all(video_list, video_dir, out_root, num_frames: int = 16):
    for vid in video_list:
        vpath = os.path.join(video_dir, f"{vid}.mp4")
        samplers = build_samplers(num_frames=num_frames)
        for name, sampler in samplers.items():
            out_dir = os.path.join(out_root, name, vid)
            if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
                continue
            sampler.extract_frames(vpath, out_dir)
