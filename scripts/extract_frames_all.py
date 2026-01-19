import os
from samplers.FrameSampler_Uniform import FrameSampler_Uniform
from samplers.FrameSampler_Random import FrameSampler_Random
from samplers.FrameSampler_Kanta import FrameSampler_Kanta
from samplers.FrameSampler_STACFP import FrameSampler_STACFP


SAMPLERS = {
    "uniform": FrameSampler_Uniform(num_frames=16),
    "random": FrameSampler_Random(num_frames=16),
    "katna": FrameSampler_Kanta(no_of_frames=16),
    "stacfp": FrameSampler_STACFP(frame_step=10),
}


def extract_all(video_list, video_dir, out_root):
    for vid in video_list:
        vpath = os.path.join(video_dir, f"{vid}.mp4")
        for name, sampler in SAMPLERS.items():
            out_dir = os.path.join(out_root, name, vid)
            if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
                continue
            sampler.extract_frames(vpath, out_dir)
