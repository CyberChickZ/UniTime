"""
Gemma3 vision processing — cached-features path only.

Mirrors collators/qwen_vision_process.py:fetch_video / process_vision_info but
much simpler:
  - We only support pre-extracted features stored as .pt files (the path used
    by extract_gemma_features.py).
  - No live frame loading / no fps math / no Qwen smart_resize.
  - Each .pt file contains:
        feature:    bf16 tensor [num_frames, 256, hidden_dim] (post-projector)
        frame_idx:  int64 tensor [num_frames]
        sample_fps: float
  - For UniTime mr_seg, we slice the saved frames to the [video_start, video_end]
    window (in seconds) and return:
        feature_inputs:        sliced feature tensor
        sampled_timestamps:    per-frame timestamp in seconds (rounded to 0.1)
"""
from typing import List, Optional, Tuple

import torch


def fetch_video_feature_only(ele: dict) -> Tuple[Optional[torch.Tensor], Optional[List[float]]]:
    """Load cached Gemma3 features for one video clip.

    Args:
        ele: dict with keys
            feature:     str path to .pt file
            video_start: float seconds (default 0)
            video_end:   float seconds (default video duration)
            (optional) num_clips, clip_length — IGNORED for Gemma3 phase-2
                       (the upstream Qwen path uses these for combine_timestamps;
                       Gemma3 keeps every frame as-is to preserve mr_seg target
                       construction sanity)

    Returns:
        (feature, sampled_timestamps) where
            feature: tensor [T, 256, hidden_dim] — sliced to the window
            sampled_timestamps: list of floats, len == T
    """
    feat_path = ele["feature"]
    payload = torch.load(feat_path, map_location="cpu")
    feature = payload["feature"]  # [T_total, 256, hidden]
    frame_idx = payload["frame_idx"]  # [T_total], frame indices into raw video
    sample_fps = float(payload["sample_fps"])

    # Convert frame_idx (frame numbers in raw video) to seconds.
    # extract_gemma_features.py records frame_idx as the actual sampled video
    # frame numbers and sample_fps as nframes / total_frames * video_fps.
    # We can recover per-frame timestamps as frame_idx / video_fps. We don't
    # have video_fps directly here, so we reconstruct it from sample_fps and
    # the total duration in the source video.
    duration = float(ele.get("duration", 0))
    if duration <= 0:
        # Fallback: treat sample_fps as the sampling rate and use uniform spacing.
        T = feature.shape[0]
        sampled_timestamps = [round(i / max(sample_fps, 1e-6), 1) for i in range(T)]
    else:
        T_total = feature.shape[0]
        # Uniform sampling between 0 and duration
        sampled_timestamps = [
            round(i / max(T_total - 1, 1) * duration, 1) for i in range(T_total)
        ]

    # Slice to [video_start, video_end]
    video_start = float(ele.get("video_start", 0))
    video_end = float(ele.get("video_end", sampled_timestamps[-1] if sampled_timestamps else duration))
    keep_idx = [i for i, t in enumerate(sampled_timestamps) if video_start <= t <= video_end]
    if not keep_idx:
        # Defensive: keep at least the closest frame to video_start
        diffs = [abs(t - video_start) for t in sampled_timestamps]
        keep_idx = [int(min(range(len(diffs)), key=lambda i: diffs[i]))]

    feature = feature[keep_idx]
    sampled_timestamps = [sampled_timestamps[i] for i in keep_idx]

    return feature, sampled_timestamps


def extract_video_info(messages):
    """Walk a UniTime-style message list and yield each video item."""
    if isinstance(messages[0], dict):
        messages = [messages]
    for conversation in messages:
        for message in conversation:
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video":
                        yield item


def process_vision_info_gemma3(messages):
    """Gemma3-equivalent of qwen_vision_process.process_vision_info.

    Returns:
        feature_inputs: list of tensors [T, 256, hidden_dim], one per video
        sampled_timestamps_list: list of [list of float], one per video
    """
    feature_inputs = []
    sampled_timestamps_list = []
    for video_item in extract_video_info(messages):
        feature, sampled_timestamps = fetch_video_feature_only(video_item)
        feature_inputs.append(feature)
        sampled_timestamps_list.append(sampled_timestamps)
    if not feature_inputs:
        return None, None
    return feature_inputs, sampled_timestamps_list
