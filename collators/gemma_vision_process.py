"""
Gemma3 vision processing — cached-features path.

Aligned with UniTime upstream (feature_offline.py for Qwen2-VL):
  - 2fps dense extraction + token compression (bilinear resize)
  - .pt file contains: feature [T, H', W', hidden_dim], frame_idx, sample_fps
  - combine_timestamps groups frames into chunks (CLIP_LENGTH=-1 auto)
"""
from typing import List, Optional, Tuple

import torch


def _generate_clip_lengths(t, clip_length):
    full_clips = t // clip_length
    remainder = t % clip_length
    result = [clip_length] * full_clips
    if remainder > 0:
        result.append(remainder)
    return result


def _combine_timestamps_gemma(feature, sampled_timestamps, num_clips=32, clip_length=-1):
    """Same logic as qwen_vision_process.combine_timestamps.

    Works with both [T, H, W, D] (4D, after token compression) and [T, 256, D] (3D, legacy).
    """
    T = feature.shape[0]
    assert len(sampled_timestamps) == T
    if clip_length <= 0:
        clip_length = max(T // num_clips, 1)
    sampled_timestamps_combine = sampled_timestamps[::int(clip_length)]
    combine_t_list = _generate_clip_lengths(T, clip_length)
    return feature, sampled_timestamps_combine, combine_t_list


def fetch_video_feature_only(ele: dict) -> Tuple[Optional[torch.Tensor], Optional[List[float]], Optional[List[int]]]:
    """Load cached Gemma3 features for one video clip.

    Returns:
        (feature, sampled_timestamps, combine_t_list) where
            feature: tensor [T, 256, hidden_dim] — sliced to the window
            sampled_timestamps: list of floats (after combine)
            combine_t_list: list of ints, frames per timestamp chunk
    """
    feat_path = ele["feature"]
    payload = torch.load(feat_path, map_location="cpu")
    feature = payload["feature"]  # [T_total, H', W', D] (4D) or [T_total, 256, D] (3D legacy)
    frame_idx = payload["frame_idx"]
    sample_fps = float(payload["sample_fps"])

    duration = float(ele.get("duration", 0))
    if duration <= 0:
        T = feature.shape[0]
        sampled_timestamps = [round(i / max(sample_fps, 1e-6), 1) for i in range(T)]
    else:
        T_total = feature.shape[0]
        sampled_timestamps = [
            round(i / max(T_total - 1, 1) * duration, 1) for i in range(T_total)
        ]

    video_start = float(ele.get("video_start", 0))
    video_end = float(ele.get("video_end", sampled_timestamps[-1] if sampled_timestamps else duration))
    keep_idx = [i for i, t in enumerate(sampled_timestamps) if video_start <= t <= video_end]
    if not keep_idx:
        diffs = [abs(t - video_start) for t in sampled_timestamps]
        keep_idx = [int(min(range(len(diffs)), key=lambda i: diffs[i]))]

    feature = feature[keep_idx]
    sampled_timestamps = [sampled_timestamps[i] for i in keep_idx]

    num_clips = int(ele.get("num_clips", 32))
    clip_length_raw = int(ele.get("clip_length", -1))
    if clip_length_raw > 0:
        clip_length = int(clip_length_raw * sample_fps / 2)
    else:
        clip_length = -1

    feature, sampled_timestamps, combine_t_list = _combine_timestamps_gemma(
        feature, sampled_timestamps, num_clips=num_clips, clip_length=clip_length
    )

    return feature, sampled_timestamps, combine_t_list


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
        sampled_timestamps_list: list of [list of float], one per video (after combine)
        combine_t_lists: list of [list of int], frames per chunk per video
    """
    feature_inputs = []
    sampled_timestamps_list = []
    combine_t_lists = []
    for video_item in extract_video_info(messages):
        feature, sampled_timestamps, combine_t_list = fetch_video_feature_only(video_item)
        feature_inputs.append(feature)
        sampled_timestamps_list.append(sampled_timestamps)
        combine_t_lists.append(combine_t_list)
    if not feature_inputs:
        return None, None, None
    return feature_inputs, sampled_timestamps_list, combine_t_lists
