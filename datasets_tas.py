"""GTEA 滑窗 Temporal Action Segmentation Dataset.

每次 __getitem__ 从一个视频中随机取 30s 窗口，返回:
  - context: 窗口前的 action 序列 (去 background, 去重连续)
  - gt_text: 窗口内的 action segments (COIN 格式, 相对时间)
  - sample_indices: 90 个采样帧的原始帧索引

全部帧索引运算为整数，避免浮点精度问题。
时间只在最后生成 GT 文本时才从整数帧偏移转为 float: round(offset / fps, 1)。
"""
import os
import random
from typing import Dict, List, Optional

from torch.utils.data import Dataset


class GTEAWindowDataset(Dataset):

    ORIGINAL_FPS = 15        # GTEA 原始视频帧率
    SAMPLE_FPS = 3           # 采样帧率
    STEP = ORIGINAL_FPS // SAMPLE_FPS  # = 5, 原始帧里每5帧取1帧
    WINDOW_SEC = 30          # 窗口长度 (秒)
    WINDOW_FRAMES = ORIGINAL_FPS * WINDOW_SEC  # = 450, 窗口在原始帧下的长度
    N_SAMPLE_FRAMES = SAMPLE_FPS * WINDOW_SEC  # = 90, 窗口内采样帧数

    def __init__(
        self,
        video_folder: str,
        gt_folder: str,
        split_file: str,
        split: str = "train",
    ) -> None:
        super().__init__()
        self.video_folder = video_folder
        self.gt_folder = gt_folder
        self.split = split

        video_ids = self._load_split(split_file)
        self.entries = []
        for vid in video_ids:
            gt_path = os.path.join(gt_folder, f"{vid}.txt")
            labels = self._load_gt(gt_path)
            # 视频必须 >= 30s (450 帧 @15fps)
            if len(labels) < self.WINDOW_FRAMES:
                continue
            video_path = None
            for ext in (".mp4", ".avi"):
                p = os.path.join(video_folder, f"{vid}{ext}")
                if os.path.exists(p):
                    video_path = p
                    break
            if video_path is None:
                continue
            self.entries.append({
                "id": vid,
                "video_path": video_path,
                "labels": labels,          # 逐帧标签 (原始 15fps)
                "total_frames": len(labels),
            })

        # TrainerWithCustomSampler 需要此属性 (所有 entry 都有视频, 无纯文本)
        self.is_text_only = [False] * len(self.entries)

    def _load_split(self, path: str) -> List[str]:
        with open(path) as f:
            return [line.strip().replace(".txt", "") for line in f if line.strip()]

    def _load_gt(self, path: str) -> List[str]:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]
        labels = entry["labels"]
        total_frames = entry["total_frames"]

        # Step 1: 随机窗口起点 (整数帧索引)
        max_s = total_frames - self.WINDOW_FRAMES
        s_frame = random.randint(0, max_s)

        # Step 2: 90 个采样帧索引 (全整数: s_frame + i * 5)
        sample_indices = [s_frame + i * self.STEP for i in range(self.N_SAMPLE_FRAMES)]

        # Step 3: 查每个采样帧的 GT 标签
        sampled_labels = [labels[idx] for idx in sample_indices]

        # Step 4: 合并连续相同标签 → segments (帧偏移, 整数)
        segments_raw = []
        current = sampled_labels[0]
        start_i = 0
        for i in range(1, self.N_SAMPLE_FRAMES):
            if sampled_labels[i] != current:
                segments_raw.append({
                    "label": current,
                    "start_offset": start_i * self.STEP,      # 整数
                    "end_offset": (i - 1) * self.STEP,        # 整数
                })
                current = sampled_labels[i]
                start_i = i
        segments_raw.append({
            "label": current,
            "start_offset": start_i * self.STEP,
            "end_offset": (self.N_SAMPLE_FRAMES - 1) * self.STEP,
        })

        # Step 5: 生成 GT 文本 — 去 background, 相对时间, COIN 格式
        # 时间 = round(帧偏移 / 原始fps, 1), 范围 [0, 29.7]
        gt_segments = []
        for seg in segments_raw:
            if seg["label"] == "background":
                continue
            t_start = round(seg["start_offset"] / self.ORIGINAL_FPS, 1)
            t_end = round(seg["end_offset"] / self.ORIGINAL_FPS, 1)
            gt_segments.append({"label": seg["label"], "start": t_start, "end": t_end})

        gt_text = ", ".join(
            f"{seg['label']} {seg['start']} {seg['end']}" for seg in gt_segments
        )
        if not gt_text:
            gt_text = "none"

        # Step 6: 前文 context — 窗口前的 action 序列, 去 background, 去重连续
        context_indices = list(range(0, s_frame, self.STEP))
        context_dedup = []
        for ci in context_indices:
            lab = labels[ci]
            if lab != "background" and (not context_dedup or context_dedup[-1] != lab):
                context_dedup.append(lab)

        # Step 7: 每帧相对时间戳 (只在此处从整数算出, 用于 collator 交错)
        frame_timestamps = [
            round(i * self.STEP / self.ORIGINAL_FPS, 1)
            for i in range(self.N_SAMPLE_FRAMES)
        ]

        return {
            "id": entry["id"],
            "video_path": entry["video_path"],
            "sample_indices": sample_indices,    # 90 个原始帧索引 (整数)
            "context": context_dedup,            # 前文 action 序列
            "gt_text": gt_text,                  # COIN 格式 GT
            "frame_timestamps": frame_timestamps,# 每帧相对时间 [0.0, 0.3, ..., 29.7]
            "split": self.split,
        }
