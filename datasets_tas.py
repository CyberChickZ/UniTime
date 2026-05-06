import os
import random
from typing import Dict, List, Optional

from torch.utils.data import Dataset


class GTEAWindowDataset(Dataset):
    """GTEA Temporal Action Segmentation with sliding window.

    Each __getitem__ returns a random 30s window from one video,
    with context (previous action sequence) and GT segments.
    All frame indexing is integer-only to avoid floating point issues.
    """

    ORIGINAL_FPS = 15
    SAMPLE_FPS = 3
    STEP = ORIGINAL_FPS // SAMPLE_FPS  # 5
    WINDOW_SEC = 30
    WINDOW_FRAMES = ORIGINAL_FPS * WINDOW_SEC  # 450
    N_SAMPLE_FRAMES = SAMPLE_FPS * WINDOW_SEC  # 90

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
                "labels": labels,
                "total_frames": len(labels),
            })

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

        # Step 1: random window start (integer frame index)
        max_s = total_frames - self.WINDOW_FRAMES
        s_frame = random.randint(0, max_s)

        # Step 2: sample indices (all integers)
        sample_indices = [s_frame + i * self.STEP for i in range(self.N_SAMPLE_FRAMES)]

        # Step 3: GT labels for sampled frames
        sampled_labels = [labels[idx] for idx in sample_indices]

        # Step 4: merge consecutive same labels → segments (integer offsets)
        segments_raw = []
        current = sampled_labels[0]
        start_i = 0
        for i in range(1, self.N_SAMPLE_FRAMES):
            if sampled_labels[i] != current:
                segments_raw.append({
                    "label": current,
                    "start_offset": start_i * self.STEP,
                    "end_offset": (i - 1) * self.STEP,
                })
                current = sampled_labels[i]
                start_i = i
        segments_raw.append({
            "label": current,
            "start_offset": start_i * self.STEP,
            "end_offset": (self.N_SAMPLE_FRAMES - 1) * self.STEP,
        })

        # Step 5: GT text — drop background, relative time, COIN format
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

        # Step 6: context — actions before window, drop background, dedup consecutive
        context_indices = list(range(0, s_frame, self.STEP))
        context_dedup = []
        for ci in context_indices:
            lab = labels[ci]
            if lab != "background" and (not context_dedup or context_dedup[-1] != lab):
                context_dedup.append(lab)

        # Step 7: per-frame relative timestamps (only computed here, from integers)
        frame_timestamps = [
            round(i * self.STEP / self.ORIGINAL_FPS, 1)
            for i in range(self.N_SAMPLE_FRAMES)
        ]

        return {
            "id": entry["id"],
            "video_path": entry["video_path"],
            "sample_indices": sample_indices,
            "context": context_dedup,
            "gt_text": gt_text,
            "frame_timestamps": frame_timestamps,
            "split": self.split,
        }
