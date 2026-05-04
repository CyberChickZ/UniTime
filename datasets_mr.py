import os
import json
from PIL import Image
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset
import random 
import pandas as pd

class VideoCentricDataset(Dataset):
    """
    Dataset for supervised fine-tuning 
    """

    def __init__(
        self,
        data_path: str,
        video_folder: Optional[str] = None,
        feat_folder: Optional[str] = None,
        fps: int = 2,
        split='train',
        num_clips=32,
        clip_length=-1,
        model_family_id: str = "qwen2-vl",
    ) -> None:
        super(VideoCentricDataset, self).__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        self.video_folder = video_folder
        self.feat_folder = feat_folder
        self.fps = fps
        self.is_text_only = [
            False
            for source in self.list_data_dict
        ]
        self.split = split
        self.num_clips = num_clips
        self.clip_length = clip_length
        # `gemma3` triggers a different message-construction path for the new
        # Gemma3 collator (collators/gemma3_vl.py); default qwen2-vl keeps the
        # original behavior unchanged.
        self.model_family_id = model_family_id

    def __len__(self) -> int:
        return len(self.list_data_dict)
    
    def construct_messages_mr_fps(self, video_path, feature_path, fps, querys, temporal_windows, retrieval_segment, retrieval_mode):
        if retrieval_mode == 'mr_seg':
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": f"{video_path}", "fps": fps, "video_start": retrieval_segment[0], "video_end": retrieval_segment[1], 
                            "feature": f"{feature_path}", "num_clips": self.num_clips, "clip_length": self.clip_length, "temporal_windows":temporal_windows},
                        {"type": "text", "text": f"This is a sequence interleaved with timestamps and frames. Your task is to identify the specific timestamp(s) when the given query appears."}
                    ]
                },
            ]
        elif retrieval_mode == 'mr':
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": f"{video_path}", "fps": fps, "video_start": retrieval_segment[0], "video_end": retrieval_segment[1]},
                        {"type": "text", "text": f"This is a sequence interleaved with timestamps and frames. Your task is to identify the temporal window (start and end timestamps) when the given query appears."}
                    ]
                },
            ]
        
        for query in querys:
            message.append(
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Query:{query}\nAnswer: "}
                ]
            }
            )
        return message
    

    def __getitem__(self, i) -> Dict[str, List]:
        source = self.list_data_dict[i]
        qid = source["qid"]
        vid = source["id"]
        annos = source["annos"]
        retrieval_mode = source["mode"]

        video_start = source.get("video_start", 0)
        video_end = source.get("video_end", source["duration"])

        temporal_window = [anno["window"] for anno in annos]
        query = [anno["query"] for anno in annos]
        duration = video_end - video_start

        retrieval_segment = [video_start, video_end]
        
        video_path = source.get("video_path", None)
        if video_path is None and self.video_folder:
            if 'tacos' in self.video_folder:
                video_path = os.path.join(self.video_folder,f"{vid}.avi")
            else:
                video_path = os.path.join(self.video_folder,f"{vid}.mp4")

        feature_path = source.get("feature_path", None)
        if self.feat_folder is not None and feature_path is None:
            feature_path = os.path.join(self.feat_folder,f"{vid}.pt")

        if self.model_family_id == "gemma3":
            message = self.construct_messages_gemma3(
                video_path=video_path,
                feature_path=feature_path,
                querys=query,
                temporal_windows=temporal_window,
                retrieval_segment=retrieval_segment,
                duration=duration,
            )
        else:
            message = self.construct_messages_mr_fps(video_path=video_path, feature_path=feature_path, fps=self.fps, querys=query, temporal_windows=temporal_window,
                                                     retrieval_segment=retrieval_segment, retrieval_mode=retrieval_mode)

        return {"message":message, "split":self.split, "temporal_window":temporal_window, "mode":retrieval_mode, "qid":qid, "duration":duration}

    def construct_messages_gemma3(self, video_path, feature_path, querys, temporal_windows, retrieval_segment, duration):
        """Build a minimal message structure for the Gemma3 collator."""
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "feature": feature_path,
                        "video_start": retrieval_segment[0],
                        "video_end": retrieval_segment[1],
                        "duration": duration,
                        "num_clips": self.num_clips,
                        "clip_length": self.clip_length,
                    }
                ],
            },
        ]
        for q in querys:
            message.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Query:{q}\nAnswer: "}],
                }
            )
        return message