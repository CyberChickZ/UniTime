"""
Extract Gemma4 vision features for UniTime training.

Mirrors extract_gemma_features.py but uses Gemma 4 instead of Gemma 3.
Requires the `UniTime-gemma4` conda env (transformers 5.x + torch 2.4+).

Output schema (same as the Gemma 3 / Qwen2VL extract scripts so the same
collator pipeline can read it):
    {
        "feature":     bf16 tensor [N, mm_tokens_per_image, hidden_dim]
        "frame_idx":   int64 tensor [N]   (raw video frame indices sampled)
        "sample_fps":  float
    }
"""
import os
# Same lesson as Gemma 3 extract — set CUDA_HOME and import torch first.
os.environ.setdefault("CUDA_HOME", "/usr/local/apps/cuda/12.1")

import faulthandler
faulthandler.enable()

import argparse

import torch
import decord
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# Requires transformers >= 5.0 (Gemma 4 added in 5.x).
from models.gemma4_vl import Gemma4VLMRForConditionalGeneration


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", required=True, type=str)
    ap.add_argument("--feat_root", required=True, type=str,
                    help="Output root; final dir is {feat_root}/{dataset_name}")
    ap.add_argument("--model_local_path", required=True, type=str)
    ap.add_argument("--dataset_name", default="gtea", type=str)
    ap.add_argument("--num_frames", default=32, type=int)
    ap.add_argument("--part", default=0, type=int)
    ap.add_argument("--num_parts", default=1, type=int)
    ap.add_argument("--gpu", default=0, type=int)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    out_dir = os.path.join(args.feat_root, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"loading Gemma4 from {args.model_local_path}...")
    model = Gemma4VLMRForConditionalGeneration.from_pretrained(
        args.model_local_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # avoid flash-attn op-builder gaps
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_local_path)

    valid_ext = (".mp4", ".avi", ".mkv", ".mov", ".webm")
    all_videos = sorted(
        f for f in os.listdir(args.video_root) if f.lower().endswith(valid_ext)
    )

    total = len(all_videos)
    part_size = total // args.num_parts
    s = args.part * part_size
    e = (args.part + 1) * part_size if args.part != args.num_parts - 1 else total
    subset = all_videos[s:e]
    print(f"part {args.part}/{args.num_parts}: {len(subset)} videos")

    with torch.no_grad():
        for filename in tqdm(subset):
            vid = os.path.splitext(filename)[0]
            video_path = os.path.join(args.video_root, filename)
            out_path = os.path.join(out_dir, f"{vid}.pt")
            if os.path.exists(out_path):
                continue

            try:
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            except Exception as ex:
                print(f"decord open failed for {video_path}: {ex}")
                continue

            total_frames = len(vr)
            video_fps = vr.get_avg_fps()
            nframes = min(args.num_frames, total_frames)
            frame_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
            sample_fps = nframes / total_frames * video_fps

            try:
                frames = vr.get_batch(frame_idx).asnumpy()
            except Exception as ex:
                print(f"decord read failed for {vid}: {ex}")
                continue

            pil_frames = [Image.fromarray(f) for f in frames]
            inputs = processor.image_processor(pil_frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)

            # Gemma4ForConditionalGeneration.get_image_features delegates to
            # self.model.get_image_features which returns BaseModelOutputWithPooling
            # whose .pooler_output is the post-projector embedding (the vision-soft
            # tokens in language space). For older Gemma4 builds the return type
            # may be a plain tensor — handle both.
            out = model.get_image_features(pixel_values)
            features = out.pooler_output if hasattr(out, "pooler_output") else out
            features = features.cpu()
            # features shape: (nframes, mm_tokens_per_image, text_hidden_size)

            torch.save(
                {
                    "feature": features,
                    "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
                    "sample_fps": float(sample_fps),
                },
                out_path,
            )


if __name__ == "__main__":
    main()
