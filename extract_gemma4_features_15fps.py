"""提取 Gemma4 vision features — 15fps 全帧, 不压缩.

每帧 → image_processor → vision_tower → embed_vision → pooler_output [2520, 2560]
存储: {video_id}.pt = {"feature": [T, 2520, 2560], "frame_count": T, "original_fps": 15}

T = 视频总帧数 (原始 15fps), 不做任何采样/压缩.
spatial pooling 在训练时由模型 forward 做, 不在此处做.
"""
import os
os.environ.setdefault("CUDA_HOME", "/usr/local/apps/cuda/12.1")

import argparse
import torch
import decord
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from models.gemma4_vl import Gemma4VLMRForConditionalGeneration


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--feat_root", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--gpu", default=0, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.feat_root, exist_ok=True)

    model = Gemma4VLMRForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    videos = sorted(f for f in os.listdir(args.video_root)
                    if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov")))

    with torch.no_grad():
        for fname in tqdm(videos):
            vid = os.path.splitext(fname)[0]
            out_path = os.path.join(args.feat_root, f"{vid}.pt")
            if os.path.exists(out_path):
                print(f"  skip {vid} (exists)")
                continue

            vr = decord.VideoReader(os.path.join(args.video_root, fname), ctx=decord.cpu(0))
            total_frames = len(vr)
            video_fps = vr.get_avg_fps()

            # 读取所有帧 (原始帧率, 不采样)
            all_frames = vr.get_batch(list(range(total_frames))).asnumpy()
            pil_frames = [Image.fromarray(f) for f in all_frames]

            # 分 batch 提取 pooler_output
            all_features = []
            for i in range(0, total_frames, args.batch_size):
                batch_imgs = pil_frames[i:i + args.batch_size]
                inputs = processor.image_processor(batch_imgs, return_tensors="pt")
                px = inputs["pixel_values"].to(device, torch.bfloat16)
                pos_ids = inputs.get("image_position_ids")
                if pos_ids is not None:
                    pos_ids = pos_ids.to(device)
                out = model.get_image_features(px, pos_ids)
                po = out.pooler_output  # [batch * 2520, 2560]
                n_batch = len(batch_imgs)
                tpi = po.shape[0] // n_batch
                po = po.reshape(n_batch, tpi, -1).cpu()  # [batch, 2520, 2560]
                all_features.append(po)

            features = torch.cat(all_features, dim=0)  # [T, 2520, 2560]

            torch.save({
                "feature": features,
                "frame_count": total_frames,
                "original_fps": video_fps,
            }, out_path)

            print(f"  {vid}: {total_frames} frames, feature={tuple(features.shape)}, "
                  f"fps={video_fps:.1f}, size={os.path.getsize(out_path)/1e6:.0f}MB")


if __name__ == "__main__":
    main()
