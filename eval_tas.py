"""TAS 滑窗评估.

对每个 test 视频:
  1. 滑窗切分 (步长=WINDOW_SEC, 不重叠)
  2. 第一个窗口 context=[], 后续用前一窗口的预测 action 序列
  3. model.generate() 得到 COIN 格式文本
  4. parse → 加窗口偏移 → 拼接所有窗口 → 映射到逐帧标签
  5. 和 GT 逐帧标签算 Acc/Edit/F1@{10,25,50}

eval 输入中没有任何 GT 信息.
"""
import os
import re
import json
import argparse
import math

import torch
import numpy as np
from peft import PeftModel
from transformers import AutoProcessor, AutoConfig

from models.gemma4_vl import Gemma4VLMRForConditionalGeneration
from datasets_tas import GTEAWindowDataset


# ============================================================
# 参数
# ============================================================
ORIGINAL_FPS = 15
SAMPLE_FPS = 5
STEP = ORIGINAL_FPS // SAMPLE_FPS  # 3
WINDOW_SEC = 30
WINDOW_FRAMES = ORIGINAL_FPS * WINDOW_SEC  # 450
N_SAMPLE_FRAMES = SAMPLE_FPS * WINDOW_SEC  # 150


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--video_folder", required=True)
    ap.add_argument("--gt_folder", required=True)
    ap.add_argument("--feat_folder", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--spatial_pool_h", type=int, default=12)
    ap.add_argument("--spatial_pool_w", type=int, default=12)
    ap.add_argument("--output", default="./results/tas_eval.json")
    ap.add_argument("--gpu", type=int, default=0)
    return ap.parse_args()


# ============================================================
# 文本构造 (和 collator 一致, 但不含 GT)
# ============================================================
def build_prompt(tokenizer, config, context, n_frames, feature_chunk,
                 spatial_pool_grid, device):
    """构造推理 prompt (不加时间戳), 返回 input_ids + feature_inputs (已 pool)."""
    bos = tokenizer.bos_token or ""
    boi_id = config.boi_token_id
    eoi_id = config.eoi_token_id
    img_tok = config.image_token_id

    tpi = spatial_pool_grid[0] * spatial_pool_grid[1]

    def tok(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    if context:
        context_str = ", ".join(context)
        header = tok(f"{bos}<|turn>user\nPrevious actions: {context_str}\n")
    else:
        header = tok(f"{bos}<|turn>user\n")

    body = []
    for _ in range(n_frames):
        body.append(boi_id)
        body.extend([img_tok] * tpi)
        body.append(eoi_id)

    instr = tok("\nList all action segments with start and end timestamps.\n")
    turn = tok("<turn|>\n<|turn>model\n")

    input_ids = header + body + instr + turn
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)

    # Spatial pooling on CPU
    raw_tpi = feature_chunk.shape[1]
    D = feature_chunk.shape[2]
    src_h, src_w = 1, raw_tpi
    for i in range(2, int(math.sqrt(raw_tpi)) + 1):
        if raw_tpi % i == 0:
            src_h, src_w = i, raw_tpi // i
    tgt_h, tgt_w = spatial_pool_grid
    pooled = feature_chunk.reshape(n_frames, src_h, src_w, D).permute(0, 3, 1, 2).float()
    pooled = torch.nn.functional.interpolate(pooled, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
    pooled = pooled.permute(0, 2, 3, 1).reshape(n_frames * tgt_h * tgt_w, D).to(feature_chunk.dtype)
    feature_inputs = pooled.to(device)

    return input_ids, attn, feature_inputs


# ============================================================
# Parse model 输出
# ============================================================
def parse_prediction(text):
    """Parse 模型输出 → list of (label, start, end).
    支持 JSON 格式: [{"start": 0.0, "end": 0.4, "action": "open"}, ...]
    也兼容 COIN 格式: "take 0.0 1.7, open 2.3 6.0"
    """
    text = text.strip()
    # 尝试 JSON parse
    try:
        # 找到第一个 [ 和最后一个 ]
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx >= 0 and end_idx > start_idx:
            data = json.loads(text[start_idx:end_idx + 1])
            segments = []
            for item in data:
                if isinstance(item, dict) and "start" in item and "end" in item:
                    label = item.get("action") or item.get("label") or "unknown"
                    segments.append({"label": label, "start": float(item["start"]), "end": float(item["end"])})
            return segments
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: COIN 格式
    segments = []
    parts = text.rstrip(".").split(",")
    for part in parts:
        part = part.strip()
        if not part or part == "none":
            continue
        tokens = part.split()
        if len(tokens) >= 3:
            try:
                segments.append({"label": tokens[0], "start": float(tokens[1]), "end": float(tokens[2])})
            except ValueError:
                continue
    return segments


def extract_context_from_prediction(segments):
    """从预测 segments 提取 action 序列 (去重连续)."""
    dedup = []
    for seg in segments:
        if not dedup or dedup[-1] != seg["label"]:
            dedup.append(seg["label"])
    return dedup


# ============================================================
# Segments → 逐帧标签
# ============================================================
def segments_to_frame_labels(segments, total_frames, original_fps):
    """把 (label, start_sec, end_sec) 列表映射到逐帧标签.
    未覆盖的帧标记为 background.
    """
    frame_labels = ["background"] * total_frames
    for seg in segments:
        s_frame = int(seg["start"] * original_fps)
        e_frame = int(seg["end"] * original_fps)
        s_frame = max(0, min(s_frame, total_frames - 1))
        e_frame = max(0, min(e_frame, total_frames - 1))
        for f in range(s_frame, e_frame + 1):
            frame_labels[f] = seg["label"]
    return frame_labels


# ============================================================
# TAS Metrics
# ============================================================
def compute_metrics(pred_labels, gt_labels):
    """计算 Acc, Edit, F1@{10,25,50}."""
    assert len(pred_labels) == len(gt_labels)
    n = len(pred_labels)

    # Accuracy
    correct = sum(1 for p, g in zip(pred_labels, gt_labels) if p == g)
    acc = correct / n * 100

    # Segment-level: 合并连续相同标签
    def to_segments(labels):
        segs = []
        cur = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != cur:
                segs.append(cur)
                cur = labels[i]
                start = i
        segs.append(cur)
        return segs

    pred_segs = to_segments(pred_labels)
    gt_segs = to_segments(gt_labels)

    # Edit distance (normalized)
    def edit_score(pred, gt):
        m, n_ = len(pred), len(gt)
        dp = [[0] * (n_ + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n_ + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n_ + 1):
                if pred[i - 1] == gt[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return 1 - dp[m][n_] / max(m, n_)

    edit = edit_score(pred_segs, gt_segs) * 100

    # F1@k (overlap-based)
    def f1_at_k(pred_labels, gt_labels, k):
        pred_s = to_segment_list(pred_labels)
        gt_s = to_segment_list(gt_labels)
        tp, fp, fn = 0, 0, 0
        gt_matched = [False] * len(gt_s)
        for ps in pred_s:
            matched = False
            for j, gs in enumerate(gt_s):
                if gt_matched[j]:
                    continue
                if ps["label"] == gs["label"]:
                    overlap = max(0, min(ps["end"], gs["end"]) - max(ps["start"], gs["start"]))
                    union = max(ps["end"], gs["end"]) - min(ps["start"], gs["start"])
                    iou = overlap / union if union > 0 else 0
                    if iou >= k / 100:
                        tp += 1
                        gt_matched[j] = True
                        matched = True
                        break
            if not matched:
                fp += 1
        fn = sum(1 for m in gt_matched if not m)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1 * 100

    def to_segment_list(labels):
        segs = []
        cur = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != cur:
                segs.append({"label": cur, "start": start, "end": i - 1})
                cur = labels[i]
                start = i
        segs.append({"label": cur, "start": start, "end": len(labels) - 1})
        return segs

    f1_10 = f1_at_k(pred_labels, gt_labels, 10)
    f1_25 = f1_at_k(pred_labels, gt_labels, 25)
    f1_50 = f1_at_k(pred_labels, gt_labels, 50)

    return {"acc": acc, "edit": edit, "f1_10": f1_10, "f1_25": f1_25, "f1_50": f1_50}


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    spatial_pool_grid = (args.spatial_pool_h, args.spatial_pool_w)

    # 加载模型
    print(f"Loading base model: {args.base_model}")
    model = Gemma4VLMRForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    base_model = model.base_model.model
    base_model.set_spatial_pool(spatial_pool_grid)

    processor = AutoProcessor.from_pretrained(args.base_model)
    tokenizer = processor.tokenizer
    config = AutoConfig.from_pretrained(args.base_model)

    # 加载 test 视频
    split_ids = []
    with open(args.split_file) as f:
        split_ids = [l.strip().replace(".txt", "") for l in f if l.strip()]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    all_results = []
    all_metrics = []

    for vid in split_ids:
        print(f"\n=== {vid} ===")

        # 加载 GT 和 feature
        gt_path = os.path.join(args.gt_folder, f"{vid}.txt")
        with open(gt_path) as f:
            gt_labels = [l.strip() for l in f if l.strip()]
        total_frames = len(gt_labels)

        feat_path = os.path.join(args.feat_folder, f"{vid}.pt")
        feat_data = torch.load(feat_path, map_location="cpu", weights_only=False)
        feature = feat_data["feature"]  # [T, 264, 2560]

        # 滑窗推理
        all_pred_segments = []  # 绝对时间的预测 segments
        context = []  # 第一个窗口 context 为空

        n_windows = math.ceil(total_frames / WINDOW_FRAMES)
        for w in range(n_windows):
            w_start_frame = w * WINDOW_FRAMES
            w_end_frame = min(w_start_frame + WINDOW_FRAMES, total_frames)
            w_actual_frames = w_end_frame - w_start_frame

            # 采样帧索引 (整数)
            n_sample = w_actual_frames // STEP
            if n_sample == 0:
                continue
            sample_indices = [w_start_frame + i * STEP for i in range(n_sample)]

            # Feature 切片
            feature_chunk = feature[sample_indices]  # [n_sample, 264, 2560]

            # 构造 prompt (无 GT, 无时间戳)
            input_ids, attn, feat_inputs = build_prompt(
                tokenizer, config, context, n_sample, feature_chunk,
                spatial_pool_grid, device,
            )

            # Generate
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    feature_inputs=feat_inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
            new_tokens = gen_ids[0][input_ids.shape[1]:]
            pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Parse
            window_segments = parse_prediction(pred_text)

            # 相对时间 → 绝对时间
            w_offset = w_start_frame / ORIGINAL_FPS
            for seg in window_segments:
                seg["start"] += w_offset
                seg["end"] += w_offset
            all_pred_segments.extend(window_segments)

            # 下一窗口的 context
            context = extract_context_from_prediction(window_segments)

            print(f"  window {w}: [{w_start_frame/ORIGINAL_FPS:.1f}s-{w_end_frame/ORIGINAL_FPS:.1f}s] "
                  f"pred='{pred_text[:80]}...' context={context}")

        # 预测 segments → 逐帧标签
        pred_frame_labels = segments_to_frame_labels(all_pred_segments, total_frames, ORIGINAL_FPS)

        # 计算 metrics
        metrics = compute_metrics(pred_frame_labels, gt_labels)
        all_metrics.append(metrics)
        print(f"  {vid}: Acc={metrics['acc']:.1f} Edit={metrics['edit']:.1f} "
              f"F1@10={metrics['f1_10']:.1f} F1@25={metrics['f1_25']:.1f} F1@50={metrics['f1_50']:.1f}")

        all_results.append({
            "id": vid,
            "total_frames": total_frames,
            "pred_segments": all_pred_segments,
            "metrics": metrics,
        })

    # 汇总
    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print(f"\n=== Average ===")
    print(f"Acc={avg['acc']:.1f} Edit={avg['edit']:.1f} "
          f"F1@10={avg['f1_10']:.1f} F1@25={avg['f1_25']:.1f} F1@50={avg['f1_50']:.1f}")

    output = {"results": all_results, "average": avg}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
