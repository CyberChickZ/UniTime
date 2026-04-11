"""
Compute standard TAS metrics (ASFormer/MS-TCN++ protocol) from UniTime eval results.

Takes our 32-frame predictions, upsamples to original video frame rate via
nearest-neighbor, then computes:
  - Frame-wise accuracy (Acc)
  - Segmental F1 @ IoU thresholds {10%, 25%, 50%}
  - Edit distance (normalized Levenshtein on segment sequences)

All metrics match the EXACT implementation from MS-TCN++ eval.py
(sj-li/MS-TCN2). Background class is excluded from F1 and Edit
computations (standard TAS convention).

Usage:
    python eval_tas_metrics.py \
        --eval_json ./results/gemma3_gtea_run1/eval.json \
        --test_json .../data/gtea/annot/test.json \
        --video_folder /nfs/hpc/dgx2-4/data/TAS_videos/gtea \
        --mode timestamp \
        --output ./results/gemma3_gtea_run1/tas_metrics.json
"""
import os
os.environ.setdefault("CUDA_HOME", "/usr/local/apps/cuda/12.1")

import argparse
import json
import re
from collections import defaultdict

import numpy as np

try:
    import decord
except ImportError:
    decord = None

TS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*seconds")

# v2 descriptive → v1 single word
V2_TO_V1 = {
    "The person takes an object or ingredient from the workspace.": "take",
    "The person puts down an object or places an ingredient on the surface.": "put",
    "The person opens a container, jar, or package.": "open",
    "The person closes a container, jar, or package.": "close",
    "The person pours liquid or contents from one container into another.": "pour",
    "The person spreads a substance onto bread or a surface with a knife.": "spread",
    "The person stirs the contents inside a bowl, cup, or pan.": "stir",
    "The person scoops contents out of a container using a spoon or utensil.": "scoop",
    "The person shakes a container to mix its contents.": "shake",
    "The person folds or wraps food items together.": "fold",
    "No specific cooking action is being performed.": "background",
}

BG_CLASS = ["background"]


# ========== MS-TCN++ / ASFormer metric functions (verbatim logic) ==========

def get_labels_start_end_time(frame_wise_labels, bg_class=BG_CLASS):
    """Convert per-frame labels to segment list (excluding background)."""
    labels, starts, ends = [], [], []
    last_label = frame_wise_labels[0]
    if last_label not in bg_class:
        labels.append(last_label)
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(len(frame_wise_labels))
    return labels, starts, ends


def levenstein(p, y, norm=True):
    m, n = len(p), len(y)
    D = np.zeros([m + 1, n + 1])
    for i in range(m + 1):
        D[i, 0] = i
    for j in range(n + 1):
        D[0, j] = j
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)
    if norm:
        score = (1 - D[-1, -1] / max(m, n)) * 100 if max(m, n) > 0 else 100.0
    else:
        score = D[-1, -1]
    return score


def edit_score(recognized, ground_truth):
    P, _, _ = get_labels_start_end_time(recognized)
    Y, _, _ = get_labels_start_end_time(ground_truth)
    return levenstein(P, Y, norm=True)


def f_score(recognized, ground_truth, overlap):
    p_label, p_start, p_end = get_labels_start_end_time(recognized)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth)

    tp, fp = 0, 0
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * np.array(
            [p_label[j] == y_label[x] for x in range(len(y_label))]
        )
        idx = np.array(IoU).argmax()
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - int(sum(hits))
    return float(tp), float(fp), float(fn)


# ========== Our prediction parsing ==========

def parse_timestamp_hits(pred_text, sampled_timestamps, tol=0.5):
    pred_ts = set(round(float(x), 1) for x in TS_RE.findall(pred_text))
    hit = set()
    for fi, t in enumerate(sampled_timestamps):
        if any(abs(t - pt) <= tol for pt in pred_ts):
            hit.add(fi)
    return hit


def parse_binary_hits(pred_text, num_frames=32):
    clean = "".join(c for c in pred_text.strip() if c in "01")
    return {fi for fi in range(min(len(clean), num_frames)) if clean[fi] == "1"}


def get_video_info(video_path):
    """Get fps and total_frames from video file."""
    if decord is None:
        return 15.0, 900  # fallback
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        return vr.get_avg_fps(), len(vr)
    except Exception:
        return 15.0, 900


def upsample_labels(sparse_labels, num_sparse, num_dense):
    """Nearest-neighbor upsample from num_sparse to num_dense frames."""
    dense = []
    for i in range(num_dense):
        sparse_idx = min(int(i / num_dense * num_sparse), num_sparse - 1)
        dense.append(sparse_labels[sparse_idx])
    return dense


# ========== Main evaluation ==========

def evaluate(args):
    test_annot = {e["qid"]: e for e in json.load(open(args.test_json))}
    eval_data = json.load(open(args.eval_json))
    preds = [p for p in eval_data["predictions"] if "error" not in p]

    # Group by video
    by_video = defaultdict(list)
    for p in preds:
        qid = p["qid"]
        if qid not in test_annot:
            continue
        entry = test_annot[qid]
        vid = entry["id"]
        query = entry["annos"][0]["query"]
        action = V2_TO_V1.get(query, query)
        windows = entry["annos"][0]["window"]
        duration = entry["duration"]
        by_video[vid].append({
            "action": action, "windows": windows,
            "duration": duration, "pred_text": p["pred_text"],
        })

    NUM_SPARSE = 32
    all_correct, all_total = 0, 0
    overlap_thresholds = [0.1, 0.25, 0.5]
    tp_all = {k: 0 for k in overlap_thresholds}
    fp_all = {k: 0 for k in overlap_thresholds}
    fn_all = {k: 0 for k in overlap_thresholds}
    edit_scores = []

    for vid in sorted(by_video.keys()):
        entries = by_video[vid]
        duration = entries[0]["duration"]

        # Get original video info
        for ext in [".mp4", ".avi"]:
            vpath = os.path.join(args.video_folder, f"{vid}{ext}")
            if os.path.exists(vpath):
                break
        orig_fps, orig_total = get_video_info(vpath)

        # Sampled timestamps (32 frames)
        sparse_ts = [round(i / max(NUM_SPARSE - 1, 1) * duration, 1) for i in range(NUM_SPARSE)]

        # Build GT: dense per-frame labels at original fps
        gt_dense = ["background"] * orig_total
        for entry in entries:
            if entry["action"] == "background":
                continue
            for w in entry["windows"]:
                s_frame = max(0, int(w[0] * orig_fps))
                e_frame = min(orig_total, int(w[1] * orig_fps) + 1)
                for fi in range(s_frame, e_frame):
                    gt_dense[fi] = entry["action"]

        # Build pred: sparse (32 frames) → dense (orig_total frames)
        sparse_pred = ["background"] * NUM_SPARSE
        for entry in entries:
            if args.mode == "binary":
                hits = parse_binary_hits(entry["pred_text"], NUM_SPARSE)
            else:
                hits = parse_timestamp_hits(entry["pred_text"], sparse_ts)
            for fi in hits:
                if fi < NUM_SPARSE and entry["action"] != "background":
                    sparse_pred[fi] = entry["action"]

        pred_dense = upsample_labels(sparse_pred, NUM_SPARSE, orig_total)

        # Frame accuracy
        correct = sum(1 for g, p in zip(gt_dense, pred_dense) if g == p)
        all_correct += correct
        all_total += orig_total

        # Segmental F1
        for ov in overlap_thresholds:
            tp, fp, fn = f_score(pred_dense, gt_dense, ov)
            tp_all[ov] += tp
            fp_all[ov] += fp
            fn_all[ov] += fn

        # Edit
        edit_scores.append(edit_score(pred_dense, gt_dense))

    # Aggregate
    acc = 100.0 * all_correct / max(all_total, 1)
    edit_avg = sum(edit_scores) / max(len(edit_scores), 1)
    f1_scores = {}
    for ov in overlap_thresholds:
        tp, fp, fn = tp_all[ov], fp_all[ov], fn_all[ov]
        precision = tp / max(tp + fp, 1e-9)
        recall = tp / max(tp + fn, 1e-9)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9) * 100
        f1_scores[ov] = f1

    result = {
        "experiment": args.name,
        "mode": args.mode,
        "videos": len(by_video),
        "total_frames": all_total,
        "sparse_frames": NUM_SPARSE,
        "Acc": round(acc, 1),
        "F1@10": round(f1_scores[0.1], 1),
        "F1@25": round(f1_scores[0.25], 1),
        "F1@50": round(f1_scores[0.5], 1),
        "Edit": round(edit_avg, 1),
    }

    print(f"\n{'='*60}")
    print(f"  {args.name}")
    print(f"  videos={result['videos']}, dense_frames={result['total_frames']}, sparse={NUM_SPARSE}")
    print(f"  Acc:   {result['Acc']:.1f}%")
    print(f"  F1@10: {result['F1@10']:.1f}%")
    print(f"  F1@25: {result['F1@25']:.1f}%")
    print(f"  F1@50: {result['F1@50']:.1f}%")
    print(f"  Edit:  {result['Edit']:.1f}")
    print(f"{'='*60}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved {args.output}")

    return result


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--video_folder", required=True)
    ap.add_argument("--mode", default="timestamp", choices=["timestamp", "binary"])
    ap.add_argument("--name", default="experiment")
    ap.add_argument("--output", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
