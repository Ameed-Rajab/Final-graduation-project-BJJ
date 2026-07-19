#!/usr/bin/env python3
"""
predict_positions_from_yolo.py

Drop-in replacement for the CLASSIFICATION stage of
predict_positions_from_tracking_ROLEFIX.py.

Reads the per-frame tracking outputs (bboxes + keypoints + track_ids, as
produced by run_tracking_detr_iou_UPGRADED_FIXED.py) AND the source video.
For each frame, runs the fine-tuned YOLO-18class detector on the image
and uses its predicted class as the position label. Each tracked player is
assigned a label by matching to YOLO bboxes via IoU. The image-level label
is the majority vote of matched labels; a geometric top/bottom fix is then
applied so bottom == suffix "2".

Output is positions.csv with the SAME columns your analyze_player_styles.py
already consumes, so downstream code needs zero changes.

DOES NOT MODIFY: run_tracking_detr_iou_UPGRADED_FIXED.py,
predict_positions_from_tracking_ROLEFIX.py, analyze_player_styles.py,
or checkpoints/.

Example:
  python src/scripts/predict_positions_from_yolo.py \\
    --tracking-root outputs_2/detr_ioutrack_input_video \\
    --video-path data/videos/input.mp4 \\
    --yolo-weights runs/detect/runs/detect/bjj_18class_yolov8s/weights/best.pt \\
    --smooth-window 15 \\
    --enforce-bottom-2 \\
    --tb-use-pose
"""
import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


POSITIONS = [
    "5050_guard", "back1", "back2", "closed_guard1", "closed_guard2",
    "half_guard1", "half_guard2", "mount1", "mount2", "open_guard1",
    "open_guard2", "side_control1", "side_control2", "standing",
    "takedown1", "takedown2", "turtle1", "turtle2",
]
POS_TO_IDX = {p: i for i, p in enumerate(POSITIONS)}

ROLE_FAMILIES = {"open_guard", "closed_guard", "side_control", "mount",
                 "half_guard", "takedown", "turtle", "back"}


# ---------- helpers copied in spirit from ROLEFIX (keep output shape identical) ----------
def natural_key(path: str):
    m = re.search(r"frame_(\d+)\.npy$", os.path.basename(path))
    return int(m.group(1)) if m else 10**18


def load_frame_npy(path: str):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return list(arr.tolist())
    return list(arr)


def bbox_center_y(b):
    b = np.asarray(b).astype(float).flatten()
    return 0.5 * (b[1] + b[3]) if b.size >= 4 else 0.0


def bbox_center_x(b):
    b = np.asarray(b).astype(float).flatten()
    return 0.5 * (b[0] + b[2]) if b.size >= 4 else 0.0


def bbox_score(b):
    b = np.asarray(b).astype(float).flatten()
    return float(b[4]) if b.size >= 5 else 0.0


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(1.0, ax2 - ax1) * max(1.0, ay2 - ay1)
    ub = max(1.0, bx2 - bx1) * max(1.0, by2 - by1)
    return float(inter / (ua + ub - inter + 1e-9))


def mean_kpt_y(kpts, idxs, min_score=0.20):
    k = np.asarray(kpts, dtype=np.float32)
    ys = []
    for i in idxs:
        if 0 <= i < k.shape[0] and k[i, 2] >= min_score:
            ys.append(float(k[i, 1]))
    return float(np.mean(ys)) if ys else None


def bottomness(person, use_pose=True, min_kpt_score=0.20):
    if use_pose and person.get("keypoints") is not None:
        k = person["keypoints"]
        y1 = mean_kpt_y(k, [11, 12], min_kpt_score)
        y2 = mean_kpt_y(k, [5, 6], min_kpt_score)
        ys = [y for y in (y1, y2) if y is not None]
        if ys:
            return float(np.mean(ys))
    return bbox_center_y(person.get("bbox", [0, 0, 0, 0, 0]))


def pick_two_people(pose_results):
    """Sort by bbox score, take top-2 with keypoints. Same rule as ROLEFIX."""
    valid = []
    for r in pose_results:
        k = r.get("keypoints")
        if k is None:
            continue
        k = np.asarray(k)
        if k.shape[0] != 17:
            continue
        valid.append((bbox_score(r.get("bbox", [0, 0, 0, 0, 0])), r))
    valid.sort(key=lambda x: x[0], reverse=True)
    if len(valid) < 2:
        return None
    return valid[0][1], valid[1][1]


def parse_family_suffix(label: str):
    m = re.match(r"^(.*?)([12])$", label)
    if not m:
        return label, None
    return m.group(1), m.group(2)


def enforce_bottom_is_2(label: str, bottom_is_person2: bool):
    fam, suf = parse_family_suffix(label)
    if suf is None or fam not in ROLE_FAMILIES:
        return label
    want = "2" if bottom_is_person2 else "1"
    return f"{fam}{want}"


def order_for_classifier(a, b, mode="trackid"):
    if mode.lower() == "trackid":
        ta = a.get("track_id")
        tb = b.get("track_id")
        if ta is not None and tb is not None and int(ta) != int(tb):
            return (a, b) if int(ta) < int(tb) else (b, a)
    ax = bbox_center_x(a.get("bbox", [0, 0, 0, 0, 0]))
    bx = bbox_center_x(b.get("bbox", [0, 0, 0, 0, 0]))
    return (a, b) if ax <= bx else (b, a)


def majority_vote_labels(labels, window=15):
    """Majority-vote smoothing over a symmetric window. Returns
    (smoothed_labels, agreement_fractions)."""
    if window <= 1:
        return list(labels), [1.0] * len(labels)
    n = len(labels)
    half = window // 2
    out_l, out_a = [], []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window_vals = labels[lo:hi]
        c = Counter(window_vals)
        best, best_n = c.most_common(1)[0]
        out_l.append(best)
        out_a.append(best_n / max(1, len(window_vals)))
    return out_l, out_a


def yolo_infer_frame(model, frame, imgsz, conf, iou, device):
    """Run YOLO once; return (N,6) [x1,y1,x2,y2,conf,cls_id]."""
    res = model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz,
                        device=device, verbose=False)
    if not res:
        return np.zeros((0, 6), dtype=np.float32)
    r0 = res[0]
    if r0.boxes is None or len(r0.boxes.xyxy) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
    scr = r0.boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
    cls = r0.boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1)
    return np.concatenate([xyxy, scr, cls], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-root", required=True,
                    help="Directory produced by run_tracking_*.py; must contain "
                         "predictions/frames/frame_*.npy")
    ap.add_argument("--video-path", required=True,
                    help="Same video file that produced the tracking outputs.")
    ap.add_argument("--yolo-weights", required=True)
    ap.add_argument("--out-dir", default=None)

    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--yolo-iou", type=float, default=0.5)
    ap.add_argument("--yolo-imgsz", type=int, default=640)
    ap.add_argument("--yolo-max-dets", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--match-iou-thr", type=float, default=0.2,
                    help="Min IoU to match a YOLO box to a tracker person.")
    ap.add_argument("--video-frame-offset", type=int, default=0,
                    help="Add this to every npy filename's numeric index to get the "
                         "true video frame index. Set to the same value you passed "
                         "as --skip-frames to the tracker so we seek the right "
                         "position in the source video.")

    ap.add_argument("--smooth-window", type=int, default=15)
    ap.add_argument("--classifier-order", default="trackid",
                    choices=["trackid", "ltr"],
                    help="Kept for output-schema compatibility with the existing "
                         "predictor. Only affects the ordering annotation column.")

    ap.add_argument("--enforce-bottom-2", action="store_true")
    ap.add_argument("--tb-use-pose", action="store_true")
    ap.add_argument("--tb-min-kpt-score", type=float, default=0.20)

    args = ap.parse_args()

    frames_dir = os.path.join(args.tracking_root, "predictions", "frames")
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(frames_dir)
    out_dir = args.out_dir or os.path.join(args.tracking_root, "position_predictions_yolo")
    os.makedirs(out_dir, exist_ok=True)

    from ultralytics import YOLO
    print(f"Loading YOLO: {args.yolo_weights}")
    model = YOLO(args.yolo_weights)
    class_names = model.names  # dict {int: name}

    # Map YOLO class ids into our canonical POSITIONS ordering (they should already
    # match because we trained with these labels — but be defensive).
    def yolo_to_pos(cid: int):
        name = class_names.get(int(cid), None)
        if name is None:
            return None
        return name if name in POS_TO_IDX else None

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video_path}")
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    npy_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.npy")),
                       key=natural_key)
    if not npy_files:
        raise FileNotFoundError("No frame_*.npy in tracking dir.")

    # We assume the tracker saved one .npy PER video frame in order — that's
    # what run_tracking_detr_iou_UPGRADED_FIXED.py does (`frame_{saved:06d}.npy`
    # with saved==frame_idx when no skipping was used). If your run used a
    # non-1 stride, pass the same stride via --video-frame-stride.
    ap.add_argument  # (keep flake happy; no-op)

    rows = []
    per_frame_labels = []
    per_frame_confs = []
    skipped_no_people = 0
    skipped_no_yolo = 0
    used = 0

    # Sequential-read strategy: pre-index npy paths by their video_frame_idx and
    # walk the video ONE frame at a time (much faster than per-frame seek on 4K
    # H.264). We accumulate frames until we hit the next expected index.
    npy_by_idx = {}
    for path in npy_files:
        m = re.search(r"frame_(\d+)\.npy$", os.path.basename(path))
        if m:
            # Filename number is the tracker's SAVED counter. Real video frame
            # is offset by --video-frame-offset (which should match tracker's
            # --skip-frames).
            saved_idx = int(m.group(1))
            npy_by_idx[saved_idx + args.video_frame_offset] = path
    target_indices = sorted(npy_by_idx.keys())

    seq_iter = iter(target_indices)
    next_target = next(seq_iter, None)
    cur_video_idx = -1

    while next_target is not None:
        # Advance the video reader one frame at a time.
        ok, frame = cap.read()
        if not ok:
            break
        cur_video_idx += 1
        if cur_video_idx < next_target:
            continue

        path = npy_by_idx[next_target]
        video_frame_idx = next_target
        next_target = next(seq_iter, None)

        pose_results = load_frame_npy(path)
        picked = pick_two_people(pose_results)
        if picked is None:
            skipped_no_people += 1
            continue

        # 1) Detect on the frame
        yolo_dets = yolo_infer_frame(
            model, frame, args.yolo_imgsz, args.yolo_conf, args.yolo_iou, args.device,
        )
        if yolo_dets.shape[0] == 0:
            skipped_no_yolo += 1
            continue
        if yolo_dets.shape[0] > args.yolo_max_dets:
            top = np.argsort(-yolo_dets[:, 4])[: args.yolo_max_dets]
            yolo_dets = yolo_dets[top]

        # 2) Match each tracker person to the best-IoU YOLO det above threshold.
        p1_track, p2_track = picked
        def best_yolo_for(person):
            pb = np.asarray(person.get("bbox", [0, 0, 0, 0, 0]), dtype=float).flatten()
            if pb.size < 4:
                return None
            best_i, best_iou = -1, 0.0
            for i in range(yolo_dets.shape[0]):
                score = iou_xyxy(pb, yolo_dets[i])
                if score > best_iou:
                    best_iou = score
                    best_i = i
            if best_i < 0 or best_iou < args.match_iou_thr:
                return None
            return int(yolo_dets[best_i, 5]), float(yolo_dets[best_i, 4]), best_iou

        m1 = best_yolo_for(p1_track)
        m2 = best_yolo_for(p2_track)

        # 3) Decide the image-level label: majority of matched labels, tie-break
        # by higher YOLO confidence. If neither matched, fall back to the top-1
        # YOLO detection overall.
        candidates = []
        if m1 is not None:
            candidates.append(m1)
        if m2 is not None:
            candidates.append(m2)
        if not candidates:
            # No tracker box matched — use top-1 YOLO detection as image label.
            top_i = int(np.argmax(yolo_dets[:, 4]))
            top_cls_id = int(yolo_dets[top_i, 5])
            top_conf = float(yolo_dets[top_i, 4])
            candidates = [(top_cls_id, top_conf, 0.0)]

        label_confs = {}
        label_hits = Counter()
        for cid, cf, _ in candidates:
            label_hits[cid] += 1
            label_confs[cid] = max(label_confs.get(cid, 0.0), cf)
        # Pick class with most hits; tie-break by max conf
        winning_cls_id = sorted(
            label_hits.items(),
            key=lambda kv: (-kv[1], -label_confs[kv[0]]),
        )[0][0]
        pred_raw = yolo_to_pos(winning_cls_id)
        pred_raw_conf = label_confs[winning_cls_id]
        if pred_raw is None:
            skipped_no_yolo += 1
            continue

        # 4) Bottom-2 enforcement using tracker keypoints
        pred_fixed = pred_raw
        if args.enforce_bottom_2:
            yX = bottomness(p1_track, use_pose=args.tb_use_pose,
                            min_kpt_score=args.tb_min_kpt_score)
            yY = bottomness(p2_track, use_pose=args.tb_use_pose,
                            min_kpt_score=args.tb_min_kpt_score)
            # order_for_classifier decides which we call p1 vs p2
            p1_ord, p2_ord = order_for_classifier(p1_track, p2_track,
                                                  mode=args.classifier_order)
            bottom = p1_track if yX >= yY else p2_track
            bottom_is_person2 = (bottom is p2_ord)
            pred_fixed = enforce_bottom_is_2(pred_raw, bottom_is_person2)

        per_frame_labels.append(pred_fixed)
        per_frame_confs.append(pred_raw_conf)

        rows.append({
            "frame_file": os.path.basename(path),
            "frame_idx": used,
            "video_frame_idx": video_frame_idx,
            "pred": pred_raw,
            "pred_fixed": pred_fixed,
            "pred_conf": pred_raw_conf,
            "classifier_order": args.classifier_order,
            "enforce_bottom_2": bool(args.enforce_bottom_2),
        })
        used += 1

    cap.release()

    if not rows:
        raise RuntimeError("No usable frames produced a prediction.")

    df = pd.DataFrame(rows)

    # 5) Smoothing on pred_fixed (majority vote) — matches the SEMANTIC intent
    # of pred_smooth in ROLEFIX. Confidence stored is the window-agreement fraction.
    smooth_lab, smooth_conf = majority_vote_labels(
        df["pred_fixed"].tolist(), window=args.smooth_window)
    df["pred_smooth"] = smooth_lab
    df["pred_smooth_conf"] = smooth_conf

    # Also emit pred_smooth_raw with the un-fixed labels for parity with ROLEFIX.
    smooth_raw_lab, smooth_raw_conf = majority_vote_labels(
        df["pred"].tolist(), window=args.smooth_window)
    df["pred_smooth_raw"] = smooth_raw_lab
    df["pred_smooth_raw_conf"] = smooth_raw_conf

    csv_path = os.path.join(out_dir, "positions.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "tracking_root": args.tracking_root,
        "video_path": args.video_path,
        "yolo_weights": args.yolo_weights,
        "frames_total_tracking": len(npy_files),
        "frames_used": int(len(df)),
        "skipped_no_people": int(skipped_no_people),
        "skipped_no_yolo": int(skipped_no_yolo),
        "classifier_order": args.classifier_order,
        "enforce_bottom_2": bool(args.enforce_bottom_2),
        "smooth_window": int(args.smooth_window),
        "match_iou_thr": float(args.match_iou_thr),
        "dist_pred_fixed": dict(Counter(df["pred_fixed"].tolist())),
        "dist_pred_smooth": dict(Counter(df["pred_smooth"].tolist())),
        "csv": csv_path,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Used: {len(df)}  |  skipped_no_people: {skipped_no_people}  "
          f"skipped_no_yolo: {skipped_no_yolo}")


if __name__ == "__main__":
    main()
