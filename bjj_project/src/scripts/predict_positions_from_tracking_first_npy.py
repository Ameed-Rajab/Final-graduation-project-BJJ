#!/usr/bin/env python3
"""
predict_positions_from_tracking_npy.py

Reads pose-tracking outputs saved as:
  <tracking_root>/predictions/frames/frame_000000.npy ... (each is a list of dicts)

Each dict is expected to contain:
  - "keypoints": (17,3) array-like
  - "bbox": [x1,y1,x2,y2,score] or similar
  - "track_id": int (optional but recommended)

Then:
  - selects 2 athletes per frame
  - orders them consistently (track_id if present, else left-to-right bbox center)
  - normalizes using bjjtrack.jiujitsu.utils.normalize (same as repo evaluation)
  - predicts position using your trained classifier.pickle
  - saves CSV + JSON summaries

This does NOT compute accuracy (no ground truth). It produces predictions timeline.
"""

import os
import re
import json
import glob
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pickle

from bjjtrack.jiujitsu.utils import normalize

# Must match training label set order from your pipeline
POSITIONS = [
    '5050_guard', 'back1', 'back2', 'closed_guard1', 'closed_guard2',
    'half_guard1', 'half_guard2', 'mount1', 'mount2', 'open_guard1',
    'open_guard2', 'side_control1', 'side_control2', 'standing',
    'takedown1', 'takedown2', 'turtle1', 'turtle2'
]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def natural_key(path: str):
    # sort frame_000123.npy numerically
    m = re.search(r"frame_(\d+)\.npy$", os.path.basename(path))
    return int(m.group(1)) if m else 10**18


def load_frame_npy(path: str):
    arr = np.load(path, allow_pickle=True)
    # stored as np.array(list_of_dicts, dtype=object)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return list(arr.tolist())
    return list(arr)


def bbox_center_x(bbox):
    b = np.asarray(bbox).astype(float).flatten()
    if b.size >= 4:
        x1, y1, x2, y2 = b[:4]
        return 0.5 * (x1 + x2)
    return 0.0


def bbox_score(bbox):
    b = np.asarray(bbox).astype(float).flatten()
    return float(b[4]) if b.size >= 5 else 0.0


def pick_two_people(pose_results):
    """
    Choose two athletes from a frame.
    Strategy:
      - keep entries that have keypoints
      - rank by bbox score (descending)
      - take top 2
    """
    valid = []
    for r in pose_results:
        k = r.get("keypoints", None)
        if k is None:
            continue
        k = np.asarray(k)
        if k.shape[0] != 17:
            continue
        bb = r.get("bbox", None)
        sc = bbox_score(bb) if bb is not None else 0.0
        valid.append((sc, r))

    valid.sort(key=lambda x: x[0], reverse=True)
    if len(valid) < 2:
        return None

    return valid[0][1], valid[1][1]


def order_two(a, b):
    """
    Make ordering stable.
    Prefer track_id if both exist and are different.
    Else, order by bbox center x (left-to-right).
    """
    ta = a.get("track_id", None)
    tb = b.get("track_id", None)
    if ta is not None and tb is not None and ta != tb:
        return (a, b) if int(ta) < int(tb) else (b, a)

    ax = bbox_center_x(a.get("bbox", [0, 0, 0, 0, 0]))
    bx = bbox_center_x(b.get("bbox", [0, 0, 0, 0, 0]))
    return (a, b) if ax <= bx else (b, a)


def predict_order_invariant(clf, feat12, feat21):
    """
    Order-invariant prediction by averaging probability across both orders.
    """
    p12 = clf.predict_proba(feat12.reshape(1, -1))[0]
    p21 = clf.predict_proba(feat21.reshape(1, -1))[0]
    p = 0.5 * (p12 + p21)
    idx = int(np.argmax(p))
    return POSITIONS[idx], p, idx


def smooth_predictions(pred_labels, pred_probs, window=15):
    """
    Simple temporal smoothing:
    For each frame i, sum probabilities in a +-window/2 neighborhood.
    """
    if window <= 1:
        return pred_labels, pred_probs

    n = len(pred_labels)
    half = window // 2
    probs = np.asarray(pred_probs)  # (N, C)
    smoothed = []
    smoothed_probs = []

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        s = probs[lo:hi].sum(axis=0)
        idx = int(np.argmax(s))
        smoothed.append(POSITIONS[idx])
        smoothed_probs.append(s / (s.sum() + 1e-9))

    return smoothed, smoothed_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-root", required=True,
                    help="Output root from run_tracking_*.py (contains predictions/frames)")
    ap.add_argument("--classifier", default="checkpoints/jiujitsu/classifier.pickle",
                    help="Path to classifier.pickle")
    ap.add_argument("--out-dir", default=None,
                    help="Where to save CSV/JSON (default: <tracking-root>/position_predictions)")
    ap.add_argument("--order-invariant", action="store_true",
                    help="Average probs across (p1,p2) and (p2,p1) to reduce 1/2 flips")
    ap.add_argument("--smooth-window", type=int, default=15,
                    help="Temporal smoothing window (odd number recommended). Use 1 to disable.")
    args = ap.parse_args()

    frames_dir = os.path.join(args.tracking_root, "predictions", "frames")
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"frames dir not found: {frames_dir}")

    if args.out_dir is None:
        out_dir = os.path.join(args.tracking_root, "position_predictions")
    else:
        out_dir = args.out_dir
    ensure_dir(out_dir)

    # Load classifier
    with open(args.classifier, "rb") as f:
        clf = pickle.load(f)

    # Find frames
    npy_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.npy")), key=natural_key)
    if not npy_files:
        raise FileNotFoundError(f"No frame_*.npy found in {frames_dir}")

    rows = []
    probs_all = []
    pred_all = []

    skipped = 0
    for fi, path in enumerate(npy_files):
        pose_results = load_frame_npy(path)

        picked = pick_two_people(pose_results)
        if picked is None:
            skipped += 1
            continue

        pA, pB = order_two(picked[0], picked[1])

        k1 = np.asarray(pA["keypoints"], dtype=np.float32)
        k2 = np.asarray(pB["keypoints"], dtype=np.float32)

        # repo normalization
        n1, n2 = normalize((k1, k2))
        feat12 = np.concatenate([n1.flatten(), n2.flatten()]).astype(np.float32)

        if args.order_invariant:
            n1b, n2b = normalize((k2, k1))
            feat21 = np.concatenate([n1b.flatten(), n2b.flatten()]).astype(np.float32)
            pred, prob, idx = predict_order_invariant(clf, feat12, feat21)
        else:
            prob = clf.predict_proba(feat12.reshape(1, -1))[0]
            idx = int(np.argmax(prob))
            pred = POSITIONS[idx]

        probs_all.append(prob)
        pred_all.append(pred)

        rows.append({
            "frame_file": os.path.basename(path),
            "frame_idx": fi,
            "pred": pred,
            "pred_conf": float(np.max(prob)),
            "track_id_a": int(pA.get("track_id", -1)) if pA.get("track_id", None) is not None else -1,
            "track_id_b": int(pB.get("track_id", -1)) if pB.get("track_id", None) is not None else -1,
        })

    if not rows:
        raise RuntimeError("No usable frames (could not find 2 people with keypoints).")

    # Optional smoothing
    probs_all = np.asarray(probs_all)
    pred_smooth, probs_smooth = smooth_predictions(pred_all, probs_all, window=args.smooth_window)

    for i in range(len(rows)):
        rows[i]["pred_smooth"] = pred_smooth[i]
        rows[i]["pred_smooth_conf"] = float(np.max(probs_smooth[i]))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "positions.csv")
    df.to_csv(csv_path, index=False)

    # Simple summary JSON
    dist_raw = Counter(df["pred"].tolist())
    dist_smooth = Counter(df["pred_smooth"].tolist())
    summary = {
        "tracking_root": args.tracking_root,
        "frames_total": len(npy_files),
        "frames_used": len(df),
        "frames_skipped": skipped,
        "order_invariant": bool(args.order_invariant),
        "smooth_window": int(args.smooth_window),
        "class_dist_raw": dict(dist_raw),
        "class_dist_smooth": dict(dist_smooth),
        "csv": csv_path,
    }
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print("Saved:", csv_path)
    print("Saved:", json_path)
    print("Used frames:", len(df), "Skipped:", skipped)


if __name__ == "__main__":
    main()
