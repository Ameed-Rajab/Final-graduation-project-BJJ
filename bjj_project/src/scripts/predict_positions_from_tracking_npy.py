import os, re, json, glob, argparse
from collections import Counter
import numpy as np
import pandas as pd
import pickle
from bjjtrack.jiujitsu.utils import normalize

POSITIONS = [
    '5050_guard', 'back1', 'back2', 'closed_guard1', 'closed_guard2',
    'half_guard1', 'half_guard2', 'mount1', 'mount2', 'open_guard1',
    'open_guard2', 'side_control1', 'side_control2', 'standing',
    'takedown1', 'takedown2', 'turtle1', 'turtle2'
]

# Which families are role-suffixed:
ROLE_FAMILIES = {"open_guard", "closed_guard", "side_control", "mount", "half_guard", "takedown", "turtle", "back"}

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def natural_key(path: str):
    m = re.search(r"frame_(\d+)\.npy$", os.path.basename(path))
    return int(m.group(1)) if m else 10**18

def load_frame_npy(path: str):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return list(arr.tolist())
    return list(arr)

def bbox_center_x(b):
    b = np.asarray(b).astype(float).flatten()
    if b.size >= 4:
        return 0.5 * (b[0] + b[2])
    return 0.0

def bbox_center_y(b):
    b = np.asarray(b).astype(float).flatten()
    if b.size >= 4:
        return 0.5 * (b[1] + b[3])
    return 0.0

def bbox_score(b):
    b = np.asarray(b).astype(float).flatten()
    return float(b[4]) if b.size >= 5 else 0.0

def mean_kpt_y(kpts, idxs, min_score=0.20):
    k = np.asarray(kpts, dtype=np.float32)
    ys = []
    for i in idxs:
        if 0 <= i < k.shape[0] and k[i,2] >= min_score:
            ys.append(float(k[i,1]))
    return float(np.mean(ys)) if ys else None

def bottomness(person, use_pose=True, min_kpt_score=0.20):
    # larger => more "bottom" (lower in image)
    if use_pose and person.get("keypoints", None) is not None:
        k = person["keypoints"]
        # shoulders 5,6 ; hips 11,12
        y1 = mean_kpt_y(k, [11,12], min_kpt_score)
        y2 = mean_kpt_y(k, [5,6],  min_kpt_score)
        ys = [y for y in (y1,y2) if y is not None]
        if ys:
            return float(np.mean(ys))
    return bbox_center_y(person.get("bbox",[0,0,0,0,0]))

def pick_two_people(pose_results):
    valid = []
    for r in pose_results:
        k = r.get("keypoints", None)
        if k is None:
            continue
        k = np.asarray(k)
        if k.shape[0] != 17:
            continue
        sc = bbox_score(r.get("bbox",[0,0,0,0,0]))
        valid.append((sc, r))
    valid.sort(key=lambda x: x[0], reverse=True)
    if len(valid) < 2:
        return None
    return valid[0][1], valid[1][1]

def order_for_classifier(a, b, mode="trackid"):
    mode = mode.lower().strip()
    if mode == "trackid":
        ta = a.get("track_id", None)
        tb = b.get("track_id", None)
        if ta is not None and tb is not None and int(ta)!=int(tb):
            return (a,b) if int(ta)<int(tb) else (b,a)
        # fallback ltr
    ax = bbox_center_x(a.get("bbox",[0,0,0,0,0]))
    bx = bbox_center_x(b.get("bbox",[0,0,0,0,0]))
    return (a,b) if ax<=bx else (b,a)

def parse_family_suffix(label: str):
    m = re.match(r"^(.*?)([12])$", label)
    if not m:
        return label, None
    return m.group(1), m.group(2)

def enforce_bottom_is_2(label: str, bottom_is_person2: bool):
    """
    We enforce:
      - If bottom athlete corresponds to the SECOND person in the classifier input, label should end with '2'
      - If bottom corresponds to FIRST person, label should end with '1'
    This matches your definition: bottom => *2, top => *1.
    """
    fam, suf = parse_family_suffix(label)
    if suf is None:
        return label  # no suffix class
    # only apply to the role families
    basefam = fam  # e.g., open_guard
    if basefam not in ROLE_FAMILIES:
        return label

    want = "2" if bottom_is_person2 else "1"
    return f"{basefam}{want}"

def smooth_predictions(labels, probs, window=15):
    if window <= 1:
        return labels, probs
    n = len(labels)
    half = window // 2
    probs = np.asarray(probs)
    out_lab, out_prob = [], []
    for i in range(n):
        lo = max(0, i-half); hi = min(n, i+half+1)
        s = probs[lo:hi].sum(axis=0)
        idx = int(np.argmax(s))
        out_lab.append(POSITIONS[idx])
        out_prob.append(s/(s.sum()+1e-9))
    return out_lab, out_prob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-root", required=True)
    ap.add_argument("--classifier", default="checkpoints/jiujitsu/classifier.pickle")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smooth-window", type=int, default=15)

    # Keep classifier ordering stable (the order it was trained on)
    ap.add_argument("--classifier-order", default="trackid", choices=["trackid","ltr"])

    # Role enforcement
    ap.add_argument("--enforce-bottom-2", action="store_true",
                    help="Post-fix predicted labels so bottom athlete is always class *2.")
    ap.add_argument("--tb-use-pose", action="store_true")
    ap.add_argument("--tb-min-kpt-score", type=float, default=0.20)
    args = ap.parse_args()

    frames_dir = os.path.join(args.tracking_root, "predictions", "frames")
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(frames_dir)

    out_dir = args.out_dir or os.path.join(args.tracking_root, "position_predictions")
    ensure_dir(out_dir)

    with open(args.classifier, "rb") as f:
        clf = pickle.load(f)

    npy_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.npy")), key=natural_key)
    if not npy_files:
        raise FileNotFoundError("No frame_*.npy found")

    rows, probs_all, pred_all = [], [], []
    skipped = 0
    used = 0

    for path in npy_files:
        pose_results = load_frame_npy(path)
        picked = pick_two_people(pose_results)
        if picked is None:
            skipped += 1
            continue

        # 1) Determine top/bottom (independent of classifier order)
        pX, pY = picked
        yX = bottomness(pX, use_pose=args.tb_use_pose, min_kpt_score=args.tb_min_kpt_score)
        yY = bottomness(pY, use_pose=args.tb_use_pose, min_kpt_score=args.tb_min_kpt_score)
        bottom, top = (pX, pY) if yX >= yY else (pY, pX)

        # 2) Decide fixed order for classifier input (stable like training)
        p1, p2 = order_for_classifier(picked[0], picked[1], mode=args.classifier_order)

        # 3) Build features in that stable order
        k1 = np.asarray(p1["keypoints"], dtype=np.float32)
        k2 = np.asarray(p2["keypoints"], dtype=np.float32)
        n1, n2 = normalize((k1, k2))
        feat = np.concatenate([n1.flatten(), n2.flatten()]).astype(np.float32)

        prob = clf.predict_proba(feat.reshape(1, -1))[0]
        idx = int(np.argmax(prob))
        pred = POSITIONS[idx]

        # 4) Post-fix: enforce bottom=2 if enabled
        if args.enforce_bottom_2:
            bottom_is_person2 = (bottom is p2)
            pred = enforce_bottom_is_2(pred, bottom_is_person2)

        probs_all.append(prob)
        pred_all.append(pred)

        rows.append({
            "frame_file": os.path.basename(path),
            "frame_idx": used,
            "pred": POSITIONS[idx],          # raw classifier label
            "pred_fixed": pred,              # after suffix fix
            "pred_conf": float(np.max(prob)),
            "classifier_order": args.classifier_order,
            "enforce_bottom_2": bool(args.enforce_bottom_2),
        })
        used += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable frames")

    # Smooth on the FIXED labels by mapping through probs smoothing (base idx), then re-apply suffix fix per frame not possible.
    # So we keep smoothing only for raw classifier index, and keep pred_fixed as per-frame.
    raw_labels = df["pred"].tolist()
    probs_all = np.asarray(probs_all)
    smooth_raw, smooth_probs = smooth_predictions(raw_labels, probs_all, window=args.smooth_window)

    df["pred_smooth_raw"] = smooth_raw
    df["pred_smooth_raw_conf"] = [float(np.max(p)) for p in smooth_probs]

    csv_path = os.path.join(out_dir, "positions.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "tracking_root": args.tracking_root,
        "frames_total": len(npy_files),
        "frames_used": int(len(df)),
        "frames_skipped": int(skipped),
        "classifier_order": args.classifier_order,
        "enforce_bottom_2": bool(args.enforce_bottom_2),
        "smooth_window": int(args.smooth_window),
        "dist_pred_fixed": dict(Counter(df["pred_fixed"].tolist())),
        "csv": csv_path,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", csv_path)
    print("Used:", len(df), "Skipped:", skipped)

if __name__ == "__main__":
    main()