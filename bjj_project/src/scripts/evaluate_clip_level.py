import os, re, glob, json, argparse
import numpy as np
import pandas as pd

def natural_key(path: str):
    name = os.path.basename(path)
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def majority_vote(labels):
    labels = [x for x in labels if pd.notna(x)]
    if not labels:
        return None
    vc = pd.Series(labels).value_counts()
    return vc.index[0]

def majority_vote_from_positions_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    # pick best column automatically
    if "pred_smooth" in df.columns:
        col = "pred_smooth"
    elif "pred" in df.columns:
        col = "pred"
    else:
        # fallback: first non-frame column
        col = [c for c in df.columns if c not in ("frame_idx", "frame", "path")][0]

    pred = majority_vote(df[col].tolist())
    votes = int((df[col] == pred).sum()) if pred is not None else 0
    total = int(len(df))
    vote_frac = (votes / total) if total > 0 else 0.0
    return pred, votes, total, vote_frac

def load_gt_from_coco(coco_json_path: str):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # category_id -> name
    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}

    # image_id -> file_name
    img_id_to_file = {im["id"]: im["file_name"] for im in coco.get("images", [])}

    # file_name -> list of category names (can be multiple annotations per image)
    file_to_labels = {}
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        fn = img_id_to_file.get(img_id)
        cn = cat_id_to_name.get(cat_id)
        if fn is None or cn is None:
            continue
        file_to_labels.setdefault(fn, []).append(cn)

    # reduce to one GT label per file_name (majority if multiple)
    file_to_gt = {fn: majority_vote(lbls) for fn, lbls in file_to_labels.items()}
    return file_to_gt

def load_gt_from_csv(gt_csv_path: str):
    df = pd.read_csv(gt_csv_path)

    # try common formats:
    # 1) file_name,label
    # 2) frame_idx,label
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    if "file_name" in cols and ("label" in cols or "gt" in cols):
        ycol = "label" if "label" in cols else "gt"
        return ("file_name", dict(zip(df["file_name"], df[ycol])))

    if ("frame_idx" in cols or "frame" in cols) and ("label" in cols or "gt" in cols):
        xcol = "frame_idx" if "frame_idx" in cols else "frame"
        ycol = "label" if "label" in cols else "gt"
        return (xcol, dict(zip(df[xcol].astype(int), df[ycol])))

    raise ValueError(f"Unsupported GT CSV columns: {df.columns.tolist()}")

def confusion_matrix(y_true, y_pred, labels):
    idx = {l:i for i,l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm

def per_class_metrics(cm, labels):
    # precision/recall/f1 per class
    eps = 1e-12
    out = []
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        sup  = cm[i, :].sum()
        out.append((lab, prec, rec, f1, sup))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-base", required=True,
                    help="Folder containing clip_*/ tracking roots (each has position_predictions/positions.csv)")
    ap.add_argument("--clip-len", type=int, default=15)
    ap.add_argument("--frames-dir", default="dataset/images/val",
                    help="Frames folder used to build the clips (sorted order must match clip creation)")
    ap.add_argument("--coco-json", default=None, help="COCO GT json (recommended if you have it)")
    ap.add_argument("--gt-csv", default=None, help="Alternative GT csv (file_name,label OR frame_idx,label)")
    ap.add_argument("--out-dir", default="outputs/clip_eval")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- clip predictions ----
    clip_dirs = sorted(glob.glob(os.path.join(args.track_base, "clip_*")), key=natural_key)

    pred_rows = []
    for d in clip_dirs:
        pos_csv = os.path.join(d, "position_predictions", "positions.csv")
        if not os.path.isfile(pos_csv):
            continue
        pred, votes, total, vote_frac = majority_vote_from_positions_csv(pos_csv)
        # clip index from name: clip_00016...
        m = re.search(r"clip_(\d+)", os.path.basename(d))
        clip_idx = int(m.group(1)) if m else None
        pred_rows.append({
            "clip_dir": d,
            "clip": os.path.basename(d),
            "clip_idx": clip_idx,
            "pred": pred,
            "votes": votes,
            "frames_in_csv": total,
            "vote_frac": vote_frac
        })

    pred_df = pd.DataFrame(pred_rows).dropna(subset=["clip_idx", "pred"]).sort_values("clip_idx")

    if len(pred_df) == 0:
        raise SystemExit("No clip predictions found. Expected: <clip>/position_predictions/positions.csv")

    # ---- build GT mapping per frame ----
    if args.coco_json:
        file_to_gt = load_gt_from_coco(args.coco_json)
        # frame order from frames-dir (must match how you made clips)
        frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")), key=natural_key)
        frame_files = [os.path.basename(p) for p in frame_paths]
        gt_per_frame = [file_to_gt.get(fn) for fn in frame_files]
    elif args.gt_csv:
        mode, mapping = load_gt_from_csv(args.gt_csv)
        if mode == "file_name":
            frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")), key=natural_key)
            frame_files = [os.path.basename(p) for p in frame_paths]
            gt_per_frame = [mapping.get(fn) for fn in frame_files]
        else:
            # frame_idx mapping
            max_i = max(mapping.keys())
            gt_per_frame = [mapping.get(i) for i in range(max_i + 1)]
    else:
        raise SystemExit("Provide GT via --coco-json OR --gt-csv")

    # ---- GT per clip (majority over its 15 frames) ----
    gt_clip = {}
    for clip_idx in pred_df["clip_idx"].unique():
        start = int(clip_idx) * args.clip_len
        end = start + args.clip_len
        labels = gt_per_frame[start:end]
        gt = majority_vote(labels)
        gt_clip[int(clip_idx)] = gt

    pred_df["gt"] = pred_df["clip_idx"].map(gt_clip)

    # keep only clips with GT
    eval_df = pred_df.dropna(subset=["gt"]).copy()
    eval_df.to_csv(os.path.join(args.out_dir, "per_clip.csv"), index=False)

    # ---- metrics ----
    y_true = eval_df["gt"].tolist()
    y_pred = eval_df["pred"].tolist()

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels)
    cm_df = pd.DataFrame(cm, index=[f"GT:{l}" for l in labels], columns=[f"P:{l}" for l in labels])
    cm_df.to_csv(os.path.join(args.out_dir, "confusion_matrix.csv"))

    acc = (np.array(y_true) == np.array(y_pred)).mean() if len(y_true) else 0.0

    per_cls = per_class_metrics(cm, labels)
    macro_f1 = float(np.mean([x[3] for x in per_cls])) if per_cls else 0.0
    weighted_f1 = float(np.average([x[3] for x in per_cls], weights=[x[4] for x in per_cls])) if per_cls else 0.0

    report_lines = []
    report_lines.append(f"Clips evaluated: {len(y_true)}")
    report_lines.append(f"Accuracy: {acc:.4f}")
    report_lines.append(f"Macro F1: {macro_f1:.4f}")
    report_lines.append(f"Weighted F1: {weighted_f1:.4f}")
    report_lines.append("")
    report_lines.append("Per-class metrics:")
    report_lines.append("label, precision, recall, f1, support")
    for lab, p, r, f1, sup in per_cls:
        report_lines.append(f"{lab}, {p:.4f}, {r:.4f}, {f1:.4f}, {int(sup)}")

    report_path = os.path.join(args.out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Saved:")
    print(" -", os.path.join(args.out_dir, "per_clip.csv"))
    print(" -", os.path.join(args.out_dir, "confusion_matrix.csv"))
    print(" -", report_path)
    print("\n".join(report_lines[:8]))

if __name__ == "__main__":
    main()
