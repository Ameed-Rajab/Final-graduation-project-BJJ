#!/usr/bin/env python3
import os, re, glob, json, argparse
import numpy as np
import pandas as pd

def natural_key(path: str):
    name = os.path.basename(path)
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def majority_vote(vals):
    vals = [v for v in vals if pd.notna(v)]
    if not vals:
        return None
    return pd.Series(vals).value_counts().index[0]

def merge_label(label: str):
    if label is None or pd.isna(label):
        return None
    s = str(label).strip()

    # collapse suffix 1/2 variants
    s = re.sub(r"(open_guard)[12]$", r"\1", s)
    s = re.sub(r"(closed_guard)[12]$", r"\1", s)
    s = re.sub(r"(side_control)[12]$", r"\1", s)
    s = re.sub(r"(mount)[12]$", r"\1", s)
    s = re.sub(r"(half_guard)[12]$", r"\1", s)
    s = re.sub(r"(back)[12]$", r"\1", s)
    s = re.sub(r"(takedown)[12]$", r"\1", s)
    s = re.sub(r"(turtle)[12]$", r"\1", s)

    # keep others as-is (standing, 5050_guard, etc.)
    return s

def load_file_to_gt_from_coco(coco_json_path: str):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    img_id_to_file = {im["id"]: im["file_name"] for im in coco.get("images", [])}

    file_to_labels = {}
    for ann in coco.get("annotations", []):
        fn = img_id_to_file.get(ann.get("image_id"))
        cn = cat_id_to_name.get(ann.get("category_id"))
        if fn is None or cn is None:
            continue
        file_to_labels.setdefault(fn, []).append(cn)

    # one label per image
    return {fn: majority_vote(lbls) for fn, lbls in file_to_labels.items()}

def confusion_matrix(y_true, y_pred, labels):
    idx = {l:i for i,l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg-csv", required=True, help="aggregated clip predictions (clip,label)")
    ap.add_argument("--frames-dir", required=True, help="dataset/images/val")
    ap.add_argument("--coco-json", required=True, help="val_annotations.coco.json")
    ap.add_argument("--clip-len", type=int, default=15)
    ap.add_argument("--out-dir", default="outputs/clip_eval_merged")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    agg = pd.read_csv(args.agg_csv)
    if "clip" not in agg.columns or "label" not in agg.columns:
        raise SystemExit(f"agg csv must have columns clip,label. Found: {agg.columns.tolist()}")

    # clip_idx from clip_00016...
    def clip_idx_from_name(s):
        m = re.search(r"clip_(\d+)", str(s))
        return int(m.group(1)) if m else None

    agg["clip_idx"] = agg["clip"].apply(clip_idx_from_name)
    agg = agg.dropna(subset=["clip_idx"]).copy()
    agg["clip_idx"] = agg["clip_idx"].astype(int)
    agg = agg.sort_values("clip_idx")

    # MERGE prediction labels
    agg["pred"] = agg["label"].apply(merge_label)

    # GT per frame
    file_to_gt = load_file_to_gt_from_coco(args.coco_json)
    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")), key=natural_key)
    frame_files = [os.path.basename(p) for p in frame_paths]
    gt_per_frame = [file_to_gt.get(fn) for fn in frame_files]

    # GT per clip: majority over 15 frames, THEN merge label
    gt_clip = {}
    for ci in agg["clip_idx"].unique():
        start = int(ci) * args.clip_len
        end = start + args.clip_len
        gt_raw = majority_vote(gt_per_frame[start:end])
        gt_clip[int(ci)] = merge_label(gt_raw)

    agg["gt"] = agg["clip_idx"].map(gt_clip)

    eval_df = agg.dropna(subset=["gt", "pred"]).copy()
    eval_df.to_csv(os.path.join(args.out_dir, "per_clip.csv"), index=False)

    y_true = eval_df["gt"].tolist()
    y_pred = eval_df["pred"].tolist()

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels)
    cm_df = pd.DataFrame(cm, index=[f"GT:{l}" for l in labels], columns=[f"P:{l}" for l in labels])
    cm_df.to_csv(os.path.join(args.out_dir, "confusion_matrix.csv"))

    acc = float((np.array(y_true) == np.array(y_pred)).mean()) if len(y_true) else 0.0

    eps = 1e-12
    per_cls = []
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        sup  = int(cm[i, :].sum())
        per_cls.append((lab, prec, rec, f1, sup))

    macro_f1 = float(np.mean([x[3] for x in per_cls])) if per_cls else 0.0
    weighted_f1 = float(np.average([x[3] for x in per_cls], weights=[x[4] for x in per_cls])) if per_cls else 0.0

    report_lines = []
    report_lines.append(f"Clips evaluated: {len(y_true)}")
    report_lines.append(f"Accuracy (MERGED): {acc:.4f}")
    report_lines.append(f"Macro F1 (MERGED): {macro_f1:.4f}")
    report_lines.append(f"Weighted F1 (MERGED): {weighted_f1:.4f}")
    report_lines.append("")
    report_lines.append("label, precision, recall, f1, support")
    for lab, p, r, f1, sup in per_cls:
        report_lines.append(f"{lab}, {p:.4f}, {r:.4f}, {f1:.4f}, {sup}")

    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines[:8]))
    print("Saved to:", args.out_dir)

if __name__ == "__main__":
    main()
