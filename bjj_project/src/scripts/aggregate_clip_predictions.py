#!/usr/bin/env python3
import os, re, glob, json, argparse
from collections import Counter
import pandas as pd

def natural_key(s: str):
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def majority_vote(series):
    series = [x for x in series if pd.notna(x)]
    if not series:
        return None, 0.0
    c = Counter(series)
    label, votes = c.most_common(1)[0]
    frac = votes / max(1, len(series))
    return label, frac

def pick_label_column(df: pd.DataFrame):
    # Prefer pred_fixed if exists
    for col in ["pred_fixed", "pred_smooth", "pred", "pred_smooth_raw"]:
        if col in df.columns:
            return col
    raise ValueError(f"No usable prediction column. Found: {df.columns.tolist()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-base", required=True, help="folder containing clip_* dirs")
    ap.add_argument("--out-csv", default="outputs/val_clip_level_predictions.csv")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    clip_dirs = sorted(glob.glob(os.path.join(args.track_base, "clip_*")), key=natural_key)

    rows = []
    for d in clip_dirs:
        pos_csv = os.path.join(d, "position_predictions", "positions.csv")
        if not os.path.isfile(pos_csv):
            continue

        df = pd.read_csv(pos_csv)
        col = pick_label_column(df)
        label, vote_frac = majority_vote(df[col].tolist())

        m = re.search(r"clip_(\d+)", os.path.basename(d))
        clip_idx = int(m.group(1)) if m else None

        rows.append({
            "clip_dir": d,
            "clip": os.path.basename(d),
            "clip_idx": clip_idx,
            "label": label,
            "vote_frac": vote_frac,
            "label_col": col,
            "frames": int(len(df)),
        })

    out_df = pd.DataFrame(rows).dropna(subset=["clip_idx", "label"]).sort_values("clip_idx")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    summary = {
        "track_base": args.track_base,
        "clips_found": len(clip_dirs),
        "clips_aggregated": int(len(out_df)),
        "out_csv": args.out_csv
    }

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("Saved:", args.out_csv)
    print("Clips aggregated:", len(out_df))

if __name__ == "__main__":
    main()
