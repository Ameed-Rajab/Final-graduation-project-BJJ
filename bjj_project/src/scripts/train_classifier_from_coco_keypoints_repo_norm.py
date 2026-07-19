#!/usr/bin/env python3
"""
Train BJJ position classifier from COCO keypoints annotations using the SAME normalize()
function used by the repo evaluation code.

Key fixes vs previous version:
- Disable sklearn internal early-stopping validation (prevents leakage).
- Use ONLY external val split for evaluation.
- Apply swap augmentation ONLY to TRAIN split (optional but recommended).
- Keep output compatible with evaluation script: predict() + predict_proba().

Usage:
  python src/scripts/train_classifier_from_coco_keypoints_repo_norm.py \
    --train-annotations dataset/annotations/train_annotations.coco.json \
    --val-annotations dataset/annotations/val_annotations.coco.json \
    --output checkpoints/jiujitsu/classifier.pickle
"""

import os
import json
import pickle
import argparse
from collections import defaultdict, Counter

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


# Must match evaluation code
from bjjtrack.jiujitsu.utils import normalize


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def swap_label(lbl: str) -> str:
    """Swap labels ending with 1/2 (mount1<->mount2, etc.)."""
    if lbl.endswith("1"):
        return lbl[:-1] + "2"
    if lbl.endswith("2"):
        return lbl[:-1] + "1"
    return lbl


def reshape_kpts_coco(kpts_flat):
    """
    COCO person keypoints: 17*(x,y,v)=51.
    Return (17,3) float32 or None.
    """
    k = np.asarray(kpts_flat, dtype=np.float32)
    if k.size != 51:
        return None
    return k.reshape(17, 3)


def pick_two_persons(anns):
    """
    Select 2 persons from COCO annotations for ONE image.
    Strategy:
      - require keypoints (51) and bbox
      - choose top2 by bbox area
      - order left-to-right by bbox center-x to stabilize p1/p2
    Returns: (kpts1, kpts2) or None
    """
    valid = []
    for a in anns:
        if "keypoints" not in a or "bbox" not in a:
            continue
        k3 = reshape_kpts_coco(a["keypoints"])
        if k3 is None:
            continue

        # COCO bbox: [x,y,w,h]
        x, y, w, h = map(float, a["bbox"])
        area = w * h
        cx = x + w / 2.0
        valid.append((area, cx, k3))

    if len(valid) < 2:
        return None

    valid.sort(key=lambda t: t[0], reverse=True)
    top2 = valid[:2]
    top2.sort(key=lambda t: t[1])  # left-to-right

    return top2[0][2], top2[1][2]


def load_split(coco_json_path, max_samples=None, add_swap_aug=False):
    """
    Load features/labels from COCO json using repo normalize().
    Assumption: image-level position label is stored as category_id of annotations,
                and all anns in one image share the same category_id.
    """
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}

    by_img = defaultdict(list)
    for a in coco.get("annotations", []):
        by_img[a["image_id"]].append(a)

    X, y = [], []
    skipped = Counter()

    used = 0
    for img_id, anns in by_img.items():
        if max_samples is not None and used >= max_samples:
            break

        cat_ids = [a.get("category_id") for a in anns if a.get("category_id") is not None]
        if not cat_ids:
            skipped["no_category_id"] += 1
            continue

        # If multiple labels inside same image, skip (safer)
        if len(set(cat_ids)) != 1:
            skipped["mixed_category_ids"] += 1
            continue

        label = cat_id_to_name.get(cat_ids[0])
        if label is None:
            skipped["unknown_category_id"] += 1
            continue

        if label == "transition":
            skipped["transition"] += 1
            continue

        pair = pick_two_persons(anns)
        if pair is None:
            skipped["<2_persons_with_keypoints"] += 1
            continue

        k1, k2 = pair

        # EXACT same normalization as evaluation code
        p1, p2 = normalize((k1, k2))
        feat = np.concatenate([p1.flatten(), p2.flatten()], axis=0).astype(np.float32)

        X.append(feat)
        y.append(label)
        used += 1

        # Optional: swap augmentation ONLY for TRAIN to reduce 1/2 confusion
        if add_swap_aug:
            lbl2 = swap_label(label)
            p1s, p2s = normalize((k2, k1))
            feat2 = np.concatenate([p1s.flatten(), p2s.flatten()], axis=0).astype(np.float32)
            X.append(feat2)
            y.append(lbl2)
            used += 1

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=object)

    return X, y, skipped, cat_id_to_name


def make_model():
    """
    Pipeline with scaling + MLP.
    IMPORTANT: early_stopping=False to avoid internal split leakage.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=64,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=1200,
            random_state=42,
            early_stopping=False,   # critical fix
            verbose=True
        ))
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-annotations", required=True, type=str)
    ap.add_argument("--val-annotations", required=True, type=str)
    ap.add_argument("--output", default="checkpoints/jiujitsu/classifier.pickle", type=str)
    ap.add_argument("--max-samples", default=None, type=int)
    ap.add_argument("--swap-aug", action="store_true", help="Enable swap augmentation on TRAIN only")
    args = ap.parse_args()

    print(f"\nTRAIN annotations: {args.train_annotations}")
    Xtr, ytr, sk_tr, cats_tr = load_split(
        args.train_annotations,
        max_samples=args.max_samples,
        add_swap_aug=args.swap_aug
    )
    print(f"TRAIN usable samples: {len(Xtr)}")
    print(f"TRAIN skipped: {dict(sk_tr)}")
    print(f"TRAIN class dist (top 15): {Counter(ytr).most_common(15)}")

    print(f"\nVAL annotations: {args.val_annotations}")
    Xva, yva, sk_va, cats_va = load_split(
        args.val_annotations,
        max_samples=args.max_samples,
        add_swap_aug=False
    )
    print(f"VAL usable samples: {len(Xva)}")
    print(f"VAL skipped: {dict(sk_va)}")
    print(f"VAL class dist (top 15): {Counter(yva).most_common(15)}")

    if len(Xtr) == 0 or len(Xva) == 0:
        print("\nERROR: Not enough usable samples.")
        print("Common causes:")
        print("- images don't have 2 persons with keypoints")
        print("- category_id is not the position label")
        return

    # Train
    print("\n=== Training model (no internal early stopping) ===")
    model = make_model()
    model.fit(Xtr, ytr)

    # Evaluate on external VAL
    pred = model.predict(Xva)
    acc = accuracy_score(yva, pred)
    print(f"\n=== External VAL Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(yva, pred, zero_division=0))

    # Save
    ensure_dir(os.path.dirname(args.output))
    with open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved classifier to: {args.output}")

    # Quick compatibility check
    proba = model.predict_proba(Xva[:2])
    print(f"predict_proba OK. shape={proba.shape} (classes={len(model.classes_)})")


if __name__ == "__main__":
    main()
