#!/usr/bin/env python3
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

from bjjtrack.jiujitsu.utils import normalize


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def swap_label(lbl: str) -> str:
    if lbl.endswith("1"):
        return lbl[:-1] + "2"
    if lbl.endswith("2"):
        return lbl[:-1] + "1"
    return lbl


def reshape_kpts_coco(kpts_flat):
    k = np.asarray(kpts_flat, dtype=np.float32)
    if k.size != 51:
        return None
    return k.reshape(17, 3)


def pick_two_persons(anns):
    valid = []
    for a in anns:
        if "keypoints" not in a or "bbox" not in a:
            continue
        k3 = reshape_kpts_coco(a["keypoints"])
        if k3 is None:
            continue
        x, y, w, h = map(float, a["bbox"])  # [x,y,w,h]
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
        p1, p2 = normalize((k1, k2))
        feat = np.concatenate([p1.flatten(), p2.flatten()], axis=0).astype(np.float32)

        X.append(feat)
        y.append(label)
        used += 1

        if add_swap_aug:
            lbl2 = swap_label(label)
            p1s, p2s = normalize((k2, k1))
            feat2 = np.concatenate([p1s.flatten(), p2s.flatten()], axis=0).astype(np.float32)
            X.append(feat2)
            y.append(lbl2)
            used += 1

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=object), skipped


def balance_oversample(X, y, target_per_class=None, max_target=None, seed=42):
    """
    Oversample each class to target_per_class.
    If target_per_class is None -> use max class count (capped by max_target if given).
    """
    rng = np.random.default_rng(seed)
    counts = Counter(y)
    if not counts:
        return X, y

    max_count = max(counts.values())
    target = target_per_class if target_per_class is not None else max_count
    if max_target is not None:
        target = min(target, int(max_target))

    idx_by_class = defaultdict(list)
    for i, lbl in enumerate(y):
        idx_by_class[lbl].append(i)

    new_idx = []
    for lbl, idxs in idx_by_class.items():
        idxs = np.asarray(idxs, dtype=int)
        if len(idxs) >= target:
            # downsample to target (optional behavior to reduce dominance)
            chosen = rng.choice(idxs, size=target, replace=False)
        else:
            # oversample with replacement
            extra = rng.choice(idxs, size=(target - len(idxs)), replace=True)
            chosen = np.concatenate([idxs, extra], axis=0)
        new_idx.append(chosen)

    new_idx = np.concatenate(new_idx, axis=0)
    rng.shuffle(new_idx)

    return X[new_idx], y[new_idx]


def make_model():
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
            early_stopping=False,   # avoid leakage
            verbose=True
        ))
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-annotations", required=True, type=str)
    ap.add_argument("--val-annotations", required=True, type=str)
    ap.add_argument("--output", default="checkpoints/jiujitsu/classifier.pickle", type=str)

    ap.add_argument("--swap-aug", action="store_true", help="Swap augmentation on TRAIN only")
    ap.add_argument("--max-samples", default=None, type=int)

    ap.add_argument("--balance", action="store_true", help="Oversample classes in TRAIN")
    ap.add_argument("--target-per-class", type=int, default=None,
                    help="If set, oversample/downsample each class to this count")
    ap.add_argument("--max-target", type=int, default=2500,
                    help="Cap target-per-class when auto-balancing (default 2500)")

    args = ap.parse_args()

    print(f"\nTRAIN: {args.train_annotations}")
    Xtr, ytr, sk_tr = load_split(args.train_annotations, max_samples=args.max_samples, add_swap_aug=args.swap_aug)
    print(f"TRAIN usable: {len(Xtr)} | skipped: {dict(sk_tr)}")
    print("TRAIN dist (top 15):", Counter(ytr).most_common(15))

    if args.balance:
        before = Counter(ytr)
        Xtr, ytr = balance_oversample(
            Xtr, ytr,
            target_per_class=args.target_per_class,
            max_target=args.max_target,
            seed=42
        )
        after = Counter(ytr)
        print("\nBalanced TRAIN:")
        print("  before (top 10):", before.most_common(10))
        print("  after  (top 10):", after.most_common(10))
        print(f"  new TRAIN size: {len(Xtr)}")

    print(f"\nVAL: {args.val_annotations}")
    Xva, yva, sk_va = load_split(args.val_annotations, max_samples=args.max_samples, add_swap_aug=False)
    print(f"VAL usable: {len(Xva)} | skipped: {dict(sk_va)}")
    print("VAL dist (top 15):", Counter(yva).most_common(15))

    if len(Xtr) == 0 or len(Xva) == 0:
        print("\nERROR: Not enough usable samples.")
        return

    model = make_model()
    print("\n=== Training ===")
    model.fit(Xtr, ytr)

    print("\n=== External VAL Evaluation ===")
    pred = model.predict(Xva)
    acc = accuracy_score(yva, pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(yva, pred, zero_division=0))

    ensure_dir(os.path.dirname(args.output))
    with open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved classifier to: {args.output}")
    print(f"predict_proba OK. shape={model.predict_proba(Xva[:2]).shape} (classes={len(model.classes_)})")


if __name__ == "__main__":
    main()
