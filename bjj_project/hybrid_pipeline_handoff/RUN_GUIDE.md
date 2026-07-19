# RUN GUIDE — Hybrid BJJ Pipeline

Run these three stages in order on one input video. Placeholders:

- `VIDEO`  = path to the input video (e.g. `data/videos/input.mp4`)
- `ROOT`   = an output directory you choose (e.g. `outputs/my_run`)
- `WEIGHTS`= `weights/bjj_yolo18_domain.pt` (in this folder)

Run everything from a directory where `src/` is importable, or first do:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

---

## ⚡ Already have the OLD pipeline working? Do the MINIMAL swap

If your app already produces tracking output (stage 1) and runs style analysis
(stage 3), you only need to replace **stage 2**. Skip to
[Stage 2 (HYBRID)](#stage-2--position-labels-hybrid--the-change) and keep the
rest of your app exactly as it is. Everything below stage 1 is provided just so
the handoff is complete and self-contained.

---

## Stage 1 — Tracking (DETR + IoU tracker + ViTPose)

Produces `ROOT/predictions/frames/frame_*.npy` (one file per frame: bounding
boxes + ViTPose keypoints + persistent track ids). Needs the DETR + ViTPose
checkpoints from your existing setup.

```bash
python src/scripts/run_tracking_detr_iou_ROBUST.py \
  --video-path VIDEO \
  --det-config   mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py \
  --det-checkpoint checkpoints/detection/deformable_detr_twostage_refine.pth \
  --pose-config  ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \
  --pose-checkpoint checkpoints/pose/vitpose.pth \
  --device cuda:0 \
  --out-root ROOT \
  --save-vis \
  --det-thr 0.10 --max-dets 12 \
  --tracker-iou 0.15 --tracker-max-lost 90 \
  --tracker-alpha-app 0.65 --tracker-app-gate 0.60 \
  --tracker-max-center-jump-frac 0.25 \
  --bbox-pad 0.15
```

(Paths for `--det-*` / `--pose-*` point into your existing mmdetection / ViTPose
install — this handoff does not bundle those large frameworks or checkpoints.)

---

## Stage 2 — Position labels (HYBRID) ← the change

Runs the **trained YOLO detector** on each frame and assigns a position label to
each tracked athlete. Writes `ROOT/position_predictions/positions.csv` — the
**same columns** the old classifier produced, so stage 3 is unchanged.

```bash
python src/scripts/predict_positions_from_yolo.py \
  --tracking-root ROOT \
  --video-path    VIDEO \
  --yolo-weights  weights/bjj_yolo18_domain.pt \
  --out-dir       ROOT/position_predictions \
  --enforce-bottom-2 --tb-use-pose --smooth-window 15
```

### Gotchas (only 3)
1. **`--video-path` is REQUIRED** — YOLO runs on the real frames, so pass the
   same video that was tracked. (The old classifier only needed the `.npy`.)
2. **`--out-dir ROOT/position_predictions`** keeps the CSV in the folder stage 3
   expects. Omit it and output goes to `position_predictions_yolo/` instead.
3. **Frame skipping:** if stage 1 skipped frames, add
   `--video-frame-offset N` with the same value so it seeks the right frame.
   No skipping → leave it out (default 0).

### No GPU?
Change `--device cuda:0` to `--device cpu` in stage 2 (same weights, just slower).
Handy extra flags: `--yolo-conf 0.25`, `--yolo-iou 0.5`, `--yolo-imgsz 640`,
`--match-iou-thr 0.2`.

### For reference: what stage 2 USED to be (old pipeline)
```bash
# predict_positions_from_tracking_ROLEFIX.py is included ONLY so you can see the
# old classifier stage it replaces. You do not need to run it.
python src/scripts/predict_positions_from_tracking_ROLEFIX.py \
  --tracking-root ROOT \
  --classifier checkpoints/jiujitsu/classifier.pickle \
  --classifier-order trackid --enforce-bottom-2 --tb-use-pose --smooth-window 15
```

---

## Stage 3 — Style analysis

Reads `positions.csv` and produces the per-player style report. Pure standard
library — nothing to install.

```bash
python3 src/scripts/analyze_player_styles.py \
  --position-dir ROOT/position_predictions \
  --fps 30 \
  --markdown
```

Outputs per-player position distribution, top/bottom control time, transition
intensity, advances/concessions, and a suggested game plan.

---

## End-to-end (copy/paste, one video)

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
VIDEO=data/videos/input.mp4
ROOT=outputs/my_run

# 1) tracking
python src/scripts/run_tracking_detr_iou_ROBUST.py --video-path "$VIDEO" \
  --det-config mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py \
  --det-checkpoint checkpoints/detection/deformable_detr_twostage_refine.pth \
  --pose-config ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \
  --pose-checkpoint checkpoints/pose/vitpose.pth \
  --device cuda:0 --out-root "$ROOT" --save-vis \
  --det-thr 0.10 --max-dets 12 --tracker-iou 0.15 --tracker-max-lost 90 \
  --tracker-alpha-app 0.65 --tracker-app-gate 0.60 --tracker-max-center-jump-frac 0.25 --bbox-pad 0.15

# 2) HYBRID position labels
python src/scripts/predict_positions_from_yolo.py --tracking-root "$ROOT" \
  --video-path "$VIDEO" --yolo-weights weights/bjj_yolo18_domain.pt \
  --out-dir "$ROOT/position_predictions" \
  --enforce-bottom-2 --tb-use-pose --smooth-window 15

# 3) style
python3 src/scripts/analyze_player_styles.py \
  --position-dir "$ROOT/position_predictions" --fps 30 --markdown
```
