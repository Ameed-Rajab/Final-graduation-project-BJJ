# BJJ Hybrid Pipeline — Full Handoff

Complete code for the **hybrid** BJJ position + style pipeline: raw sparring
video → tracked athletes → 18-class position labels (via a fine-tuned YOLO) →
explainable style report.

This is the **improved** pipeline. Compared to the old one, only the middle
stage changes: position labels come from a **fine-tuned, domain-adapted YOLOv8
detector** instead of a keypoint classifier. That single change fixes the
real-world failures (e.g. back control 5.7% → 96.5%, 50/50 → 100%) while keeping
in-distribution accuracy (98.3% → 99.0%).

## The pipeline (3 stages)

```
  video.mp4
     │
     ▼
 [1] TRACKING          run_tracking_detr_iou_ROBUST.py
     DETR person boxes + IoU tracker + ViTPose keypoints
     → <ROOT>/predictions/frames/frame_*.npy   (bbox + keypoints + track_id per frame)
     │
     ▼
 [2] POSITION LABELS   predict_positions_from_yolo.py     ← THE HYBRID STAGE
     fine-tuned YOLO-18class on each frame, matched to tracked athletes
     → <ROOT>/position_predictions/positions.csv
     │
     ▼
 [3] STYLE ANALYSIS    analyze_player_styles.py
     per-player position %, top/bottom control, transitions, game plan
     → style report (markdown / json)
```

## What's in this folder

```
hybrid_pipeline_handoff/
├── README.md                     ← you are here (overview)
├── RUN_GUIDE.md                  ← exact commands to run all 3 stages
├── requirements.txt              ← what to install
├── weights/
│   └── bjj_yolo18_domain.pt      ← the TRAINED YOLO detector (22 MB, no retraining)
└── src/
    ├── bjjtrack/                 ← shared helper package (used by stages 1 & 3)
    └── scripts/
        ├── run_tracking_detr_iou_ROBUST.py           ← [1] tracking
        ├── predict_positions_from_yolo.py            ← [2] HYBRID position labels
        ├── predict_positions_from_tracking_ROLEFIX.py ← [2] OLD classifier (reference only)
        └── analyze_player_styles.py                  ← [3] style analysis
```

## What you must already have (from the old pipeline)

Stage 1 (tracking) uses the **DETR detector** and **ViTPose** through the
`mmdet` / `mmpose` frameworks. These are large and version-tied to your CUDA /
PyTorch build, so they are **not** bundled here — you already have them running
in the old pipeline. You need:

| Item | Path in the original project | Size |
|---|---|---|
| DETR config | `mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py` | tiny (needs the mmdetection config tree) |
| DETR checkpoint | `checkpoints/detection/deformable_detr_twostage_refine.pth` | 158 MB |
| ViTPose config | `ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py` | 5 KB |
| ViTPose checkpoint | `checkpoints/pose/vitpose.pth` | 376 MB |

If you keep your existing tracking stage as-is and only swap stage 2, you don't
even need to touch these — see the "minimal swap" note in `RUN_GUIDE.md`.

## Install

```bash
pip install -r requirements.txt   # really just: ultralytics + opencv/numpy/pandas
```

`mmdet` / `mmpose` / `torch` come from your existing environment.

## The one thing that's genuinely NEW

`weights/bjj_yolo18_domain.pt` is a fully **trained** model — you do **not**
retrain, you do **not** need the dataset. `predict_positions_from_yolo.py` just
loads it and runs inference. See `RUN_GUIDE.md` for the exact commands.
