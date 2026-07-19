#!/usr/bin/env python3
"""
run_tracking_detr_iou.py

Option 2:
- Use DETR detector (MMDetection) per frame -> detections
- Assign stable track IDs with a simple IoU tracker
- Run MMPose top-down pose for each tracked bbox
- Save per-frame predictions and optional visualization

Requires:
- mmdet installed (same environment as mmtrack/mmpose often works)
- A DETR config (.py) that matches your trained checkpoint (.pth)

Example:
python src/scripts/run_tracking_detr_iou.py \
  --video-path data/videos/input.mp4 \
  --det-config path/to/your_detr_config.py \
  --det-checkpoint checkpoints/detection/deformable_detr_twostage_refine.pth \
  --det-class-idx 0 \
  --det-thr 0.2 \
  --max-dets 2 \
  --pose-config "$POSE_CONFIG" \
  --pose-checkpoint "$POSE_CHECKPOINT" \
  --device cuda:0 \
  --out-root outputs/detr_ioutrack \
  --save-vis
"""

import os
import sys
import time
import json
import logging
import warnings
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import List, Tuple, Optional

import cv2
import numpy as np

from mmdet.apis import init_detector, inference_detector

from mmpose.apis import init_pose_model, inference_top_down_pose_model
from mmpose.datasets import DatasetInfo

# Optional: repo visualizer
try:
    from bjjtrack.utils import vis_pose_tracking_result  # type: ignore
    HAS_REPO_VIS = True
except Exception:
    HAS_REPO_VIS = False

warnings.filterwarnings("ignore")


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def clip_bbox_xyxy(b: np.ndarray, w: int, h: int) -> np.ndarray:
    b = b.copy()
    b[0] = float(np.clip(b[0], 0, w - 1))
    b[1] = float(np.clip(b[1], 0, h - 1))
    b[2] = float(np.clip(b[2], 0, w - 1))
    b[3] = float(np.clip(b[3], 0, h - 1))
    return b


def pad_bbox_xyxy(b: np.ndarray, pad: float, w: int, h: int) -> np.ndarray:
    """
    Pad bbox by fraction of its size (pad=0.1 adds 10% each side).
    b: [x1,y1,x2,y2,score]
    """
    if pad <= 0:
        return clip_bbox_xyxy(b, w, h)
    b = b.copy()
    x1, y1, x2, y2 = b[:4].astype(float)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    dx = bw * pad
    dy = bh * pad
    b[0] = x1 - dx
    b[1] = y1 - dy
    b[2] = x2 + dx
    b[3] = y2 + dy
    return clip_bbox_xyxy(b, w, h)


def filter_huge_dets(dets: np.ndarray, frame_w: int, frame_h: int, max_area_frac: float) -> np.ndarray:
    """Drop detections whose box area exceeds max_area_frac of the frame.

    DETR sometimes emits a single large bbox covering both grappling athletes;
    filtering by area removes those merged-blob detections before tracking.
    """
    if dets.size == 0 or max_area_frac >= 1.0:
        return dets
    frame_area = float(frame_w) * float(frame_h)
    if frame_area <= 0:
        return dets
    bw = np.maximum(0.0, dets[:, 2] - dets[:, 0])
    bh = np.maximum(0.0, dets[:, 3] - dets[:, 1])
    area = bw * bh
    keep = area <= max_area_frac * frame_area
    return dets[keep]


def keypoint_bbox(kpts: Optional[np.ndarray], kpt_thr: float, min_kpts: int) -> Optional[Tuple[float, float, float, float]]:
    """Return (x1, y1, x2, y2) tight box around keypoints with score >= kpt_thr, or None."""
    if kpts is None:
        return None
    k = np.asarray(kpts, dtype=np.float32)
    if k.ndim != 2 or k.shape[1] < 3:
        return None
    mask = k[:, 2] >= float(kpt_thr)
    if int(mask.sum()) < int(min_kpts):
        return None
    xs = k[mask, 0]
    ys = k[mask, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def expand_bbox(b: Tuple[float, float, float, float], margin: float, frame_w: int, frame_h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    dx = bw * float(margin)
    dy = bh * float(margin)
    nx1 = float(np.clip(x1 - dx, 0, frame_w - 1))
    ny1 = float(np.clip(y1 - dy, 0, frame_h - 1))
    nx2 = float(np.clip(x2 + dx, 0, frame_w - 1))
    ny2 = float(np.clip(y2 + dy, 0, frame_h - 1))
    return nx1, ny1, nx2, ny2


def bbox_area_xyxy(b) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def fallback_draw(img, pose_results, kpt_thr=0.3):
    out = img.copy()
    for r in pose_results:
        bbox = np.asarray(r.get("bbox", None))
        if bbox is not None and bbox.size >= 4:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            score = float(bbox[4]) if bbox.size >= 5 else 0.0
            tid = int(r.get("track_id", -1))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"id={tid} s={score:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        kpts = r.get("keypoints", None)
        if kpts is None:
            continue
        kpts = np.asarray(kpts)
        if kpts.ndim != 2 or kpts.shape[1] < 2:
            continue
        for i in range(min(kpts.shape[0], 50)):
            x, y = float(kpts[i, 0]), float(kpts[i, 1])
            c = float(kpts[i, 2]) if kpts.shape[1] >= 3 else 1.0
            if c < kpt_thr:
                continue
            cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)
    return out


# ----------------------------
# ----------------------------
# Robust overlap-aware tracker (2 athletes)
# IoU + motion prediction + appearance (HSV hist) + global assignment
# ----------------------------
@dataclass
class Track2D:
    track_id: int
    bbox: np.ndarray          # [x1,y1,x2,y2,score]
    lost: int = 0
    vx: float = 0.0
    vy: float = 0.0
    hist: any = None          # appearance histogram

    def predict_bbox(self) -> np.ndarray:
        b = self.bbox.copy()
        cx = 0.5 * (float(b[0]) + float(b[2]))
        cy = 0.5 * (float(b[1]) + float(b[3]))
        bw = max(1.0, float(b[2]) - float(b[0]))
        bh = max(1.0, float(b[3]) - float(b[1]))
        cx2, cy2 = cx + self.vx, cy + self.vy
        b[0] = cx2 - bw / 2
        b[1] = cy2 - bh / 2
        b[2] = cx2 + bw / 2
        b[3] = cy2 + bh / 2
        return b


def _clip_bbox_xyxy(b: np.ndarray, w: int, h: int) -> np.ndarray:
    bb = np.asarray(b, dtype=np.float32).copy()
    bb[0] = np.clip(bb[0], 0, w - 1)
    bb[1] = np.clip(bb[1], 0, h - 1)
    bb[2] = np.clip(bb[2], 0, w - 1)
    bb[3] = np.clip(bb[3], 0, h - 1)
    if bb[2] <= bb[0]:
        bb[2] = min(w - 1, bb[0] + 1)
    if bb[3] <= bb[1]:
        bb[3] = min(h - 1, bb[1] + 1)
    return bb


def _hsv_hist(frame: np.ndarray, bbox5: np.ndarray, bins: int = 16):
    h, w = frame.shape[:2]
    bb = _clip_bbox_xyxy(bbox5[:4], w, h)
    x1, y1, x2, y2 = map(int, bb[:4])
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist


def _hist_dist(h1, h2) -> float:
    if h1 is None or h2 is None:
        return 1.0
    d = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    return float(np.clip(d, 0.0, 1.0))


class RobustOverlapTracker:
    """Two-person tracker tuned for heavy overlap.

    Matching uses:
      - IoU between predicted bbox and detections
      - motion (constant velocity on center)
      - appearance (HSV histogram distance)
      - global assignment for the first 2 tracks (try all pairings)

    This greatly reduces ID swaps / wrong association when athletes overlap.
    """

    def __init__(
        self,
        iou_thr: float = 0.45,
        max_lost: int = 60,
        max_tracks: int = 2,
        alpha_app: float = 0.65,
        app_gate: float = 0.60,
        max_center_jump_frac: float = 0.25,
    ):
        self.iou_thr = float(iou_thr)
        self.max_lost = int(max_lost)
        self.max_tracks = int(max_tracks)
        self.alpha_app = float(alpha_app)
        self.app_gate = float(app_gate)
        self.max_center_jump_frac = float(max_center_jump_frac)

        self.tracks: List[Track2D] = []
        self._next_id = 1
        self.W: Optional[int] = None
        self.H: Optional[int] = None

    def set_frame_size(self, w: int, h: int):
        self.W, self.H = int(w), int(h)

    def _center_jump_ok(self, track: Track2D, det5: np.ndarray) -> bool:
        if self.W is None or self.H is None:
            return True
        tcx = 0.5 * (float(track.bbox[0]) + float(track.bbox[2]))
        tcy = 0.5 * (float(track.bbox[1]) + float(track.bbox[3]))
        dcx = 0.5 * (float(det5[0]) + float(det5[2]))
        dcy = 0.5 * (float(det5[1]) + float(det5[3]))
        diag = (self.W * self.W + self.H * self.H) ** 0.5
        dist = ((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5
        return dist <= self.max_center_jump_frac * diag

    def _match_cost(self, frame: np.ndarray, track: Track2D, det5: np.ndarray):
        pred = track.predict_bbox()
        iou = iou_xyxy(pred, det5)
        if iou < self.iou_thr:
            return None

        if not self._center_jump_ok(track, det5):
            return None

        det_hist = _hsv_hist(frame, det5)
        d_app = _hist_dist(track.hist, det_hist)

        # Appearance gate (only if we have a stored hist)
        if track.hist is not None and d_app > self.app_gate:
            return None

        cost_iou = 1.0 - float(iou)  # 0 is best
        cost_app = float(d_app)      # 0 is best

        cost = (1.0 - self.alpha_app) * cost_iou + self.alpha_app * cost_app
        return cost, det_hist

    def update(self, frame: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """detections: (N,5) [x1,y1,x2,y2,score]
        returns: (M,6) [track_id,x1,y1,x2,y2,score]
        """
        dets = np.asarray(detections, dtype=np.float32)

        if dets.size == 0:
            for t in self.tracks:
                t.lost += 1
            self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
            return np.zeros((0, 6), dtype=np.float32)

        # Init from top detections
        if len(self.tracks) == 0:
            for j in range(min(self.max_tracks, dets.shape[0])):
                b = dets[j].copy()
                t = Track2D(track_id=self._next_id, bbox=b, lost=0)
                t.hist = _hsv_hist(frame, b)
                self._next_id += 1
                self.tracks.append(t)
            return np.asarray([[t.track_id, *t.bbox[:5]] for t in self.tracks], dtype=np.float32)

        M = len(self.tracks)
        N = dets.shape[0]

        # Compute candidate costs (None => not matchable)
        costs = [[None] * N for _ in range(M)]
        hist_cache = [[None] * N for _ in range(M)]
        for i, t in enumerate(self.tracks):
            for j in range(N):
                res = self._match_cost(frame, t, dets[j])
                if res is None:
                    continue
                c, det_hist = res
                costs[i][j] = c
                hist_cache[i][j] = det_hist

        assign = [-1] * M
        used = set()

        # Global assignment: try all pairings for first 2 tracks (best for 2 athletes)
        if M >= 2:
            best = None  # (sum_cost, j0, j1)
            for j0 in range(N):
                c0 = costs[0][j0]
                if c0 is None:
                    continue
                for j1 in range(N):
                    if j1 == j0:
                        continue
                    c1 = costs[1][j1]
                    if c1 is None:
                        continue
                    s = float(c0) + float(c1)
                    if best is None or s < best[0]:
                        best = (s, j0, j1)
            if best is not None:
                _, j0, j1 = best
                assign[0], assign[1] = j0, j1
                used.add(j0)
                used.add(j1)

        # If any remaining track unmatched, do best-match (helps when only 1 detection exists)
        for i in range(M):
            if assign[i] != -1:
                continue
            bestj = -1
            bestc = None
            for j in range(N):
                if j in used:
                    continue
                c = costs[i][j]
                if c is None:
                    continue
                if bestc is None or c < bestc:
                    bestc = c
                    bestj = j
            if bestj != -1:
                assign[i] = bestj
                used.add(bestj)

        # Update tracks
        for i, t in enumerate(self.tracks):
            j = assign[i]
            if j == -1:
                t.lost += 1
                continue

            det = dets[j].copy()

            # velocity update (EMA on center delta)
            tcx = 0.5 * (float(t.bbox[0]) + float(t.bbox[2]))
            tcy = 0.5 * (float(t.bbox[1]) + float(t.bbox[3]))
            dcx = 0.5 * (float(det[0]) + float(det[2]))
            dcy = 0.5 * (float(det[1]) + float(det[3]))
            dvx, dvy = (dcx - tcx), (dcy - tcy)
            t.vx = 0.8 * t.vx + 0.2 * dvx
            t.vy = 0.8 * t.vy + 0.2 * dvy

            t.bbox = det
            t.hist = hist_cache[i][j] if hist_cache[i][j] is not None else _hsv_hist(frame, det)
            t.lost = 0

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]

        # Spawn new tracks from unused detections
        for j in range(N):
            if j in used:
                continue
            if len(self.tracks) >= self.max_tracks:
                break
            b = dets[j].copy()
            t = Track2D(track_id=self._next_id, bbox=b, lost=0)
            t.hist = _hsv_hist(frame, b)
            self._next_id += 1
            self.tracks.append(t)

        out = []
        for t in self.tracks:
            b = t.bbox
            out.append([t.track_id, b[0], b[1], b[2], b[3], b[4]])
        return np.asarray(out, dtype=np.float32)

# Detection extraction
# ----------------------------
def extract_class_dets(det_result, class_idx: int) -> np.ndarray:
    """
    MMDetection inference_detector output varies by version/model:
      - list[np.ndarray] per class: det_result[class_idx] -> (N,5)
      - tuple(list, segm) etc: take det_result[0]
    Returns (N,5): [x1,y1,x2,y2,score]
    """
    res = det_result
    if isinstance(res, tuple):
        res = res[0]
    if isinstance(res, list):
        if len(res) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        idx = int(np.clip(class_idx, 0, len(res) - 1))
        arr = res[idx]
        if arr is None:
            return np.zeros((0, 5), dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 5:
            return arr[:, :5].astype(np.float32)
        return np.zeros((0, 5), dtype=np.float32)

    # ndarray case (rare)
    arr = np.asarray(res, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] >= 5:
        return arr[:, :5].astype(np.float32)
    return np.zeros((0, 5), dtype=np.float32)


# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = ArgumentParser()

    p.add_argument("--video-path", type=str, required=True)

    p.add_argument("--det-config", type=str, required=True, help="MMDetection DETR config (.py)")
    p.add_argument("--det-checkpoint", type=str, required=True, help="DETR checkpoint (.pth)")

    p.add_argument("--det-class-idx", type=int, default=0, help="Class index to use (0 is person for COCO)")
    p.add_argument("--det-thr", type=float, default=0.2, help="Detection score threshold")
    p.add_argument("--max-dets", type=int, default=2, help="Keep at most K detections per frame (2 players)")

    p.add_argument("--tracker-iou", type=float, default=0.3, help="IoU threshold for matching")
    p.add_argument("--tracker-max-lost", type=int, default=30, help="Frames to keep track without match")
    p.add_argument("--tracker-alpha-app", type=float, default=0.65, help="Weight for appearance in matching cost (0..1)")
    p.add_argument("--tracker-app-gate", type=float, default=0.60, help="Reject match if appearance distance > gate")
    p.add_argument("--tracker-max-center-jump-frac", type=float, default=0.25, help="Reject matches that jump too far (fraction of frame diagonal)")

    p.add_argument("--pose-config", type=str, required=True)
    p.add_argument("--pose-checkpoint", type=str, required=True)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-root", type=str, default="outputs/detr_ioutrack")

    p.add_argument("--bbox-pad", type=float, default=0.05, help="Pad bbox before pose (fraction)")
    p.add_argument("--kpt-thr", type=float, default=0.3)

    # B) Drop merged-blob detections that cover too much of the frame
    p.add_argument(
        "--max-bbox-area-frac",
        type=float,
        default=0.6,
        help="Drop detections whose bbox area exceeds this fraction of the frame area "
             "(filters DETR's 'merged-blob' boxes that cover both athletes). Set to 1.0 to disable.",
    )

    # C) Refine each athlete's bbox from their detected keypoints
    p.add_argument(
        "--refine-bbox-from-pose",
        action="store_true",
        help="After pose inference, tighten each person's bbox to a margin around their confident keypoints. "
             "Self-corrects loose detector boxes, including merged-blob cases.",
    )
    p.add_argument("--refine-min-kpts", type=int, default=4,
                   help="Minimum number of confident keypoints required to refine a bbox.")
    p.add_argument("--refine-kpt-thr", type=float, default=0.3,
                   help="Keypoint confidence threshold used by --refine-bbox-from-pose.")
    p.add_argument("--refine-margin", type=float, default=0.10,
                   help="Margin (fraction) added around the keypoint-derived bbox before replacing the detection bbox.")
    p.add_argument("--refine-shrink-ratio", type=float, default=0.85,
                   help="Only replace the bbox if the pose-derived box is at most this fraction of the original area "
                        "(prevents replacing already-tight boxes with slightly looser ones).")

    p.add_argument("--skip-frames", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=-1)

    p.add_argument("--save-vis", action="store_true")
    p.add_argument("--vis-fps", type=float, default=None)

    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    out_root = args.out_root
    pred_dir = os.path.join(out_root, "predictions")
    frames_dir = os.path.join(pred_dir, "frames")
    vis_dir = os.path.join(out_root, "vis")

    ensure_dir(out_root)
    ensure_dir(pred_dir)
    ensure_dir(frames_dir)
    if args.save_vis:
        ensure_dir(vis_dir)

    logfile = os.path.join(out_root, "log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
    )
    logger = logging.getLogger("run_tracking_detr_iou")

    logger.info(f"Video: {args.video_path}")
    logger.info(f"DETR:  cfg={args.det_config} ckpt={args.det_checkpoint}")

    # Load models
    det_model = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=args.device)

    dataset = pose_model.cfg.data["test"].get("type", None)
    dataset_info_cfg = pose_model.cfg.data["test"].get("dataset_info", None)
    dataset_info = None
    if dataset_info_cfg is not None:
        try:
            dataset_info = DatasetInfo(dataset_info_cfg)
        except Exception as e:
            logger.warning(f"DatasetInfo init failed: {e}")
            dataset_info = None

    tracker = RobustOverlapTracker(iou_thr=args.tracker_iou, max_lost=args.tracker_max_lost, max_tracks=2, alpha_app=args.tracker_alpha_app, app_gate=args.tracker_app_gate, max_center_jump_frac=args.tracker_max_center_jump_frac)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Could not open video: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if in_fps <= 0:
        in_fps = 30.0
    out_fps = float(args.vis_fps) if args.vis_fps is not None else float(in_fps)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker.set_frame_size(w, h)
    size = (w, h)

    video_writer = None
    out_vid = None
    if args.save_vis:
        base = os.path.basename(args.video_path)
        out_vid = os.path.join(out_root, f"vis_{base}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(out_vid, fourcc, out_fps, size)

    frame_idx = 0
    saved = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx < args.skip_frames:
            frame_idx += 1
            continue

        if args.max_frames > 0 and saved >= args.max_frames:
            break

        # 1) DETR detection
        det_result = inference_detector(det_model, frame)
        dets = extract_class_dets(det_result, args.det_class_idx)  # (N,5)

        # 2) Score threshold + top-K (2 players)
        n_dropped_huge = 0
        if dets.size:
            dets = dets[dets[:, 4] >= float(args.det_thr)]
            n_before_filter = int(dets.shape[0])
            dets = filter_huge_dets(dets, w, h, args.max_bbox_area_frac)
            n_dropped_huge = n_before_filter - int(dets.shape[0])
            if dets.shape[0] > 1:
                order = np.argsort(-dets[:, 4])
                dets = dets[order]
            if args.max_dets > 0 and dets.shape[0] > args.max_dets:
                dets = dets[:args.max_dets]

            # clip/pad later (after tracking)

        # 3) Tracking (assign IDs)
        track_bboxes = tracker.update(frame, dets)  # (M,6) [id,x1,y1,x2,y2,score]

        # 4) Prepare pose bboxes: (M,5) and track_ids
        if track_bboxes.size == 0:
            bboxes_5 = np.zeros((0, 5), dtype=np.float32)
            track_ids = np.zeros((0,), dtype=np.int32)
        else:
            track_ids = track_bboxes[:, 0].astype(np.int32)
            bboxes_5 = track_bboxes[:, 1:6].astype(np.float32)

        # pad/clip before pose
        b2 = []
        ids2 = []
        for i in range(bboxes_5.shape[0]):
            b = pad_bbox_xyxy(bboxes_5[i], args.bbox_pad, w, h)
            b2.append(b)
            ids2.append(track_ids[i])
        if len(b2) > 0:
            bboxes_5 = np.stack(b2, axis=0).astype(np.float32)
            track_ids = np.asarray(ids2, dtype=np.int32)

        # 5) Pose inference
        pose_results = []
        if bboxes_5.shape[0] > 0:
            # IMPORTANT: this ViTPose/MMPose fork expects list[dict] with 'bbox'
            person_results = [{"bbox": bboxes_5[i]} for i in range(bboxes_5.shape[0])]

            pose_results, _ = inference_top_down_pose_model(
                pose_model,
                frame,
                person_results,
                bbox_thr=0.0,  # we already thresholded on det_thr; keep pose stage permissive
                format="xyxy",
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=False,
            )

            for i, r in enumerate(pose_results):
                r["track_id"] = int(track_ids[i]) if i < len(track_ids) else -1

        # 5b) Refine each person's bbox from their detected keypoints
        n_refined = 0
        if args.refine_bbox_from_pose and pose_results:
            for r in pose_results:
                kbb = keypoint_bbox(r.get("keypoints"), args.refine_kpt_thr, args.refine_min_kpts)
                if kbb is None:
                    continue
                nx1, ny1, nx2, ny2 = expand_bbox(kbb, args.refine_margin, w, h)
                if nx2 - nx1 < 2.0 or ny2 - ny1 < 2.0:
                    continue
                old_bbox = np.asarray(r.get("bbox", [0, 0, 0, 0, 1.0]), dtype=np.float32).flatten()
                old_area = bbox_area_xyxy(old_bbox[:4]) if old_bbox.size >= 4 else 0.0
                new_area = (nx2 - nx1) * (ny2 - ny1)
                # Only refine when the pose box is meaningfully tighter than the detector box.
                if old_area > 0 and new_area > args.refine_shrink_ratio * old_area:
                    continue
                score = float(old_bbox[4]) if old_bbox.size >= 5 else 1.0
                new_bbox = np.array([nx1, ny1, nx2, ny2, score], dtype=np.float32)
                r["bbox"] = new_bbox
                # Propagate the refined bbox back to the tracker so the next frame's
                # IoU and motion prior are computed from the tighter box.
                tid = int(r.get("track_id", -1))
                if tid > 0:
                    for tr in tracker.tracks:
                        if tr.track_id == tid:
                            tr.bbox = new_bbox
                            new_hist = _hsv_hist(frame, new_bbox)
                            if new_hist is not None:
                                tr.hist = new_hist
                            break
                n_refined += 1

        # 6) Save predictions
        npy_path = os.path.join(frames_dir, f"frame_{saved:06d}.npy")
        np.save(npy_path, np.array(pose_results, dtype=object), allow_pickle=True)

        # 7) Visualization
        if args.save_vis:
            if HAS_REPO_VIS:
                vis = vis_pose_tracking_result(
                    pose_model,
                    frame,
                    pose_results,
                    radius=2,
                    thickness=1,
                    dataset=dataset,
                    dataset_info=dataset_info,
                    kpt_score_thr=args.kpt_thr,
                    show=False,
                    sort=True,
                    vis_bg=True,
                )
            else:
                vis = fallback_draw(frame, pose_results, kpt_thr=args.kpt_thr)

            jpg_path = os.path.join(vis_dir, f"{saved:06d}.jpg")
            cv2.imwrite(jpg_path, vis)
            if video_writer is not None:
                video_writer.write(vis)

        if saved % max(1, args.log_every) == 0:
            logger.info(
                f"frame={frame_idx} saved={saved} dets={dets.shape[0] if dets is not None else 0} "
                f"dropped_huge={n_dropped_huge} tracks={bboxes_5.shape[0]} poses={len(pose_results)} "
                f"refined={n_refined} elapsed={(time.time()-t0):.1f}s"
            )

        frame_idx += 1
        saved += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()

    meta = {
        "video": args.video_path,
        "frames_saved": saved,
        "det_config": args.det_config,
        "det_checkpoint": args.det_checkpoint,
        "det_class_idx": args.det_class_idx,
        "det_thr": args.det_thr,
        "max_dets": args.max_dets,
        "tracker_iou": args.tracker_iou,
        "tracker_max_lost": args.tracker_max_lost,
        "pose_config": args.pose_config,
        "pose_checkpoint": args.pose_checkpoint,
        "device": args.device,
        "out_root": args.out_root,
        "vis_video": out_vid if args.save_vis else None,
        "max_bbox_area_frac": args.max_bbox_area_frac,
        "refine_bbox_from_pose": bool(args.refine_bbox_from_pose),
        "refine_min_kpts": args.refine_min_kpts,
        "refine_kpt_thr": args.refine_kpt_thr,
        "refine_margin": args.refine_margin,
        "refine_shrink_ratio": args.refine_shrink_ratio,
    }
    with open(os.path.join(out_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Saved {saved} frames to: {frames_dir}")
    if args.save_vis:
        print(f"Visualization saved to: {vis_dir}")
        if out_vid is not None:
            print(f"Video saved to: {out_vid}")


if __name__ == "__main__":
    main()
