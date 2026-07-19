import cv2
import numpy as np
import os


class VideoProcessor:
    def __init__(self, target_fps=2, max_frames=120):
        self.target_fps = target_fps
        self.max_frames = max_frames

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps

        frame_interval = max(1, int(original_fps / self.target_fps))
        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append({
                    'image': frame_rgb,
                    'frame_number': frame_idx,
                    'timestamp': frame_idx / original_fps
                })
                if len(frames) >= self.max_frames:
                    break
            frame_idx += 1

        cap.release()
        return frames, duration, original_fps

    def load_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {
            'image': img_rgb,
            'frame_number': 0,
            'timestamp': 0.0
        }

    def resize_frame(self, frame, max_dim=640):
        h, w = frame.shape[:2]
        scale = min(max_dim / w, max_dim / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
