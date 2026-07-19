import json
import os
import subprocess
import sys
from pathlib import Path


def run_command(command, env, working_directory):
    """Run one pipeline stage and stop if it fails."""
    print("\nRunning command:")
    print(" ".join(str(item) for item in command))

    subprocess.run(
        [str(item) for item in command],
        check=True,
        env=env,
        cwd=str(working_directory)
    )


def process_video(
    video_path,
    output_root="outputs_all_clips",
    fps=30,
    device="cuda:0"
):
    project_root = Path.cwd().resolve()
    handoff_root = project_root / "hybrid_pipeline_handoff"

    video_path = Path(video_path).expanduser().resolve()
    output_root = Path(output_root)

    if not output_root.is_absolute():
        output_root = project_root / output_root

    output_root = output_root.resolve()
    position_directory = output_root / "position_predictions"

    tracking_script = (
        handoff_root
        / "src"
        / "scripts"
        / "run_tracking_detr_iou_ROBUST.py"
    )

    position_script = (
        handoff_root
        / "src"
        / "scripts"
        / "predict_positions_from_yolo.py"
    )

    style_script = (
        handoff_root
        / "src"
        / "scripts"
        / "analyze_player_styles.py"
    )

    yolo_weights = (
        handoff_root
        / "weights"
        / "bjj_yolo18_domain.pt"
    )

    detection_config = (
        project_root
        / "mmdetection"
        / "configs"
        / "deformable_detr"
        / "deformable_detr_twostage_refine_r50_16x2_50e_coco.py"
    )

    detection_checkpoint = (
        project_root
        / "checkpoints"
        / "detection"
        / "deformable_detr_twostage_refine.pth"
    )

    pose_config = (
        project_root
        / "ViTPose"
        / "configs"
        / "body"
        / "2d_kpt_sview_rgb_img"
        / "topdown_heatmap"
        / "coco"
        / "ViTPose_base_coco_256x192.py"
    )

    pose_checkpoint = (
        project_root
        / "checkpoints"
        / "pose"
        / "vitpose.pth"
    )

    required_files = {
        "Input video": video_path,
        "Tracking script": tracking_script,
        "Position prediction script": position_script,
        "Style analysis script": style_script,
        "YOLO weights": yolo_weights,
        "Detection config": detection_config,
        "Detection checkpoint": detection_checkpoint,
        "Pose config": pose_config,
        "Pose checkpoint": pose_checkpoint
    }

    missing_files = [
        f"{name}: {path}"
        for name, path in required_files.items()
        if not path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(missing_files)

        raise FileNotFoundError(
            "The following required files were not found:\n"
            f"{missing_text}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    position_directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    source_directory = handoff_root / "src"
    old_python_path = env.get("PYTHONPATH", "")

    if old_python_path:
        env["PYTHONPATH"] = (
            f"{source_directory}{os.pathsep}{old_python_path}"
        )
    else:
        env["PYTHONPATH"] = str(source_directory)

    # --------------------------------------------------
    # Stage 1: DETR + IoU tracker + ViTPose
    # --------------------------------------------------

    tracking_command = [
        sys.executable,
        tracking_script,

        "--video-path",
        video_path,

        "--det-config",
        detection_config,

        "--det-checkpoint",
        detection_checkpoint,

        "--pose-config",
        pose_config,

        "--pose-checkpoint",
        pose_checkpoint,

        "--device",
        device,

        "--out-root",
        output_root,

        "--save-vis",

        "--det-thr",
        "0.10",

        "--max-dets",
        "12",

        "--tracker-iou",
        "0.15",

        "--tracker-max-lost",
        "90",

        "--tracker-alpha-app",
        "0.65",

        "--tracker-app-gate",
        "0.60",

        "--tracker-max-center-jump-frac",
        "0.25",

        "--bbox-pad",
        "0.15"
    ]

    run_command(
        tracking_command,
        env=env,
        working_directory=project_root
    )

    # --------------------------------------------------
    # Stage 2: Hybrid YOLO position prediction
    # --------------------------------------------------

    position_command = [
        sys.executable,
        position_script,

        "--tracking-root",
        output_root,

        "--video-path",
        video_path,

        "--yolo-weights",
        yolo_weights,

        "--out-dir",
        position_directory,

        "--device",
        device,

        "--enforce-bottom-2",
        "--tb-use-pose",

        "--smooth-window",
        "15"
    ]

    run_command(
        position_command,
        env=env,
        working_directory=project_root
    )

    positions_csv = position_directory / "positions.csv"

    if not positions_csv.exists():
        raise FileNotFoundError(
            "Stage 2 finished, but positions.csv was not created:\n"
            f"{positions_csv}"
        )

    # --------------------------------------------------
    # Stage 3: Player style analysis
    # --------------------------------------------------

    style_command = [
        sys.executable,
        style_script,

        "--position-dir",
        position_directory,

        "--fps",
        str(fps),

        "--markdown"
    ]

    run_command(
        style_command,
        env=env,
        working_directory=project_root
    )

    json_path = position_directory / "style_analysis.json"

    if not json_path.exists():
        raise FileNotFoundError(
            "Style analysis finished, but style_analysis.json "
            "was not created:\n"
            f"{json_path}"
        )

    with json_path.open("r", encoding="utf-8") as file:
        style_analysis = json.load(file)

    print("\nPipeline completed successfully.")
    print(f"Output directory: {output_root}")
    print(f"Positions CSV: {positions_csv}")
    print(f"Style JSON: {json_path}")

    return style_analysis


if __name__ == "__main__":
    result = process_video(
        video_path="data/videos/input.mp4",
        output_root="outputs_all_clips",
        fps=30,
        device="cuda:0"
    )

    print(json.dumps(result, indent=4, ensure_ascii=False))