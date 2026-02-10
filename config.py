"""
Configuration hub for YOLO Vision Studio.
All paths, model configs, UI settings, and constants are defined here.
"""

from pathlib import Path
import os
import sys

# ─── Paths ───────────────────────────────────────────────────────────────────
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ASSETS_DIR = ROOT / "assets"
IMAGES_DIR = ROOT / "images"
VIDEOS_DIR = ROOT / "videos"
WEIGHTS_DIR = ROOT / "weights"

# Ensure weights directory exists
WEIGHTS_DIR.mkdir(exist_ok=True)

# ─── App Metadata ────────────────────────────────────────────────────────────
APP_TITLE = "YOLO Vision Studio"
APP_ICON = "🔬"
APP_VERSION = "2.1.0"
APP_DESCRIPTION = (
    "Real-time Object Detection, Segmentation, Pose Estimation & Tracking "
    "powered by YOLO26, YOLO World v2, RT-DETR & Streamlit"
)

# ─── Inference Modes ─────────────────────────────────────────────────────────
MODE_IMAGE = "📷 Image Inference"
MODE_VIDEO = "🎬 Video Inference"
MODES_LIST = [MODE_IMAGE, MODE_VIDEO]

# ─── Tasks ───────────────────────────────────────────────────────────────────
TASK_DETECT = "Detection"
TASK_SEGMENT = "Segmentation"
TASK_WORLD = "YOLO World v2 (Text Prompt)"
TASK_POSE = "Pose Estimation"
TASKS_LIST = [TASK_DETECT, TASK_SEGMENT, TASK_WORLD, TASK_POSE]

# ─── Video Sources ───────────────────────────────────────────────────────────
SOURCE_STORED = "Stored Video"
SOURCE_WEBCAM = "Webcam"
SOURCE_RTSP = "RTSP Stream"
SOURCE_YOUTUBE = "YouTube"
VIDEO_SOURCES = [SOURCE_STORED, SOURCE_WEBCAM, SOURCE_RTSP, SOURCE_YOUTUBE]

# ─── Model Catalog ────────────────────────────────────────────────────────────
# Each task has a dict of {display_label: model_filename}.
# Ultralytics auto-downloads any model not already in weights/.

DETECTION_MODELS = {
    "YOLO26-nano (fastest)": "yolo26n.pt",
    "YOLO26-small": "yolo26s.pt",
    "YOLO26-medium": "yolo26m.pt",
    "YOLO26-large": "yolo26l.pt",
    "YOLO26-xlarge (best accuracy)": "yolo26x.pt",
    "RT-DETR-Large (transformer)": "rtdetr-l.pt",
    "RT-DETR-XLarge (transformer)": "rtdetr-x.pt",
}

SEGMENTATION_MODELS = {
    "YOLO26-nano-seg (fastest)": "yolo26n-seg.pt",
    "YOLO26-small-seg": "yolo26s-seg.pt",
    "YOLO26-medium-seg": "yolo26m-seg.pt",
    "YOLO26-large-seg": "yolo26l-seg.pt",
    "YOLO26-xlarge-seg (best accuracy)": "yolo26x-seg.pt",
}

POSE_MODELS = {
    "YOLO26-nano-pose (fastest)": "yolo26n-pose.pt",
    "YOLO26-small-pose": "yolo26s-pose.pt",
    "YOLO26-medium-pose": "yolo26m-pose.pt",
    "YOLO26-large-pose": "yolo26l-pose.pt",
    "YOLO26-xlarge-pose (best accuracy)": "yolo26x-pose.pt",
}

WORLD_MODELS = {
    "YOLOv8-small-worldv2": "yolov8s-worldv2.pt",
    "YOLOv8-medium-worldv2": "yolov8m-worldv2.pt",
    "YOLOv8-large-worldv2 (recommended)": "yolov8l-worldv2.pt",
    "YOLOv8-xlarge-worldv2 (best accuracy)": "yolov8x-worldv2.pt",
}

# Defaults (first key in each dict)
DETECTION_MODEL = "yolo26n.pt"
SEGMENTATION_MODEL = "yolo26n-seg.pt"
POSE_MODEL = "yolo26n-pose.pt"

# YOLO World v2: open-vocabulary detection via natural language text prompts
YOLO_WORLD_MODEL = "yolov8l-worldv2.pt"

# ─── Default Assets ──────────────────────────────────────────────────────────
DEFAULT_IMAGE = IMAGES_DIR / "office_4.jpg"
DEFAULT_DETECT_IMAGE = IMAGES_DIR / "office_4_detected.jpg"

# ─── Video Catalog ───────────────────────────────────────────────────────────
VIDEOS_DICT = (
    {name.stem: name for name in sorted(VIDEOS_DIR.glob("*.mp4"))}
    if VIDEOS_DIR.exists()
    else {}
)

# ─── Inference Defaults ──────────────────────────────────────────────────────
DEFAULT_CONFIDENCE = 0.40
DEFAULT_IOU = 0.50
MIN_CONFIDENCE = 10  # slider min (%)
MAX_CONFIDENCE = 100  # slider max (%)
VIDEO_DISPLAY_WIDTH = 720
WEBCAM_PATH = 0

# ─── Skip Frames ─────────────────────────────────────────────────────────────
DEFAULT_SKIP_FRAMES = 1  # process every frame
MIN_SKIP_FRAMES = 1
MAX_SKIP_FRAMES = 8

# ─── Tracker Config ──────────────────────────────────────────────────────────
TRACKER_BYTETRACK = "bytetrack.yaml"
TRACKER_BOTSORT = "botsort.yaml"
TRACKERS_LIST = [TRACKER_BYTETRACK, TRACKER_BOTSORT]

# ─── YOLO World v2 Defaults ───────────────────────────────────────────────────
# Supports natural language prompts like "person in black", "red car", etc.
DEFAULT_WORLD_CLASSES = "person, car, dog, cat, chair, table, laptop, phone"


def resolve_model_path(model_name: str) -> str:
    """Return local weights path if it exists, else the bare name for auto-download.

    After auto-download, call ``sweep_stray_weights()`` to move any
    ``.pt`` files that landed in the project root into ``weights/``.
    """
    local = WEIGHTS_DIR / model_name
    if local.exists():
        return str(local)
    # Not in weights/ yet — check project root (old download location)
    root_copy = ROOT / model_name
    if root_copy.exists():
        root_copy.rename(local)
        return str(local)
    # Will be auto-downloaded to CWD; return bare name
    return model_name


def sweep_stray_weights() -> None:
    """Move any ``.pt`` files from the project root into ``weights/``."""
    for pt_file in ROOT.glob("*.pt"):
        dest = WEIGHTS_DIR / pt_file.name
        if not dest.exists():
            pt_file.rename(dest)


def get_model_catalog(task: str) -> dict[str, str]:
    """Return ``{display_label: filename}`` for the given *task*."""
    _CATALOGS = {
        TASK_DETECT: DETECTION_MODELS,
        TASK_SEGMENT: SEGMENTATION_MODELS,
        TASK_POSE: POSE_MODELS,
        TASK_WORLD: WORLD_MODELS,
    }
    return _CATALOGS.get(task, DETECTION_MODELS)
