"""
Configuration hub for YOLO Vision Studio.
All paths, model configs, UI settings, and constants are defined here.
"""

from pathlib import Path
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

# ─── App Metadata ────────────────────────────────────────────────────────────
APP_TITLE = "YOLO Vision Studio"
APP_ICON = "🔬"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = (
    "Real-time Object Detection, Segmentation, Pose Estimation & Tracking "
    "powered by YOLO26, YOLO World v2 & Streamlit"
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

# ─── Model Paths ─────────────────────────────────────────────────────────────
# YOLO26 models — auto-downloaded by ultralytics on first run
DETECTION_MODEL = "yolo26n.pt"
SEGMENTATION_MODEL = "yolo26n-seg.pt"
POSE_MODEL = "yolo26n-pose.pt"

# YOLO World v2: open-vocabulary detection via natural language text prompts
# Supports descriptive prompts like "person in black", "red car", etc.
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
    """Return local weights path if it exists, else the name for auto-download."""
    local = WEIGHTS_DIR / model_name
    return str(local) if local.exists() else model_name
