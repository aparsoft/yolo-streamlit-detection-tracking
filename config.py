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
TASK_YOLOE = "YOLOE (Text → Segmentation)"
TASK_POSE = "Pose Estimation"
TASKS_LIST = [TASK_DETECT, TASK_SEGMENT, TASK_WORLD, TASK_YOLOE, TASK_POSE]

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

YOLOE_MODELS = {
    "YOLOE-26n-seg (fastest)": "yoloe-26n-seg.pt",
    "YOLOE-26s-seg": "yoloe-26s-seg.pt",
    "YOLOE-26m-seg": "yoloe-26m-seg.pt",
    "YOLOE-26l-seg (recommended)": "yoloe-26l-seg.pt",
    "YOLOE-26x-seg (best accuracy)": "yoloe-26x-seg.pt",
}

# Defaults (first key in each dict)
DETECTION_MODEL = "yolo26n.pt"
SEGMENTATION_MODEL = "yolo26n-seg.pt"
POSE_MODEL = "yolo26n-pose.pt"

# YOLO World v2: open-vocabulary detection via natural language text prompts
YOLO_WORLD_MODEL = "yolov8l-worldv2.pt"

# YOLOE: open-vocabulary text-prompted detection + segmentation
YOLOE_MODEL = "yoloe-26l-seg.pt"

# ─── Default Assets ──────────────────────────────────────────────────────────
DEFAULT_IMAGE = IMAGES_DIR / "office_4.jpg"
DEFAULT_DETECT_IMAGE = IMAGES_DIR / "office_4_detected.jpg"

# ─── Video Catalog ───────────────────────────────────────────────────────────
_VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.wmv", "*.webm")


def get_videos_dict() -> dict[str, Path]:
    """Scan ``videos/`` directory each time so newly added files appear."""
    if not VIDEOS_DIR.exists():
        return {}
    vids: dict[str, Path] = {}
    for ext in _VIDEO_EXTENSIONS:
        for p in sorted(VIDEOS_DIR.glob(ext)):
            vids[p.stem] = p
    return dict(sorted(vids.items()))


# Kept for backward compat — but prefer get_videos_dict()
VIDEOS_DICT = get_videos_dict()

# ─── Inference Defaults ──────────────────────────────────────────────────────
DEFAULT_CONFIDENCE = 0.40
DEFAULT_IOU = 0.50
MIN_CONFIDENCE = 10  # slider min (%)
MAX_CONFIDENCE = 100  # slider max (%)
VIDEO_DISPLAY_WIDTH = 720  # a *ceiling*, never a target — see _display_size()
WEBCAM_PATH = 0

# ─── Video Streaming Budget ──────────────────────────────────────────────────
# A playback loop lives inside one Streamlit script run, and the parts of Streamlit
# that reclaim memory and flush the socket only run *between* script runs. Everything
# below exists to keep one long run from starving them.

# JPEG quality for streamed frames. 90 → 82 KB/frame, 75 → 51 KB/frame at 720p; the
# difference is invisible on an annotated frame and it is 38% less to push per frame.
VIDEO_JPEG_QUALITY = 75

# Sidebar metrics refresh rate. Each update is ~7 ForwardMsgs; at 85 fps that is 600
# messages/s the browser must apply, for numbers no one can read that fast.
METRICS_REFRESH_HZ = 5.0

# Ceiling on how many frames a second are *painted*. Inference still runs on every frame
# the skip-frames slider allows, so tracking quality is untouched — this only limits how
# many of those results are pushed to the browser.
#
# The two rates were the same number before, which is why playback stuttered: the loop
# produced ~78 fps of distinct JPEGs and each one costs the browser a ForwardMsg *and* a
# separate GET of /media/<hash>.jpg. Nothing in a human eye needs 78; capping at 30
# roughly halves the browser's work without dropping a single detection.
VIDEO_DISPLAY_FPS = 30.0

# How often to collect the JPEGs of frames already replaced on screen. Streamlit only
# does this when a script run ends, so a 1000-frame video otherwise holds ~80 MB.
MEDIA_GC_EVERY_N_FRAMES = 30

# ─── Skip Frames ─────────────────────────────────────────────────────────────
DEFAULT_SKIP_FRAMES = 1  # process every frame
MIN_SKIP_FRAMES = 1
MAX_SKIP_FRAMES = 8

# ─── Tracker Config ──────────────────────────────────────────────────────────
TRACKER_BYTETRACK = "bytetrack.yaml"
TRACKER_BOTSORT = "botsort.yaml"
TRACKER_DEEPOCSORT = "deepocsort.yaml"
TRACKER_TRACKTRACK = "tracktrack.yaml"
TRACKER_OCSORT = "ocsort.yaml"
TRACKER_FASTTRACK = "fasttrack.yaml"

TRACKERS_LIST = [
    TRACKER_BYTETRACK,
    TRACKER_BOTSORT,
    TRACKER_DEEPOCSORT,
    TRACKER_TRACKTRACK,
]

# Trackers whose YAML has a ``with_reid`` switch (appearance matching).
REID_CAPABLE_TRACKERS = {TRACKER_BOTSORT, TRACKER_DEEPOCSORT, TRACKER_TRACKTRACK}

# ─── ReID (appearance re-identification) ─────────────────────────────────────
# ReID gives the tracker a second opinion: instead of "is this box where the last
# one was?" it can also ask "does this crop look like that object?" — which is how
# an ID survives an occlusion. It costs an extra forward pass per detection crop.
#
# ``model: auto`` is documented as "use the detector's own features", but that path
# needs a non-end2end head. YOLO26 is NMS-free (end2end=True), so auto silently
# falls back to yolo26n-cls.pt. We name that file explicitly: no surprise download,
# and it already sits in weights/.
REID_ENCODER_CHOICES = {
    "yolo26n-cls (local, weak)": str(WEIGHTS_DIR / "yolo26n-cls.pt"),
    "yolo26s-reid.onnx (needs onnxruntime)": "yolo26s-reid.onnx",
    "detector features (auto)": "auto",
}
REID_ENCODER = str(WEIGHTS_DIR / "yolo26n-cls.pt")
REID_PROXIMITY_THRESH = 0.30  # min IoU before appearance votes (lower = more rescues)
REID_APPEARANCE_THRESH = 0.55  # min appearance similarity (lower = more rescues)
REID_TRACK_BUFFER = 60  # frames a lost track stays re-findable
GMC_METHOD = "none"  # camera-motion compensation; "sparseOptFlow" if the camera moves

# A track must be seen this many frames before it counts as a real object. Without
# it every one-frame false positive inflates the "unique objects" headline number.
MIN_TRACK_HITS = 5

# ─── YOLO World v2 Defaults ───────────────────────────────────────────────────
# Supports natural language prompts like "person in black", "red car", etc.
DEFAULT_WORLD_CLASSES = "person, car, dog, cat, chair, table, laptop, phone"

# ─── YOLOE Defaults ──────────────────────────────────────────────────────────
# YOLOE supports category-level text prompts (NOT descriptive phrases).
# Unlike YOLO World v2, YOLOE provides detection + segmentation masks.
DEFAULT_YOLOE_CLASSES = "person, car, dog, cat, chair, table, laptop, phone"


def use_local_weights_dir() -> str | None:
    """Point Ultralytics' asset lookup at this repo's ``weights/``.

    Ultralytics resolves a missing asset against the current directory first and
    then ``SETTINGS["weights_dir"]``. Unless that setting points here, a weight we
    already have in ``weights/`` gets downloaded a second time into whatever
    directory the process happens to be running from.

    Returns the previous value if it was changed, else ``None``.
    """
    from ultralytics.utils import SETTINGS  # local import keeps config import cheap

    previous = SETTINGS["weights_dir"]
    if previous == str(WEIGHTS_DIR):
        return None
    SETTINGS["weights_dir"] = str(WEIGHTS_DIR)
    return previous


def resolve_model_path(model_name: str) -> str:
    """Return local weights path if it exists, else the bare name for auto-download.

    After auto-download, call ``sweep_stray_weights()`` to move any
    weight files that landed in the project root into ``weights/``.
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


# What an auto-download can drop in the project root: ``.pt`` checkpoints and the
# ``.ts`` TorchScript CLIP text encoder that YOLO World v2 / YOLOE pull in.
_STRAY_WEIGHT_PATTERNS = ("*.pt", "*.ts")


def sweep_stray_weights() -> None:
    """Move any stray weight files from the project root into ``weights/``."""
    for pattern in _STRAY_WEIGHT_PATTERNS:
        for stray in ROOT.glob(pattern):
            dest = WEIGHTS_DIR / stray.name
            if not dest.exists():
                stray.rename(dest)


def get_model_catalog(task: str) -> dict[str, str]:
    """Return ``{display_label: filename}`` for the given *task*."""
    _CATALOGS = {
        TASK_DETECT: DETECTION_MODELS,
        TASK_SEGMENT: SEGMENTATION_MODELS,
        TASK_POSE: POSE_MODELS,
        TASK_WORLD: WORLD_MODELS,
        TASK_YOLOE: YOLOE_MODELS,
    }
    return _CATALOGS.get(task, DETECTION_MODELS)
