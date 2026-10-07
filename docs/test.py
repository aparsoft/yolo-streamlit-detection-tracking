from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # the repo, wherever it was cloned

# 1. Initialize an Edge-optimized YOLO26 Model
# Checkpoints auto-download from official Ultralytics repositories
model = YOLO("yolo26n.pt")  # Object Detection Model
# model_seg = YOLO("yolo26n-seg.pt")  # Instance Segmentation Model
# model_cls = YOLO("yolo26n-cls.pt")  # Classification Model

# 2. Run Accelerated Native Inference (NMS-Free for Detect)
results = model(str(ROOT / "images" / "office_4.jpg"))

# 3. Process Results
for result in results:
    boxes = result.boxes  # Bounding box coordinates
    probs = result.probs  # Classification probabilities (if cls)
    masks = result.masks  # Segmentation mask arrays (if seg)
    result.show()
