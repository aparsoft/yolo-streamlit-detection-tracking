<div align="center">

# 🔬 YOLO Vision Studio

**Real-time Object Detection · Segmentation · Pose Estimation · Tracking**
**Powered by YOLO26, YOLOE, YOLO World v2, RT-DETR & Streamlit**

[![Stars](https://img.shields.io/github/stars/aparsoft/yolo-streamlit-detection-tracking?style=for-the-badge&logo=github)](https://github.com/aparsoft/yolo-streamlit-detection-tracking/stargazers)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4+-purple?style=for-the-badge)](https://ultralytics.com)
[![License](https://img.shields.io/github/license/aparsoft/yolo-streamlit-detection-tracking?style=for-the-badge)](LICENSE)

[Live Demo](https://yolov8-object-detection-and-tracking-app.streamlit.app/) · [Blog Series](https://rs-punia.medium.com/building-a-real-time-object-detection-and-tracking-app-with-yolov8-and-streamlit-part-1-30c56f5eb956) · [Report Bug](https://github.com/aparsoft/yolo-streamlit-detection-tracking/issues)

</div>

---

## 🆕 What's New in v2.1

- **More models, one selector**: every YOLO26 size (nano → xlarge) per task, **RT-DETR** (transformer) for detection, and **YOLOE** for text-prompted detection *with masks*.
- **Four trackers**: ByteTrack, BoT-SORT, Deep OC-SORT and TrackTrack, with optional **appearance ReID** and sidebar-tunable gates.
- **Honest counts**: an object counts once it has been tracked for 5 frames, and a *churn* figure shows when the tracker is fragmenting IDs.
- **Limit to classes**, **multi-video** side-by-side runs, and a **⏹ Stop** button.
- **Fixes**: identical runs now give identical counts (a reset used to stack tracker callbacks, so every run after the first was counted differently and ran slower); each browser session gets its own tracking model, so two visitors never share a tracker; the default image runs every task; phone photos keep their orientation; `requirements.txt` installs on Windows and macOS.

## What's New in v2.0

> **Thank you for 400+ ⭐ stars!** This major update brings a completely rewritten, modular codebase with exciting new capabilities.

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Object Detection | YOLOv8n | YOLO26n (NMS-free, 43% faster CPU) |
| Segmentation | YOLOv8n-seg | YOLO26n-seg (multi-scale proto + semantic loss) |
| Pose Estimation | ❌ | ✅ YOLO26n-pose (RLE-based keypoints) |
| Open-Vocabulary | ❌ | ✅ YOLO World v2 (natural language text prompts) |
| Tracking | Basic | ByteTrack + BoTSORT with local + global counting |
| Object Counting | ❌ | ✅ Per-frame local + cumulative global counts |
| Skip Frames | ❌ | ✅ 1–8× skip for fast inference on long videos |
| Webcam | OpenCV (broken in cloud) | ✅ streamlit-webrtc (browser-native) |
| Architecture | Monolithic | Modular service-based design |
| Video Metrics | ❌ | ✅ Live FPS, local/global counts & tracking overlay |
| Codebase | `helper.py + settings.py` | `config · model_loader · image_service · video_service` |

---

## ✨ Features

### 📷 Image Inference
- **Object Detection** — Detect 80+ COCO classes with YOLO26 (NMS-free, edge-optimized)
- **YOLO World v2 (Text Prompt)** — Natural language prompts like *"person in black"*, *"red car"*, *"laptop on table"* for open-vocabulary detection
- **YOLOE (Text → Segmentation)** — Category-level prompts (*person, car, laptop*) with instance masks
- **RT-DETR** — Transformer detector, selectable alongside every YOLO26 size
- **Instance Segmentation** — Pixel-level object segmentation with multi-scale proto modules
- **Pose Estimation** — Human body keypoint and skeleton detection with RLE precision
- Per-class metrics, confidence scores and detailed results table
- Works on the bundled default image straight away: no upload needed to try a task

### 🎬 Video Inference
- **Multiple Sources**: Stored videos, Webcam (browser-native via WebRTC), RTSP streams, YouTube URLs
- **Real-time Tracking**: ByteTrack, BoT-SORT, Deep OC-SORT and TrackTrack (enabled by default), with optional appearance ReID
- **Limit to Classes**: track only the classes you care about (cheaper, and cleaner counts)
- **Multi-Video**: pick several stored videos and run them side by side, each with its own tracker
- **Local + Global Counting**: Per-frame counts (green) and cumulative unique-object counts (yellow) displayed on every frame
- **Skip Frames**: Adjustable 1–8× slider for faster inference on long or high-FPS videos
- **YOLO World v2 in Video**: Natural language text-prompt search in video streams
- **Live Metrics**: Separate local (this frame) and global (cumulative) sections in sidebar
- **Count Overlay**: On-frame badge — local in green, global in yellow, track quality (churn) in grey
- **⏹ Stop**: ends a playback and keeps the counts so far

### 🏗️ Architecture
- **Modular Design**: Separate services for image and video inference
- **Centralized Config**: Single `config.py` for all settings
- **Cached Models**: `@st.cache_resource` for instant model reuse; tracking runs get a session-owned model so visitors never share tracker state
- **Clean Routing**: Task + Mode based dispatch in `app.py`

---

## 📸 Demo

### Tracking with Object Detection
<https://user-images.githubusercontent.com/104087274/234874398-75248e8c-6965-4c91-9176-622509f0ad86.mov>

### Application Overview
<https://github.com/user-attachments/assets/85df351a-371c-47e0-91a0-a816cf468d19.mov>

### Screenshots

| Home Page | Detection Result | Segmentation |
|:---------:|:----------------:|:------------:|
| <img src="assets/pic1.png" width="300"> | <img src="assets/pic3.png" width="300"> | <img src="assets/segmentation.png" width="300"> |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher (tested on 3.12)
- GPU recommended (NVIDIA CUDA) for real-time video inference
- Webcam (optional, for live detection)

### Installation with uv (recommended)

[uv](https://docs.astral.sh/uv/) installs the exact versions in `uv.lock`, on Linux, macOS and Windows. Thanks to [@AmirMahdiRezaeiEECS](https://github.com/AmirMahdiRezaeiEECS) for contributing this setup ([#19](https://github.com/aparsoft/yolo-streamlit-detection-tracking/pull/19)).

```bash
# Clone the repository
git clone https://github.com/aparsoft/yolo-streamlit-detection-tracking.git
cd yolo-streamlit-detection-tracking

# Install uv once (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv and install the locked dependencies
uv sync

# Run the app
uv run streamlit run app.py
```

On Linux, PyTorch from PyPI already includes CUDA. On Windows, for a CUDA build, install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) into the environment first.

### Installation with pip

```bash
# 1. Clone the repository
git clone https://github.com/aparsoft/yolo-streamlit-detection-tracking.git
cd yolo-streamlit-detection-tracking

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` works on Linux, Windows and macOS. `requirements-lock.txt` pins the exact Linux + CUDA 12.8 environment the app was developed and tested in. For a specific CUDA build of PyTorch, install it first from [pytorch.org](https://pytorch.org/get-started/locally/).

### Download Model Weights

The default detection and segmentation weights are auto-downloaded by Ultralytics on first use. All YOLO26 models (detection, segmentation, pose) and YOLO World v2 (open-vocabulary) are fetched automatically.

To pre-download manually:

```bash
# Detection
wget -P weights/ https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

# Segmentation
wget -P weights/ https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt

# Pose estimation
wget -P weights/ https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt

# YOLO World v2 open-vocabulary (auto-downloads if not present)
wget -P weights/ https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-worldv2.pt
```

### Run the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## 📖 Usage Guide

### Sidebar Controls

1. **Inference Mode** — Choose between 📷 *Image Inference* or 🎬 *Video Inference*
2. **Task** — Select one of:
   - **Detection** — YOLO26 (NMS-free, end-to-end) or RT-DETR
   - **Segmentation** — Instance segmentation with pixel masks
   - **YOLO World v2 (Text Prompt)** — Open-vocabulary detection with natural language prompts
   - **YOLOE (Text → Segmentation)** — Open-vocabulary detection + masks from category names
   - **Pose Estimation** — Human body keypoint detection
3. **🧠 Model** — Pick the model size (nano is fastest; xlarge is most accurate)
4. **Model Confidence** — Adjust the confidence threshold (10–100%)

### Image Inference

1. Select **📷 Image Inference** mode
2. Choose a task (Detection, Segmentation, YOLO World, YOLOE or Pose)
3. Upload an image, or run on the default one
4. For **YOLO World v2**: type descriptive phrases (e.g., `person in black, red car, laptop on table`)
5. Click **🚀 Run** to see results with per-class metrics

### Video Inference

1. Select **🎬 Video Inference** mode
2. Choose a task
3. Pick a video source: **Stored Video**, **Webcam**, **RTSP**, or **YouTube**
4. **Object Tracking** is enabled by default (ByteTrack, BoT-SORT, Deep OC-SORT or TrackTrack; ReID optional) — local + global counts display automatically
5. Optionally **🎯 Limit to classes**, and adjust **Skip Frames** (1–8) for faster inference on long videos
6. For **YOLO World v2** / **YOLOE**: enter the prompts to search for in the video
7. Click **🚀 Detect** — local and global metrics appear in the sidebar; **⏹ Stop** ends it early

### Sample Videos

Nine short clips ship in `videos/`: street crossings, traffic, a red car, a person in red, dogs and bikes, and two pose scenes. They are from [Pexels](https://www.pexels.com/license/), credited in [`videos/ATTRIBUTION.md`](videos/ATTRIBUTION.md), and were picked by running this app's own detector and tracker over 731 candidates. Older clips live in `videos/archive/`.

### Adding Your Own Videos

Drop `.mp4` (or `.avi`, `.mkv`, `.mov`, `.webm`) files into the `videos/` directory. They appear in the stored-video dropdown on the next rerun, with no code changes (the folder is scanned every time).

---

## 🗂️ Project Structure

```
yolo-streamlit-detection-tracking/
├── app.py                # Main Streamlit application & routing
├── config.py             # Centralized configuration (paths, models, UI)
├── model_loader.py       # Model loading with @st.cache_resource
├── image_service.py      # Image inference (detection, segmentation, world, pose)
├── video_service.py      # Video inference (tracking, counting, all sources)
├── requirements.txt      # Python dependencies (any OS)
├── requirements-lock.txt # exact Linux + CUDA environment
├── packages.txt          # System packages for Streamlit Cloud
├── README.md
├── assets/               # Screenshots and demo media
├── images/               # Sample images
├── videos/               # Sample videos (add your .mp4 files here)
└── weights/              # Model weights (yolo26n.pt, yolo26n-seg.pt, ...; auto-downloaded)
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | All paths, model names, UI constants, and default values |
| `model_loader.py` | Cached model loading; resolves local weights vs auto-download |
| `image_service.py` | Full image-mode UI: upload → inference → results display |
| `video_service.py` | Full video-mode UI: source selection → frame loop → live metrics |
| `app.py` | Page config, sidebar, and routing to the correct service |

---

## ⚙️ Configuration

All configuration lives in `config.py`. Key settings:

```python
# Models — change to larger variants for better accuracy
DETECTION_MODEL    = "yolo26n.pt"        # or yolo26s.pt, yolo26m.pt, yolo26l.pt
SEGMENTATION_MODEL = "yolo26n-seg.pt"    # or yolo26s-seg.pt
YOLO_WORLD_MODEL   = "yolov8l-worldv2.pt" # open-vocabulary (natural language)
POSE_MODEL         = "yolo26n-pose.pt"   # or yolo26s-pose.pt

# Inference defaults
DEFAULT_CONFIDENCE = 0.40
DEFAULT_IOU        = 0.50
VIDEO_DISPLAY_WIDTH = 720

# Skip-frame control for video inference
DEFAULT_SKIP_FRAMES = 1   # process every frame (1–8)

# YOLO World v2 default prompts
DEFAULT_WORLD_CLASSES = "person, car, dog, cat, chair, table, laptop, phone"
```

### Custom Models

To use your own trained model:

```python
# In config.py
DETECTION_MODEL = "my_custom_model.pt"
# Place the .pt file in the weights/ directory
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file path to `app.py`
4. The `packages.txt` file handles system-level dependencies automatically

> **Note**: Streamlit Cloud has no GPU — video inference will be slower. Image inference works well. Webcam uses **streamlit-webrtc** so it works natively in the browser (no server-side camera access needed).

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m "Add amazing feature"`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Ideas for Contributions

- [ ] Add model benchmarking / comparison page
- [ ] Export detection results to CSV / JSON
- [ ] Add YOLO-NAS or RT-DETR model support
- [ ] Region of Interest (ROI) based counting
- [ ] Multi-camera RTSP dashboard

---

## 📚 Resources

- [Ultralytics YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [YOLO World v2 Documentation](https://docs.ultralytics.com/models/yolo-world/)
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) — Browser-native webcam
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [Blog Series — Building this App](https://medium.com/@mycodingmantras/building-a-real-time-object-detection-and-tracking-app-with-yolov8-and-streamlit-part-1-30c56f5eb956)

---

## 📄 License

This project's code is released under the [Apache License 2.0](LICENSE).

The models it runs come from [Ultralytics](https://github.com/ultralytics/ultralytics), which is licensed under **AGPL-3.0** (with an Ultralytics Enterprise License for closed-source commercial use). Check those terms before you ship a product built on them.

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO26 and YOLO World v2
- [Streamlit](https://github.com/streamlit/streamlit) for the web framework
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) for browser-based webcam
- All **400+** stargazers for the love and support!

---

<div align="center">

**If you find this project useful, please consider giving it a ⭐!**

Made with ❤️ by [Aparsoft](https://aparsoft.com/)

</div>
