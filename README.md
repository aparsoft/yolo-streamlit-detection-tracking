<div align="center">

# 🔬 YOLO Vision Studio

**Real-time Object Detection · Segmentation · Pose Estimation · Tracking**
**Powered by YOLO26, YOLO World v2 & Streamlit**

[![Stars](https://img.shields.io/github/stars/CodingMantras/yolov8-streamlit-detection-tracking?style=for-the-badge&logo=github)](https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/stargazers)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3+-purple?style=for-the-badge)](https://ultralytics.com)
[![License](https://img.shields.io/github/license/aparsoft/yolov8-streamlit-detection-tracking?style=for-the-badge)](LICENSE)

[Live Demo](https://yolov8-object-detection-and-tracking-app.streamlit.app/) · [Blog Series](https://rs-punia.medium.com/building-a-real-time-object-detection-and-tracking-app-with-yolov8-and-streamlit-part-1-30c56f5eb956) · [Report Bug](https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/issues)

</div>

---

## 🆕 What's New in v2.0

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
- **Instance Segmentation** — Pixel-level object segmentation with multi-scale proto modules
- **Pose Estimation** — Human body keypoint and skeleton detection with RLE precision
- Per-class metrics, confidence scores and detailed results table

### 🎬 Video Inference
- **Multiple Sources**: Stored videos, Webcam (browser-native via WebRTC), RTSP streams, YouTube URLs
- **Real-time Tracking**: ByteTrack and BoTSORT algorithms (enabled by default)
- **Local + Global Counting**: Per-frame counts (green) and cumulative unique-object counts (yellow) displayed on every frame
- **Skip Frames**: Adjustable 1–8× slider for faster inference on long or high-FPS videos
- **YOLO World v2 in Video**: Natural language text-prompt search in video streams
- **Live Metrics**: Separate local (this frame) and global (cumulative) sections in sidebar
- **Count Overlay**: Two-line on-frame badge — local in green, global in yellow

### 🏗️ Architecture
- **Modular Design**: Separate services for image and video inference
- **Centralized Config**: Single `config.py` for all settings
- **Cached Models**: `@st.cache_resource` for instant model reuse
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
| <img src="https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/blob/master/assets/pic1.png" width="300"> | <img src="https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/blob/master/assets/pic3.png" width="300"> | <img src="https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/blob/master/assets/segmentation.png" width="300"> |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- GPU recommended (NVIDIA CUDA) for real-time video inference
- Webcam (optional, for live detection)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/CodingMantras/yolov8-streamlit-detection-tracking.git
cd yolov8-streamlit-detection-tracking

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

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
   - **Object Detection** — Standard YOLO26 object detection (NMS-free, end-to-end)
   - **Segmentation** — Instance segmentation with pixel masks
   - **YOLO World v2 (Text Prompt)** — Open-vocabulary detection with natural language prompts
   - **Pose Estimation** — Human body keypoint detection
3. **Model Confidence** — Adjust the confidence threshold (10–100%)

### Image Inference

1. Select **📷 Image Inference** mode
2. Choose a task (Detection, Segmentation, YOLO World, or Pose)
3. Upload an image or use the default
4. For **YOLO World v2**: type descriptive phrases (e.g., `person in black, red car, laptop on table`)
5. Click **🚀 Run** to see results with per-class metrics

### Video Inference

1. Select **🎬 Video Inference** mode
2. Choose a task
3. Pick a video source: **Stored Video**, **Webcam**, **RTSP**, or **YouTube**
4. **Object Tracking** is enabled by default (ByteTrack or BoTSORT) — local + global counts display automatically
5. Adjust **Skip Frames** (1–8) in the sidebar for faster inference on long videos
6. For **YOLO World v2**: enter natural language prompts to search for in the video
7. Click **🚀 Detect** — local and global metrics appear in the sidebar

### Adding Your Own Videos

Drop `.mp4` files into the `videos/` directory. They appear automatically in the stored-video dropdown — no code changes required (the config scans the folder at startup).

---

## 🗂️ Project Structure

```
yolov8-streamlit-detection-tracking/
├── app.py                # Main Streamlit application & routing
├── config.py             # Centralized configuration (paths, models, UI)
├── model_loader.py       # Model loading with @st.cache_resource
├── image_service.py      # Image inference (detection, segmentation, world, pose)
├── video_service.py      # Video inference (tracking, counting, all sources)
├── requirements.txt      # Python dependencies
├── packages.txt          # System packages for Streamlit Cloud
├── README.md
├── assets/               # Screenshots and demo media
├── images/               # Sample images
├── videos/               # Sample videos (add your .mp4 files here)
└── weights/              # Model weights (yolov8n.pt, yolov8n-seg.pt, ...)
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
DEFAULT_WORLD_CLASSES = "person in black, red car, dog, laptop on table"
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

This project is open-source and available for educational and research purposes.

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
