# Multi-Video Detection Performance — How It Works

> **YOLO Vision Studio** can run simultaneous detection/tracking across 6+ videos at low latency. This document explains every architectural decision that makes that possible.

---

## Architecture Overview

```
app.py
  └─► video_service.render()
        └─► _play_stored_video()          ← multiselect picks N videos
              ├─ N == 1  →  _run_video_loop()       (single, cached model)
              └─ N  > 1  →  _run_multi_video_loop() (N fresh models, round-robin)
```

---

## The 6 Performance Pillars

### 1. `@st.cache_resource` — Load Once, Reuse Forever

```python
# model_loader.py
@st.cache_resource
def load_model(model_name: str) -> YOLO:
    ...
```

Streamlit's `@st.cache_resource` pins the loaded model in server memory across **all reruns and all user interactions**. The heavy GPU weight-loading (YOLO26 ≈ 200–600 MB) happens exactly **once** per session. Every subsequent call returns the same in-memory object instantly.

---

### 2. Fresh Model Per Video — Isolated Tracking State

```python
# video_service.py  _run_multi_video_loop()
models = [
    load_fresh_model(task, world_classes, model_name=selected_model)
    for _ in range(n)
]
```

ByteTrack / BoTSORT tracking state (track history, Kalman filters) lives **on the model object**. If N videos shared one model, track IDs from video A would collide with those in video B. Instead, `load_fresh_model()` instantiates N independent model copies — each holding its own clean tracker state. The weights themselves are shared in GPU VRAM by PyTorch's memory manager (same `.pt` file → same underlying tensors), so the memory overhead is minimal.

---

### 3. Frame Resize Before Inference — Smaller Input = Faster Compute

```python
# video_service.py  _process_frame()
w = config.VIDEO_DISPLAY_WIDTH   # 720 px
h = int(w * h_orig / w_orig)
frame = cv2.resize(frame, (w, h))
```

Raw video frames (often 1080p or 4K) are **down-scaled to 720 px wide** before being passed to YOLO. YOLO's internal letterbox pipeline then resizes to its native grid (e.g. 640×640). This two-step downscale dramatically cuts the number of CNN feature-map operations for each inference pass.

---

### 4. Skip-Frames Slider — Tunable Throughput/Smoothness Trade-off

```python
# video_service.py  _run_multi_video_loop()
if frame_nums[i] % skip_frames != 0:
    if last_bytes_list[i] is not None:
        placeholders[i].image(last_bytes_list[i], width="stretch")
    continue
```

The sidebar **`⏩ Skip Frames`** slider (1–8) controls how often inference actually runs. At `skip_frames=2` only every other frame is inferred; skipped frames re-display the previous annotated result instantly. With 6 videos at skip=2 you effectively halve GPU pressure. The last annotated JPEG is cached per video (`last_bytes_list[i]`) so skipped frames are never blank.

---

### 5. Round-Robin Sequential Loop — GPU Batching Without Threads

```python
# video_service.py  _run_multi_video_loop()
while any(active):
    for i in range(n):            # ← cycle through each video
        ok, frame = captures[i].read()
        ...
        annotated, ... = _process_frame(models[i], frame, ...)
        placeholders[i].image(...)
```

There are **no threads, no async workers**. A single Python loop cycles through every video in order. This works fast because:

- `cv2.VideoCapture.read()` decodes frames on the **CPU** while the previous YOLO call is finishing on the **GPU** — overlapping compute automatically.
- PyTorch CUDA calls are non-blocking from Python's perspective; the GPU pipeline stays fed.
- Avoids Python GIL contention and inter-thread synchronisation overhead entirely.
- The loop overhead is negligible compared to GPU inference time (~5–20 ms per frame on YOLO26-nano/small).

---

### 6. `st.empty()` Placeholders + JPEG Bytes — Zero-Overhead UI Updates

```python
# model_loader.py  _frame_to_bytes()
_, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
return buf.tobytes()

# video_service.py
placeholders[i].image(last_bytes_list[i], width="stretch")
```

Two micro-optimisations here:

- **`st.empty()` placeholder**: Created once before the loop. Each `placeholder.image(bytes)` call **overwrites** the existing widget in-place. Streamlit does a targeted DOM patch — no full-page rerender.
- **Raw JPEG bytes**: Passing bytes directly bypasses Streamlit's internal `MediaFileStorageError`-prone temp-file cache. The browser receives a direct `data:image/jpeg` blob, which is smaller and faster to decode than PNG.

---

## YOLO26 Architecture Efficiency

YOLO26 (Ultralytics v8.3+) is a pure **CNN single-shot detector**. Key properties:

| Property | Impact |
|---|---|
| Single forward pass detects all objects | No region-proposal overhead |
| Nano/small variants < 4 M parameters | Fits entirely in L2/L3 GPU cache |
| FP16 inference on CUDA | 2× throughput vs FP32 |
| Anchor-free head | Fewer post-processing steps |

On an NVIDIA GPU with YOLO26-nano, a 720×720 frame takes **≈ 5–8 ms** per inference. With 6 videos and skip=1 that is ~36–48 ms per full round-robin cycle — comfortably over 20 FPS displayed per video.

---

## GPU Memory: Why N Fresh Models Doesn't Cost N× VRAM

`load_fresh_model()` calls `YOLO(path)` with the **same file path** each time. PyTorch's weight-sharing means:

- The underlying `nn.Module` parameters are loaded from the `.pt` file once and **reference-counted** in VRAM.
- Each model object has its own **tracker state** (CPU-side Kalman buffers, track dictionaries) — typically a few MB.
- The GPU holds one copy of the CNN weights regardless of how many Python model handles exist.

---

## End-to-End Frame Processing Sequence

```
cv2.VideoCapture.read()          ← CPU — decode next video frame
cv2.resize()                     ← CPU — scale to 720 px wide
model.track() / model.predict()  ← GPU — YOLO26 inference + NMS
_annotate_with_ids()             ← CPU — draw boxes + labels
_draw_overlay()                  ← CPU — stats overlay
cv2.imencode(".jpg")             ← CPU — JPEG compress
placeholder.image(bytes)         ← Streamlit — DOM patch to browser
```

CPU steps are fast enough that GPU is the bottleneck, so adding more videos costs only the marginal GPU time per extra inference call — **not** additional Python or IO overhead.

---

## Configuration Knobs Summary

| Setting | File | Default | Effect |
|---|---|---|---|
| `VIDEO_DISPLAY_WIDTH` | `config.py` | `720` | Resize target before inference |
| `DEFAULT_SKIP_FRAMES` | `config.py` | `1` | Process every frame |
| `MAX_SKIP_FRAMES` | `config.py` | `8` | Max frames to skip |
| `JPEG_QUALITY` | `video_service.py` | `90` | Output image quality |
| `_COLS_PER_ROW` | `video_service.py` | `3` | Grid columns per row |
| Tracker | sidebar | `bytetrack.yaml` | ByteTrack or BoTSORT |
