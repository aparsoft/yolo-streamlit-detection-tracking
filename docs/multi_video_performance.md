# Multi-Video Detection Performance — How It Works

> **YOLO Vision Studio** can run simultaneous detection/tracking across 6+ videos at low latency. This document explains every architectural decision that makes that possible.
>
> Scope: the `_run_multi_video_loop()` path (2+ videos selected). For the one-video path,
> and for the transport findings the two paths share, see
> [`single_video_performance.md`](single_video_performance.md).
>
> Numbers measured on an NVIDIA RTX PRO 4500 Blackwell, ultralytics 8.4.116,
> torch 2.8.0+cu128, `yolo26n.pt`, on the clips in `videos/`.

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

ByteTrack / BoTSORT tracking state (track history, Kalman filters) lives **on the model object**. If N videos shared one model, track IDs from video A would collide with those in video B. Instead, `load_fresh_model()` instantiates N independent model copies — each holding its own clean tracker state.

**This costs N× VRAM.** An earlier version of this document claimed PyTorch shares the weights between handles loaded from the same `.pt`. It does not — each `YOLO(path)` loads and uploads its own copy:

| Model | Per instance | 6 videos |
|---|---|---|
| `yolo26n.pt` | 10.6 MB | 63.7 MB |
| `yolo26l.pt` | 106.6 MB | 639.8 MB |
| `yolo26x.pt` | **241.3 MB** | **1447.8 MB** |

Isolated tracker state is still the right trade — colliding IDs would make the counts meaningless — but it is a real linear cost. Pick nano or small for 4+ simultaneous videos.

---

### 3. Frame Resize Before Inference — Never Larger Than Source

```python
# video_service.py  _display_size()
w = min(config.VIDEO_DISPLAY_WIDTH, w_orig)     # a ceiling, not a target
h = int(round(w * h_orig / w_orig))
```

Raw 1080p/4K frames are down-scaled before being passed to YOLO; YOLO's letterbox pipeline then resizes to its native grid. That part always worked, and `video_1.mp4` (3840×2160) genuinely benefits.

What did not work: `VIDEO_DISPLAY_WIDTH` used to be applied *unconditionally*. Every clip in `videos/` except two is 640×360, so this **upscaled** them 1.12× — 22.09 ms of inference instead of 17.26 ms, **+28% for pixels the camera never captured**. Interpolation cannot add detail a detector can use. It is a ceiling now.

---

### 4. Skip-Frames Slider — GPU Relief Only, and It Costs Accuracy

```python
# video_service.py  _run_multi_video_loop()
if frame_nums[i] % skip_frames != 0:
    continue      # leave the frame already on screen
```

The sidebar **`⏩ Skip Frames`** slider (1–8) controls how often inference runs. At `skip_frames=2` only every other frame is inferred, so with 6 videos you halve GPU pressure.

It used to re-display the previous annotated JPEG for every skipped frame. That re-registered the same bytes in Streamlit's media manager and queued another ForwardMsg to paint a picture already on screen — so the slider bought **zero** transport savings, which is most of what it needed to save. Skipped frames are simply not drawn now; the last one stays up, which is both correct and free.

Reach for **`VIDEO_DISPLAY_FPS`** before this slider: it relieves the same transport pressure without dropping detections. Skipping distorts unique-object counts in both directions — see §9.2 of [`yolo26_playground.ipynb`](yolo26_playground.ipynb).

---

### 5. Round-Robin Sequential Loop — No Threads

```python
# video_service.py  _run_multi_video_loop()
while any(active):
    for i in range(n):            # ← cycle through each video
        ok, frame = captures[i].read()
        ...
        annotated, ... = _process_frame(models[i], frame, ...)
        placeholders[i].image(...)
```

There are **no threads, no async workers**. A single Python loop cycles through every video in order. This is efficient, though not for the reason an earlier version of this document gave: `cv2.VideoCapture.read()` does **not** overlap with inference — it runs before it, on the same thread. Profiling one frame:

```
decode      0.23 ms      inference   10.86 ms      annotate  0.85 ms
resize      0.01 ms      overlay      0.26 ms      jpeg      0.62 ms
```

What actually keeps the loop cheap is that the per-frame CPU work is small next to GPU inference, and CUDA launches are asynchronous so the GPU stays fed within a single `track()` call. It also avoids GIL contention and inter-thread synchronisation entirely. Note that a decode worker thread — the obvious parallelisation — would buy ~0.23 ms of a 12.89 ms frame.

**Each video now keeps its own clock.** Sharing a single `prev_time` across the round-robin measured the gap between *consecutive videos'* frames, so every video reported the same FPS and it was n× the truth.

---

### 6. `st.empty()` Placeholders + JPEG Bytes — and Their Real Cost

```python
# video_service.py  _frame_to_bytes()
_, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.VIDEO_JPEG_QUALITY])
return buf.tobytes()

# video_service.py
if throttles[i].ready(now):
    placeholders[i].image(_frame_to_bytes(annotated), width="stretch")
```

The **`st.empty()` placeholder** part is true: created once before the loop, each `placeholder.image(bytes)` overwrites in place and Streamlit does a targeted DOM patch, not a full rerender.

The **raw JPEG bytes** part was wrong in two ways, and they were the app's biggest hidden costs:

- Passing bytes does **not** produce a `data:image/jpeg` blob. `st.image()` registers the payload in the global `MediaFileManager` and returns a `/media/<content-hash>.jpg` URL, which the browser fetches over a **separate HTTP GET**. Every frame is a unique hash, so every frame is a guaranteed cache miss. At 6 videos × 30 fps that is 180 GETs/s.
- Those registrations are collected only when a *script run ends* (`script_runner.py:906`) — and a playback loop is one long script run. Measured on a single 2809-frame clip: **183.8 MB retained**, versus 11.4 MB once `_gc_media_files()` runs inside the loop. N videos fill it N× faster.

Two mitigations, both in `config.py` and both shared with the single-video path:

- **`VIDEO_DISPLAY_FPS`** (default 30) caps the *paint* rate per video while every frame still goes through the detector. On one video: 775 images → 265, 44% fewer ForwardMsgs, **same 122 unique objects tracked**.
- **`MEDIA_GC_EVERY_N_FRAMES`** (default 30) collects the dead frame JPEGs mid-loop, at 0.07 ms/frame amortised.

Full derivation of both in [`single_video_performance.md`](single_video_performance.md).

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
