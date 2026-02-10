"""
Video inference service — Stored videos, Webcam, RTSP & YouTube.

Features
--------
* Object detection, segmentation, YOLO World v2 & pose estimation
* ByteTrack / BoTSORT tracking with per-box track IDs (class | conf | ID:N)
* Local (per-frame) + Global (cumulative) tracking metrics
* Skip-frames slider for faster inferencing
* Multi-video simultaneous detection in side-by-side columns
* Browser-based webcam via streamlit-webrtc
"""

from __future__ import annotations

import time
from collections import defaultdict

import cv2
import numpy as np
import streamlit as st
import yt_dlp

import config
from model_loader import get_model_for_task, load_fresh_model


# ── Track-ID colour palette (16 distinct BGR colours) ────────────────────────

_TRACK_COLORS = [
    (46, 204, 113),  # emerald
    (52, 152, 219),  # peter river
    (231, 76, 60),  # alizarin
    (241, 196, 15),  # sun flower
    (155, 89, 182),  # amethyst
    (26, 188, 156),  # turquoise
    (230, 126, 34),  # carrot
    (52, 73, 94),  # wet asphalt
    (22, 160, 133),  # green sea
    (39, 174, 96),  # nephritis
    (41, 128, 185),  # belize hole
    (142, 68, 173),  # wisteria
    (243, 156, 18),  # orange
    (211, 84, 0),  # pumpkin
    (192, 57, 43),  # pomegranate
    (127, 140, 141),  # asbestos
]


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    """Return a distinct BGR colour for *track_id*."""
    return _TRACK_COLORS[abs(track_id) % len(_TRACK_COLORS)]


# ── Custom annotation with track IDs on bounding boxes ───────────────────────


def _annotate_with_ids(
    frame: np.ndarray,
    result,
    enable_tracking: bool,
    font_scale: float = 0.50,
    box_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes with ``class | conf% | ID:N`` labels.

    For segmentation / pose tasks, masks / keypoints are rendered first
    via ``result.plot(labels=False, boxes=False)`` then custom box
    labels with track IDs are overlaid.
    """
    has_masks = getattr(result, "masks", None) is not None and len(result.masks)
    has_kpts = getattr(result, "keypoints", None) is not None and len(result.keypoints)

    if has_masks or has_kpts:
        annotated = result.plot(labels=False, boxes=False, conf=False)
    else:
        annotated = frame.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return annotated

    names = result.names
    boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    track_ids = None
    if enable_tracking and result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (box, cls_id, conf) in enumerate(zip(boxes_xyxy, classes, confs)):
        x1, y1, x2, y2 = box
        tid = track_ids[i] if track_ids is not None else None
        color = _color_for_track(tid) if tid is not None else _color_for_track(cls_id)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        # Label: "class | 87% | ID:5"
        name = names[cls_id]
        parts = [name, f"{conf:.0%}"]
        if tid is not None:
            parts.append(f"ID:{tid}")
        label = " | ".join(parts)

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(
            annotated,
            (x1, label_y - th - 4),
            (x1 + tw + 6, label_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y - 2),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


# ── Frame → JPEG bytes (avoids Streamlit MediaFileHandler cache issue) ───────


def _frame_to_bytes(frame: np.ndarray) -> bytes:
    """Encode a BGR *frame* to JPEG bytes for ``st.image()``.

    Sending raw bytes avoids Streamlit's internal temp-file caching,
    which can cause ``MediaFileStorageError`` during fast video loops.
    """
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


# ── Public API ────────────────────────────────────────────────────────────────


def render(task: str, confidence: float, selected_model: str | None = None) -> None:
    """Render the full video-inference page for the chosen *task*."""
    st.header(f"🎬 Video · {task}")

    # YOLO World text prompt
    world_classes: list[str] | None = None
    if task == config.TASK_WORLD:
        world_classes = _world_class_input()
        if not world_classes:
            return

    model = get_model_for_task(task, world_classes, model_name=selected_model)
    if model is None:
        return

    # Video source
    source = st.sidebar.radio("📹 Video Source", config.VIDEO_SOURCES, key="vid_source")

    # Tracking options (enabled by default)
    enable_tracking, tracker = _tracker_options()

    # Skip frames slider for faster inference
    skip_frames = st.sidebar.slider(
        "⏩ Skip Frames",
        min_value=config.MIN_SKIP_FRAMES,
        max_value=config.MAX_SKIP_FRAMES,
        value=config.DEFAULT_SKIP_FRAMES,
        help="Process every Nth frame. Higher = faster but less smooth.",
        key="skip_frames",
    )

    # Dispatch — pass task, world_classes & selected_model for multi-video isolation
    _SOURCE_HANDLERS[source](
        model,
        confidence,
        enable_tracking,
        tracker,
        skip_frames,
        task,
        world_classes,
        selected_model,
    )


# ── YOLO World helpers ───────────────────────────────────────────────────────


def _world_class_input() -> list[str] | None:
    st.markdown(
        "💡 **Tip**: YOLO World v2 supports natural language! "
        "Try `person in black`, `red car`, `man with backpack`."
    )
    text = st.text_area(
        "🔍 Enter object classes or descriptions to search in video (comma-separated)",
        value=config.DEFAULT_WORLD_CLASSES,
        help="YOLO World v2 will search for these objects/descriptions in every frame.",
    )
    classes = [c.strip() for c in text.split(",") if c.strip()]
    if classes:
        st.info(f"🎯 Searching: **{', '.join(classes)}**")
    else:
        st.warning("⚠️ Enter at least one class.")
    return classes or None


# ── Tracking config ──────────────────────────────────────────────────────────


def _tracker_options() -> tuple[bool, str | None]:
    """Sidebar widgets for tracker selection."""
    enable = st.sidebar.checkbox("Enable Object Tracking", value=True)
    tracker = None
    if enable:
        tracker = st.sidebar.radio(
            "Tracker Algorithm",
            config.TRACKERS_LIST,
            key="tracker_algo",
        )
    return enable, tracker


# ── Frame processor ──────────────────────────────────────────────────────────


def _process_frame(
    model,
    frame: np.ndarray,
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
    tracked_ids: set[int],
    class_tracked: dict[str, set[int]],
) -> tuple[np.ndarray, int, dict[str, int]]:
    """Run inference on a single frame.

    Returns ``(annotated_frame, object_count, per_class_counts)``.
    """
    h_orig, w_orig = frame.shape[:2]
    w = config.VIDEO_DISPLAY_WIDTH
    h = int(w * h_orig / w_orig)
    frame = cv2.resize(frame, (w, h))

    if enable_tracking and tracker:
        results = model.track(frame, conf=confidence, persist=True, tracker=tracker)
    else:
        results = model.predict(frame, conf=confidence)

    result = results[0]
    frame_obj_count = 0
    frame_class_counts: dict[str, int] = {}

    if result.boxes is not None and len(result.boxes):
        names = result.names
        classes = result.boxes.cls.cpu().numpy()
        frame_obj_count = len(classes)

        for cls_id in classes:
            name = names[int(cls_id)]
            frame_class_counts[name] = frame_class_counts.get(name, 0) + 1

        # Accumulate unique tracked IDs
        if enable_tracking and result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy()
            for track_id, cls_id in zip(ids, classes):
                tracked_ids.add(int(track_id))
                name = names[int(cls_id)]
                class_tracked.setdefault(name, set()).add(int(track_id))

    # Custom annotation with track IDs on bounding boxes
    annotated = _annotate_with_ids(frame, result, enable_tracking)

    # Overlay local + global counts
    annotated = _draw_overlay(
        annotated,
        frame_obj_count,
        frame_class_counts,
        len(tracked_ids) if enable_tracking else None,
        class_tracked if enable_tracking else None,
    )
    return annotated, frame_obj_count, frame_class_counts


def _draw_overlay(
    frame: np.ndarray,
    total: int,
    class_counts: dict[str, int],
    tracked_total: int | None = None,
    class_tracked: dict[str, set[int]] | None = None,
) -> np.ndarray:
    """Draw local (per-frame) + global (cumulative) tracking overlay."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.45, 1
    y_offset = 5
    line_h = 20
    pad = 10

    lines: list[str] = []

    # ── Local (this frame) ────────────────────────────────
    local_parts = [f"In Frame: {total}"]
    for name, cnt in list(class_counts.items())[:5]:
        local_parts.append(f"{name}: {cnt}")
    lines.append(" | ".join(local_parts))

    # ── Global (cumulative tracked) ───────────────────────
    if tracked_total is not None:
        global_parts = [f"Total Tracked: {tracked_total}"]
        if class_tracked:
            for name, ids in list(class_tracked.items())[:5]:
                global_parts.append(f"{name}: {len(ids)}")
        lines.append(" | ".join(global_parts))

    # Compute box size
    max_tw = 0
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        max_tw = max(max_tw, tw)

    box_h = y_offset + line_h * len(lines) + pad
    box_w = max_tw + 2 * pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (box_w + 5, box_h + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        color = (0, 255, 0) if i == 0 else (0, 200, 255)  # green local, yellow global
        cv2.putText(
            frame,
            line,
            (pad, y_offset + line_h * (i + 1)),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return frame


# ── Sidebar live metrics ─────────────────────────────────────────────────────


class _LiveMetrics:
    """Manages sidebar placeholder widgets that update each frame."""

    def __init__(self, enable_tracking: bool):
        self.container = st.sidebar.container()
        self.enable_tracking = enable_tracking
        with self.container:
            st.subheader("📈 Live Metrics")
            self._frame_ph = st.empty()
            self._fps_ph = st.empty()
            st.markdown("**🟢 Local (this frame)**")
            self._objects_ph = st.empty()
            self._classes_ph = st.empty()
            if enable_tracking:
                st.markdown("**🟡 Global (cumulative)**")
                self._tracked_ph = st.empty()
                self._global_classes_ph = st.empty()

    def update(
        self,
        frame_num: int,
        frame_obj_count: int,
        frame_class_counts: dict[str, int],
        tracked_total: int,
        class_tracked: dict[str, set[int]],
        fps: float,
    ):
        self._frame_ph.metric("Frame", frame_num)
        self._fps_ph.metric("FPS", f"{fps:.1f}")
        self._objects_ph.metric("Objects in Frame", frame_obj_count)
        local_str = " · ".join(f"**{k}**: {v}" for k, v in frame_class_counts.items())
        self._classes_ph.markdown(local_str or "—")
        if self.enable_tracking:
            self._tracked_ph.metric("Total Unique Objects", tracked_total)
            global_str = " · ".join(
                f"**{k}**: {len(ids)}" for k, ids in class_tracked.items()
            )
            self._global_classes_ph.markdown(global_str or "—")


# ── Single-video capture loop ────────────────────────────────────────────────


def _run_video_loop(
    vid_cap: cv2.VideoCapture,
    model,
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
    skip_frames: int = 1,
) -> None:
    """Common processing loop for any ``cv2.VideoCapture`` source."""
    if not vid_cap.isOpened():
        st.error("❌ Could not open video source.")
        return

    metrics = _LiveMetrics(enable_tracking)
    st_frame = st.empty()
    tracked_ids: set[int] = set()
    class_tracked: dict[str, set[int]] = defaultdict(set)
    frame_num = 0
    processed = 0
    prev_time = time.time()
    last_bytes: bytes | None = None

    try:
        while vid_cap.isOpened():
            ok, frame = vid_cap.read()
            if not ok:
                break
            frame_num += 1

            # Skip frames for faster inference
            if frame_num % skip_frames != 0:
                if last_bytes is not None:
                    st_frame.image(last_bytes, width="stretch")
                continue

            processed += 1

            annotated, obj_count, cls_counts = _process_frame(
                model,
                frame,
                confidence,
                enable_tracking,
                tracker,
                tracked_ids,
                class_tracked,
            )

            last_bytes = _frame_to_bytes(annotated)
            st_frame.image(last_bytes, width="stretch")

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            metrics.update(
                frame_num,
                obj_count,
                cls_counts,
                len(tracked_ids),
                class_tracked,
                fps,
            )
    finally:
        vid_cap.release()

    # Final summary
    skipped = frame_num - processed
    summary_parts = [f"**{frame_num}** frames read", f"**{processed}** processed"]
    if skipped:
        summary_parts.append(f"**{skipped}** skipped")

    if enable_tracking and tracked_ids:
        st.success(
            f"✅ {' · '.join(summary_parts)} — "
            f"**{len(tracked_ids)}** unique objects tracked"
        )
        with st.expander("📊 Tracking Summary", expanded=True):
            cols = st.columns(min(len(class_tracked), 4) or 1)
            for idx, (name, ids) in enumerate(class_tracked.items()):
                cols[idx % len(cols)].metric(name.capitalize(), len(ids))
    else:
        st.success(f"✅ {' · '.join(summary_parts)}")


# ── Multi-video simultaneous loop ────────────────────────────────────────────


def _run_multi_video_loop(
    vid_names: list[str],
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
    skip_frames: int,
    task: str,
    world_classes: list[str] | None,
    selected_model: str | None = None,
) -> None:
    """Process multiple videos simultaneously in side-by-side columns.

    Each video gets a **fresh model** instance so that ByteTrack /
    BoTSORT tracking state is isolated per video.
    """
    n = len(vid_names)

    # Fresh model per video — tracking state isolation
    models = [
        load_fresh_model(task, world_classes, model_name=selected_model)
        for _ in range(n)
    ]

    cols = st.columns(n)
    placeholders = []
    for i, name in enumerate(vid_names):
        with cols[i]:
            st.markdown(f"**{name}**")
            placeholders.append(st.empty())

    captures = [cv2.VideoCapture(str(config.VIDEOS_DICT[nm])) for nm in vid_names]
    tracked_sets: list[set[int]] = [set() for _ in range(n)]
    class_tracked_dicts: list[dict[str, set[int]]] = [
        defaultdict(set) for _ in range(n)
    ]
    frame_nums = [0] * n
    last_bytes_list: list[bytes | None] = [None] * n
    active = [cap.isOpened() for cap in captures]

    # Sidebar compact metrics
    with st.sidebar:
        st.subheader("📈 Multi-Video Metrics")
        metric_phs = [st.empty() for _ in vid_names]

    prev_time = time.time()

    try:
        while any(active):
            for i in range(n):
                if not active[i]:
                    continue

                ok, frame = captures[i].read()
                if not ok:
                    active[i] = False
                    continue

                frame_nums[i] += 1

                if frame_nums[i] % skip_frames != 0:
                    if last_bytes_list[i] is not None:
                        placeholders[i].image(
                            last_bytes_list[i],
                            width="stretch",
                        )
                    continue

                annotated, obj_count, cls_counts = _process_frame(
                    models[i],
                    frame,
                    confidence,
                    enable_tracking,
                    tracker,
                    tracked_sets[i],
                    class_tracked_dicts[i],
                )

                last_bytes_list[i] = _frame_to_bytes(annotated)
                placeholders[i].image(last_bytes_list[i], width="stretch")

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                metric_phs[i].markdown(
                    f"**{vid_names[i]}** — Frame {frame_nums[i]} · "
                    f"{obj_count} obj · {len(tracked_sets[i])} tracked · "
                    f"{fps:.1f} FPS"
                )
    finally:
        for cap in captures:
            cap.release()

    # Per-video summary
    for i, name in enumerate(vid_names):
        t = tracked_sets[i]
        st.success(
            f"✅ **{name}**: {frame_nums[i]} frames"
            + (f" — **{len(t)}** unique objects" if t else "")
        )


# ── Source handlers ──────────────────────────────────────────────────────────


def _play_stored_video(
    model,
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
    skip_frames: int,
    task: str,
    world_classes: list[str] | None,
    selected_model: str | None = None,
) -> None:
    if not config.VIDEOS_DICT:
        st.warning("No videos found in the `videos/` directory.")
        return

    vid_names = st.sidebar.multiselect(
        "Choose video(s)",
        list(config.VIDEOS_DICT.keys()),
        default=[list(config.VIDEOS_DICT.keys())[0]] if config.VIDEOS_DICT else [],
        help="Select multiple videos for simultaneous detection.",
    )

    if not vid_names:
        st.info("Select at least one video from the sidebar.")
        return

    # Preview selected videos
    preview_cols = st.columns(min(len(vid_names), 3))
    for i, name in enumerate(vid_names):
        with preview_cols[i % len(preview_cols)]:
            with open(config.VIDEOS_DICT[name], "rb") as f:
                st.video(f.read())

    if st.sidebar.button("🚀 Detect Video Objects", type="primary"):
        if len(vid_names) == 1:
            _run_video_loop(
                cv2.VideoCapture(str(config.VIDEOS_DICT[vid_names[0]])),
                model,
                confidence,
                enable_tracking,
                tracker,
                skip_frames,
            )
        else:
            _run_multi_video_loop(
                vid_names,
                confidence,
                enable_tracking,
                tracker,
                skip_frames,
                task,
                world_classes,
                selected_model,
            )


def _play_webcam(
    model,
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
    skip_frames: int,
    task: str,
    world_classes: list[str] | None,
    selected_model: str | None = None,
) -> None:
    """Browser-based webcam via streamlit-webrtc (works locally + cloud)."""
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
        import av
    except ImportError:
        st.error(
            "❌ `streamlit-webrtc` is required for webcam access. "
            "Install it with: `pip install streamlit-webrtc`"
        )
        return

    st.info(
        "📷 Click **START** below to activate your webcam. "
        "Your browser will ask for camera permission — please allow it."
    )

    tracked_ids_global: set[int] = set()
    class_tracked_global: dict[str, set[int]] = defaultdict(set)

    class YOLOVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.frame_count = 0
            self.last_annotated = None

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            # Skip frames
            if self.frame_count % skip_frames != 0:
                if self.last_annotated is not None:
                    return av.VideoFrame.from_ndarray(
                        self.last_annotated, format="bgr24"
                    )
                return frame

            h_orig, w_orig = img.shape[:2]
            w = config.VIDEO_DISPLAY_WIDTH
            h = int(w * h_orig / w_orig)
            img = cv2.resize(img, (w, h))

            if enable_tracking and tracker:
                results = model.track(
                    img, conf=confidence, persist=True, tracker=tracker
                )
            else:
                results = model.predict(img, conf=confidence)

            result = results[0]
            frame_class_counts: dict[str, int] = {}

            if result.boxes is not None and len(result.boxes):
                names = result.names
                classes = result.boxes.cls.cpu().numpy()

                for cls_id in classes:
                    name = names[int(cls_id)]
                    frame_class_counts[name] = frame_class_counts.get(name, 0) + 1

                if enable_tracking and result.boxes.id is not None:
                    ids = result.boxes.id.cpu().numpy()
                    for track_id, cls_id in zip(ids, classes):
                        tracked_ids_global.add(int(track_id))
                        name = names[int(cls_id)]
                        class_tracked_global.setdefault(name, set()).add(int(track_id))

            annotated = _annotate_with_ids(img, result, enable_tracking)
            annotated = _draw_overlay(
                annotated,
                len(result.boxes) if result.boxes is not None else 0,
                frame_class_counts,
                len(tracked_ids_global) if enable_tracking else None,
                class_tracked_global if enable_tracking else None,
            )
            self.last_annotated = annotated
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="yolo-webcam",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def _play_rtsp(
    model,
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
    skip_frames: int,
    task: str,
    world_classes: list[str] | None,
    selected_model: str | None = None,
) -> None:
    url = st.sidebar.text_input(
        "RTSP Stream URL",
        placeholder="rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101",
    )
    if st.sidebar.button("🚀 Start RTSP Stream", type="primary"):
        if not url:
            st.sidebar.error("Please enter an RTSP URL.")
            return
        _run_video_loop(
            cv2.VideoCapture(url),
            model,
            confidence,
            enable_tracking,
            tracker,
            skip_frames,
        )


def _play_youtube(
    model,
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
    skip_frames: int,
    task: str,
    world_classes: list[str] | None,
    selected_model: str | None = None,
) -> None:
    url = st.sidebar.text_input(
        "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
    )
    if st.sidebar.button("🚀 Detect YouTube Video", type="primary"):
        if not url:
            st.sidebar.error("Please enter a YouTube URL.")
            return
        try:
            with st.sidebar:
                with st.spinner("Extracting stream URL…"):
                    stream_url = _get_youtube_stream(url)
            _run_video_loop(
                cv2.VideoCapture(stream_url),
                model,
                confidence,
                enable_tracking,
                tracker,
                skip_frames,
            )
        except Exception as exc:
            st.sidebar.error(f"YouTube error: {exc}")


def _get_youtube_stream(youtube_url: str) -> str:
    ydl_opts = {"format": "best[ext=mp4]", "no_warnings": True, "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


# ── Handler dispatch table ───────────────────────────────────────────────────
_SOURCE_HANDLERS = {
    config.SOURCE_STORED: _play_stored_video,
    config.SOURCE_WEBCAM: _play_webcam,
    config.SOURCE_RTSP: _play_rtsp,
    config.SOURCE_YOUTUBE: _play_youtube,
}
