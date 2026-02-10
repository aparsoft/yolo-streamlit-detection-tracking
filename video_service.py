"""
Video inference service — Stored videos, Webcam, RTSP & YouTube.

Features
--------
* Object detection, segmentation, YOLO World & pose estimation
* ByteTrack / BoTSORT tracking with unique-object counting
* Real-time per-class & total-object metrics in the sidebar
* YOLO World text-prompt search in video streams
"""

from __future__ import annotations

import time
from collections import defaultdict

import cv2
import numpy as np
import streamlit as st
import yt_dlp

import config
from model_loader import get_model_for_task


# ── Public API ────────────────────────────────────────────────────────────────


def render(task: str, confidence: float) -> None:
    """Render the full video-inference page for the chosen *task*."""
    st.header(f"🎬 Video · {task}")

    # YOLO World text prompt
    world_classes: list[str] | None = None
    if task == config.TASK_WORLD:
        world_classes = _world_class_input()
        if not world_classes:
            return

    model = get_model_for_task(task, world_classes)
    if model is None:
        return

    # Video source
    source = st.sidebar.radio("📹 Video Source", config.VIDEO_SOURCES, key="vid_source")

    # Tracking options
    enable_tracking, tracker = _tracker_options()

    # Dispatch
    _SOURCE_HANDLERS[source](model, confidence, enable_tracking, tracker)


# ── YOLO World helpers ───────────────────────────────────────────────────────


def _world_class_input() -> list[str] | None:
    text = st.text_area(
        "🔍 Enter object classes to search in video (comma-separated)",
        value=config.DEFAULT_WORLD_CLASSES,
        help="YOLOE-26 will search for these objects in every frame.",
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

    Returns
    -------
    annotated : np.ndarray
        Frame with drawn annotations.
    frame_obj_count : int
        Number of objects detected in this frame.
    frame_class_counts : dict[str, int]
        Per-class counts for this frame.
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

    annotated = result.plot()

    # Overlay count badge
    annotated = _draw_overlay(
        annotated,
        frame_obj_count,
        frame_class_counts,
        len(tracked_ids) if enable_tracking else None,
    )
    return annotated, frame_obj_count, frame_class_counts


def _draw_overlay(
    frame: np.ndarray,
    total: int,
    class_counts: dict[str, int],
    tracked_total: int | None = None,
) -> np.ndarray:
    """Draw a compact count overlay at the top-left of the frame."""
    parts = [f"Objects: {total}"]
    for name, cnt in list(class_counts.items())[:5]:
        parts.append(f"{name}: {cnt}")
    if tracked_total is not None:
        parts.append(f"Unique tracked: {tracked_total}")
    text = " | ".join(parts)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.50, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (tw + 15, th + baseline + 15), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame, text, (10, th + 10), font, scale, (0, 255, 0), thickness, cv2.LINE_AA
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
            self._objects_ph = st.empty()
            self._tracked_ph = st.empty()
            self._classes_ph = st.empty()
            self._fps_ph = st.empty()

    def update(
        self,
        frame_num: int,
        frame_obj_count: int,
        frame_class_counts: dict[str, int],
        tracked_total: int,
        fps: float,
    ):
        self._frame_ph.metric("Frame", frame_num)
        self._objects_ph.metric("Objects in Frame", frame_obj_count)
        if self.enable_tracking:
            self._tracked_ph.metric("Total Unique Objects", tracked_total)
        class_str = " · ".join(f"**{k}**: {v}" for k, v in frame_class_counts.items())
        self._classes_ph.markdown(class_str or "—")
        self._fps_ph.metric("FPS", f"{fps:.1f}")


# ── Video-capture loop (shared by all sources) ───────────────────────────────


def _run_video_loop(
    vid_cap: cv2.VideoCapture,
    model,
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
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
    prev_time = time.time()

    try:
        while vid_cap.isOpened():
            ok, frame = vid_cap.read()
            if not ok:
                break
            frame_num += 1

            annotated, obj_count, cls_counts = _process_frame(
                model,
                frame,
                confidence,
                enable_tracking,
                tracker,
                tracked_ids,
                class_tracked,
            )

            st_frame.image(annotated, channels="BGR", use_container_width=True)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            metrics.update(frame_num, obj_count, cls_counts, len(tracked_ids), fps)
    finally:
        vid_cap.release()

    # Final summary
    if enable_tracking and tracked_ids:
        st.success(
            f"✅ Processed **{frame_num}** frames — "
            f"**{len(tracked_ids)}** unique objects tracked"
        )
        with st.expander("📊 Tracking Summary", expanded=True):
            for name, ids in class_tracked.items():
                st.metric(name.capitalize(), len(ids))


# ── Source handlers ──────────────────────────────────────────────────────────


def _play_stored_video(
    model, confidence: float, enable_tracking: bool, tracker: str | None
) -> None:
    if not config.VIDEOS_DICT:
        st.warning("No videos found in the `videos/` directory.")
        return

    vid_name = st.sidebar.selectbox("Choose a video", list(config.VIDEOS_DICT.keys()))
    vid_path = config.VIDEOS_DICT[vid_name]

    with open(vid_path, "rb") as f:
        st.video(f.read())

    if st.sidebar.button("🚀 Detect Video Objects", type="primary"):
        _run_video_loop(
            cv2.VideoCapture(str(vid_path)),
            model,
            confidence,
            enable_tracking,
            tracker,
        )


def _play_webcam(
    model, confidence: float, enable_tracking: bool, tracker: str | None
) -> None:
    if st.sidebar.button("🚀 Start Webcam", type="primary"):
        _run_video_loop(
            cv2.VideoCapture(config.WEBCAM_PATH),
            model,
            confidence,
            enable_tracking,
            tracker,
        )


def _play_rtsp(
    model, confidence: float, enable_tracking: bool, tracker: str | None
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
        )


def _play_youtube(
    model, confidence: float, enable_tracking: bool, tracker: str | None
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
