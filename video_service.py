"""
Video inference service — Stored videos, Webcam, RTSP & YouTube.

Features
--------
* Object detection, segmentation, YOLO World v2 & pose estimation
* ByteTrack / BoTSORT tracking with unique-object counting (enabled by default)
* Local (per-frame) + Global (cumulative) tracking metrics
* Skip-frames slider for faster inferencing of similar frames
* Real-time per-class & total-object metrics in the sidebar
* YOLO World v2 natural language text search in video streams
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

    # Tracking options (enabled by default)
    enable_tracking, tracker = _tracker_options()

    # Skip frames slider for faster inference
    skip_frames = st.sidebar.slider(
        "⏩ Skip Frames",
        min_value=config.MIN_SKIP_FRAMES,
        max_value=config.MAX_SKIP_FRAMES,
        value=config.DEFAULT_SKIP_FRAMES,
        help="Process every Nth frame. Higher = faster but less smooth. 1 = every frame.",
        key="skip_frames",
    )

    # Dispatch
    _SOURCE_HANDLERS[source](model, confidence, enable_tracking, tracker, skip_frames)


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

    # Overlay local + global tracking counts
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


# ── Video-capture loop (shared by all sources) ───────────────────────────────


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
    last_annotated = None

    try:
        while vid_cap.isOpened():
            ok, frame = vid_cap.read()
            if not ok:
                break
            frame_num += 1

            # Skip frames for faster inference
            if frame_num % skip_frames != 0:
                # Show last annotated frame if available (keeps display smooth)
                if last_annotated is not None:
                    st_frame.image(
                        last_annotated, channels="BGR", use_container_width=True
                    )
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

            last_annotated = annotated
            st_frame.image(annotated, channels="BGR", use_container_width=True)

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


# ── Source handlers ──────────────────────────────────────────────────────────


def _play_stored_video(
    model,
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
    skip_frames: int = 1,
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
            skip_frames,
        )


def _play_webcam(
    model,
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
    skip_frames: int = 1,
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

    # We need to create a processor class that captures detection state
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

            annotated = result.plot()
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
    skip_frames: int = 1,
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
    skip_frames: int = 1,
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
