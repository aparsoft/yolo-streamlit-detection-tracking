"""
Video inference service — Stored videos, Webcam, RTSP & YouTube.

Features
--------
* Object detection, segmentation, YOLO World v2 & pose estimation
* ByteTrack / BoT-SORT / Deep OC-SORT / TrackTrack with per-box IDs (class | conf | ID:N)
* Optional appearance ReID (BoT-SORT family) with sidebar-tunable gates
* Local (per-frame) + Global (confirmed-only) counts + track-quality metrics
* Skip-frames slider for faster inferencing
* Multi-video simultaneous detection in side-by-side columns
* Browser-based webcam via streamlit-webrtc
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import yaml
import yt_dlp
from ultralytics.utils import YAML

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

    # YOLO World / YOLOE text prompt
    world_classes: list[str] | None = None
    if task == config.TASK_WORLD:
        world_classes = _world_class_input()
        if not world_classes:
            return
    elif task == config.TASK_YOLOE:
        world_classes = _yoloe_class_input()
        if not world_classes:
            return

    model = get_model_for_task(task, world_classes, model_name=selected_model)
    if model is None:
        return

    # Video source
    source = st.sidebar.radio("📹 Video Source", config.VIDEO_SOURCES, key="vid_source")

    # Tracking options (enabled by default)
    enable_tracking, tracker = _tracker_options()

    # Trackers are built once and cached along with the model, so a changed tracker or
    # a changed ReID setting only takes effect if we drop the existing ones.
    if st.session_state.get("_active_tracker") != tracker:
        _reset_trackers(model)
        st.session_state["_active_tracker"] = tracker

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


# ── YOLOE helpers ────────────────────────────────────────────────────────────


def _yoloe_class_input() -> list[str] | None:
    """Show a text-area for category-level object classes (YOLOE)."""
    st.markdown(
        "💡 **Tip**: YOLOE supports **category-level** labels like `person`, `car`, `dog`. "
        "It does **NOT** support descriptive phrases like *person in red shirt*. "
        "Results include both bounding boxes **and** segmentation masks."
    )
    text = st.text_area(
        "🔍 Enter object categories to detect & segment in video (comma-separated)",
        value=config.DEFAULT_YOLOE_CLASSES,
        help="YOLOE will search for these object categories in every frame and produce segmentation masks.",
    )
    classes = [c.strip() for c in text.split(",") if c.strip()]
    if classes:
        st.info(f"🎯 Detecting & segmenting: **{', '.join(classes)}**")
    else:
        st.warning("⚠️ Enter at least one object category.")
    return classes or None


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


def _reset_trackers(model) -> None:
    """Force Ultralytics to rebuild trackers from the YAML on the next ``track()`` call.

    Tracker objects are created once in ``on_predict_start`` and reused for as long as
    ``persist=True``, so a changed YAML — or a different tracker entirely — is otherwise
    silently ignored. Our models are ``@st.cache_resource``d, so they survive every
    Streamlit rerun and would keep the tracker chosen on the very first run forever.
    """
    predictor = getattr(model, "predictor", None)
    if predictor is not None and hasattr(predictor, "trackers"):
        del predictor.trackers


def _tracker_defaults(tracker_yaml: str) -> dict:
    """Load the packaged defaults for a tracker (``bytetrack.yaml``, ``botsort.yaml``…)."""
    import ultralytics

    return YAML.load(
        Path(ultralytics.__file__).parent / "cfg" / "trackers" / tracker_yaml
    )


def _build_tracker_yaml(tracker_yaml: str, overrides: dict) -> str:
    """Write a tracker config with *overrides* applied and return its path.

    Ultralytics accepts a path anywhere it accepts a tracker name, so sidebar settings
    become a real YAML file. The filename hashes the settings: identical settings reuse
    one file, and a changed setting produces a new path — which is also how
    ``_reset_trackers`` knows something changed.
    """
    cfg = _tracker_defaults(tracker_yaml)
    cfg.update(overrides)
    key = hashlib.md5(repr(sorted(cfg.items())).encode()).hexdigest()[:8]
    path = Path(tempfile.gettempdir()) / f"yolo_studio_{cfg['tracker_type']}_{key}.yaml"
    if not path.exists():
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(path)


def _tracker_options() -> tuple[bool, str | None]:
    """Sidebar widgets for tracker selection, ReID and its thresholds.

    Returns ``(enable_tracking, tracker)`` where *tracker* is either a packaged name
    (``"bytetrack.yaml"``) or the path of a generated config — ``model.track()``
    takes both.
    """
    enable = st.sidebar.checkbox("Enable Object Tracking", value=True)
    if not enable:
        return False, None

    tracker = st.sidebar.radio(
        "Tracker Algorithm",
        config.TRACKERS_LIST,
        key="tracker_algo",
    )

    overrides: dict = {}
    if tracker != config.TRACKER_BYTETRACK:
        # Camera-motion compensation costs ~10 ms/frame and buys nothing on fixed cameras.
        overrides["gmc_method"] = config.GMC_METHOD

    if tracker in config.REID_CAPABLE_TRACKERS:
        if st.sidebar.checkbox(
            "🧬 Appearance ReID",
            value=False,
            key="use_reid",
            help="Match objects by how they look, not only by where they are. Recovers "
            "IDs through occlusion — and costs roughly two-thirds of your FPS.",
        ):
            encoder = st.sidebar.selectbox(
                "ReID encoder",
                list(config.REID_ENCODER_CHOICES),
                key="reid_encoder",
            )
            overrides |= {
                "with_reid": True,
                "model": config.REID_ENCODER_CHOICES[encoder],
                "proximity_thresh": st.sidebar.slider(
                    "Proximity gate (IoU)",
                    0.10,
                    0.90,
                    config.REID_PROXIMITY_THRESH,
                    0.05,
                    help="Lower lets appearance rescue boxes that moved further — better "
                    "through occlusion, but too low merges different objects.",
                ),
                "appearance_thresh": st.sidebar.slider(
                    "Appearance gate",
                    0.30,
                    0.95,
                    config.REID_APPEARANCE_THRESH,
                    0.05,
                    help="Lower accepts weaker look-alike matches.",
                ),
                "track_buffer": st.sidebar.slider(
                    "Lost-track memory (frames)",
                    30,
                    150,
                    config.REID_TRACK_BUFFER,
                    10,
                    help="How long a vanished track stays re-findable. Only useful "
                    "together with ReID.",
                ),
            }
            st.sidebar.caption(
                "⚠️ The local -cls encoder recovers occlusions but cannot reliably tell "
                "two similar people apart. Use a -reid encoder for that."
            )

    if not overrides:
        return True, tracker
    return True, _build_tracker_yaml(tracker, overrides)


# ── Frame processor ──────────────────────────────────────────────────────────


def _confirmed(track_hits: Counter, min_hits: int = config.MIN_TRACK_HITS) -> set[int]:
    """Track IDs seen at least *min_hits* times — the ones worth reporting.

    A tracker hands out an ID the moment it sees something; plenty of those live for a
    frame or two (a chair that briefly looked like a person). Counting every ID ever
    issued is how "unique objects" ends up two to three times the truth.
    """
    return {tid for tid, hits in track_hits.items() if hits >= min_hits}


def _track_quality(track_hits: Counter) -> tuple[float, int]:
    """Return ``(churn, stable_tracks)``.

    ``churn = max_id / unique_ids``. 1.0 is perfect. 2.5 means the tracker issued two
    and a half IDs for every object it actually followed — fragmentation, which is
    exactly what appearance ReID is there to fix. Watch it while tuning the sliders.
    """
    if not track_hits:
        return 0.0, 0
    return round(max(track_hits) / len(track_hits), 2), len(_confirmed(track_hits))


def _process_frame(
    model,
    frame: np.ndarray,
    confidence: float,
    enable_tracking: bool,
    tracker: str | None,
    track_hits: Counter,
    class_hits: dict[str, Counter],
) -> tuple[np.ndarray, int, dict[str, int]]:
    """Run inference on a single frame.

    *track_hits* and *class_hits* accumulate **how many frames each track ID was seen
    in**, not merely that it existed — see :func:`_confirmed`.

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

        # Accumulate per-ID frame counts
        if enable_tracking and result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy()
            for track_id, cls_id in zip(ids, classes):
                track_hits[int(track_id)] += 1
                name = names[int(cls_id)]
                class_hits.setdefault(name, Counter())[int(track_id)] += 1

    # Custom annotation with track IDs on bounding boxes
    annotated = _annotate_with_ids(frame, result, enable_tracking)

    # Overlay local + global counts
    annotated = _draw_overlay(
        annotated,
        frame_obj_count,
        frame_class_counts,
        len(_confirmed(track_hits)) if enable_tracking else None,
        class_hits if enable_tracking else None,
        _track_quality(track_hits) if enable_tracking else None,
    )
    return annotated, frame_obj_count, frame_class_counts


def _draw_overlay(
    frame: np.ndarray,
    total: int,
    class_counts: dict[str, int],
    tracked_total: int | None = None,
    class_hits: dict[str, Counter] | None = None,
    quality: tuple[float, int] | None = None,
) -> np.ndarray:
    """Draw local (per-frame) + global (cumulative) + track-quality overlay."""
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

    # ── Global (cumulative tracked, confirmed only) ───────
    if tracked_total is not None:
        global_parts = [f"Total Tracked: {tracked_total}"]
        if class_hits:
            for name, hits in list(class_hits.items())[:5]:
                global_parts.append(f"{name}: {len(_confirmed(hits))}")
        lines.append(" | ".join(global_parts))

    # ── Track quality — is the tracker fragmenting? ───────
    if quality is not None:
        churn, stable = quality
        lines.append(
            f"Churn: {churn} | Stable: {stable} | Min hits: {config.MIN_TRACK_HITS}"
        )

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

    # green = local, yellow = global, grey = track quality
    _LINE_COLORS = [(0, 255, 0), (0, 200, 255), (200, 200, 200)]

    for i, line in enumerate(lines):
        color = _LINE_COLORS[min(i, len(_LINE_COLORS) - 1)]
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
                st.markdown("**⚪ Track quality**")
                self._quality_ph = st.empty()

    def update(
        self,
        frame_num: int,
        frame_obj_count: int,
        frame_class_counts: dict[str, int],
        track_hits: Counter,
        class_hits: dict[str, Counter],
        fps: float,
    ):
        self._frame_ph.metric("Frame", frame_num)
        self._fps_ph.metric("FPS", f"{fps:.1f}")
        self._objects_ph.metric("Objects in Frame", frame_obj_count)
        local_str = " · ".join(f"**{k}**: {v}" for k, v in frame_class_counts.items())
        self._classes_ph.markdown(local_str or "—")
        if self.enable_tracking:
            churn, stable = _track_quality(track_hits)
            self._tracked_ph.metric("Total Unique Objects", stable)
            global_str = " · ".join(
                f"**{k}**: {len(_confirmed(hits))}" for k, hits in class_hits.items()
            )
            self._global_classes_ph.markdown(global_str or "—")
            self._quality_ph.markdown(
                f"Churn **{churn}** · IDs issued **{len(track_hits)}** · "
                f"counted after **{config.MIN_TRACK_HITS}** frames"
            )


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
    track_hits: Counter = Counter()  # track_id -> frames seen
    class_hits: dict[str, Counter] = defaultdict(Counter)
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
                track_hits,
                class_hits,
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
                track_hits,
                class_hits,
                fps,
            )
    finally:
        vid_cap.release()
        # The next playback should start its IDs from 1, not continue this run's.
        _reset_trackers(model)

    # Final summary
    skipped = frame_num - processed
    summary_parts = [f"**{frame_num}** frames read", f"**{processed}** processed"]
    if skipped:
        summary_parts.append(f"**{skipped}** skipped")

    if enable_tracking and track_hits:
        churn, stable = _track_quality(track_hits)
        st.success(
            f"✅ {' · '.join(summary_parts)} — "
            f"**{stable}** unique objects tracked "
            f"(seen ≥{config.MIN_TRACK_HITS} frames)"
        )
        with st.expander("📊 Tracking Summary", expanded=True):
            cols = st.columns(min(len(class_hits), 4) or 1)
            for idx, (name, hits) in enumerate(class_hits.items()):
                cols[idx % len(cols)].metric(name.capitalize(), len(_confirmed(hits)))
            st.caption(
                f"{len(track_hits)} IDs issued, {stable} confirmed · churn {churn} "
                "(1.0 = every ID followed one object; higher means tracks broke and "
                "re-registered — try a ReID tracker)."
            )
    else:
        st.success(f"✅ {' · '.join(summary_parts)}")


# ── Multi-video simultaneous loop ────────────────────────────────────────────


def _run_multi_video_loop(
    vid_names: list[str],
    videos: dict[str, object],
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
    _COLS_PER_ROW = 3

    # Fresh model per video — tracking state isolation
    models = [
        load_fresh_model(task, world_classes, model_name=selected_model)
        for _ in range(n)
    ]

    # Build placeholders in a 3-per-row grid
    placeholders: list[st.delta_generator.DeltaGenerator] = []
    for row_start in range(0, n, _COLS_PER_ROW):
        row_names = vid_names[row_start : row_start + _COLS_PER_ROW]
        cols = st.columns(_COLS_PER_ROW)
        for j, name in enumerate(row_names):
            with cols[j]:
                st.markdown(f"**{name}**")
                placeholders.append(st.empty())

    captures = [cv2.VideoCapture(str(videos[nm])) for nm in vid_names]
    track_hits_list: list[Counter] = [Counter() for _ in range(n)]
    class_hits_list: list[dict[str, Counter]] = [defaultdict(Counter) for _ in range(n)]
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
                    track_hits_list[i],
                    class_hits_list[i],
                )

                last_bytes_list[i] = _frame_to_bytes(annotated)
                placeholders[i].image(last_bytes_list[i], width="stretch")

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                churn, stable = _track_quality(track_hits_list[i])
                metric_phs[i].markdown(
                    f"**{vid_names[i]}** — Frame {frame_nums[i]} · "
                    f"{obj_count} obj · {stable} tracked · churn {churn} · "
                    f"{fps:.1f} FPS"
                )
    finally:
        for cap in captures:
            cap.release()

    # Per-video summary. IDs are per-video: two videos may both report ID 3 for
    # different objects, so these counts are never summed.
    for i, name in enumerate(vid_names):
        churn, stable = _track_quality(track_hits_list[i])
        st.success(
            f"✅ **{name}**: {frame_nums[i]} frames"
            + (f" — **{stable}** unique objects · churn {churn}" if stable else "")
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
    # Scan videos/ on every run so newly added files appear immediately
    videos = config.get_videos_dict()

    if not videos:
        st.warning("No videos found in the `videos/` directory.")
        return

    vid_names = st.sidebar.multiselect(
        "Choose video(s)",
        list(videos.keys()),
        default=[list(videos.keys())[0]],
        help="Select multiple videos for simultaneous detection.",
    )

    if not vid_names:
        st.info("Select at least one video from the sidebar.")
        return

    # ── Preview selected videos in a 3-per-row grid ──────────────
    _COLS_PER_ROW = 3
    for row_start in range(0, len(vid_names), _COLS_PER_ROW):
        row_slice = vid_names[row_start : row_start + _COLS_PER_ROW]
        cols = st.columns(_COLS_PER_ROW)
        for j, name in enumerate(row_slice):
            with cols[j]:
                st.markdown(f"**{name}**")
                st.video(str(videos[name]))

    if st.sidebar.button("🚀 Detect Video Objects", type="primary"):
        if len(vid_names) == 1:
            _run_video_loop(
                cv2.VideoCapture(str(videos[vid_names[0]])),
                model,
                confidence,
                enable_tracking,
                tracker,
                skip_frames,
            )
        else:
            _run_multi_video_loop(
                vid_names,
                videos,
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

    track_hits_global: Counter = Counter()
    class_hits_global: dict[str, Counter] = defaultdict(Counter)

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
                        track_hits_global[int(track_id)] += 1
                        name = names[int(cls_id)]
                        class_hits_global.setdefault(name, Counter())[
                            int(track_id)
                        ] += 1

            annotated = _annotate_with_ids(img, result, enable_tracking)
            annotated = _draw_overlay(
                annotated,
                len(result.boxes) if result.boxes is not None else 0,
                frame_class_counts,
                len(_confirmed(track_hits_global)) if enable_tracking else None,
                class_hits_global if enable_tracking else None,
                _track_quality(track_hits_global) if enable_tracking else None,
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
