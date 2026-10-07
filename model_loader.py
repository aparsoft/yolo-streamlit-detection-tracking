"""
Centralized model loading with Streamlit caching.
Models are loaded once and reused across reruns.
"""

import threading
import warnings

import torch
import streamlit as st
from ultralytics import YOLO, YOLOWorld, YOLOE

import config

# Reuse weights already in weights/ instead of re-downloading them into the CWD.
config.use_local_weights_dir()

# ``@st.cache_resource`` models are shared by every browser session, and each session's
# script runs on its own thread. An Ultralytics predictor is not thread-safe (its
# dataset, batch and tracker state live on the object), so two visitors running at once
# could interleave inside one ``predict()``. Inference on a shared model takes this lock;
# session-owned models (see ``get_session_model``) don't need it.
SHARED_MODEL_LOCK = threading.RLock()

# Set on models that belong to one session / one video, so callers can skip the lock.
_OWNED_ATTR = "_studio_session_owned"


def is_session_owned(model) -> bool:
    return bool(getattr(model, _OWNED_ATTR, False))


@st.cache_resource
def load_model(model_name: str) -> YOLO:
    """Load a YOLO / RT-DETR model (detection / segmentation / pose).

    Checks the local ``weights/`` directory first; falls back to
    ultralytics auto-download, then sweeps stray weights into ``weights/``.
    """
    path = config.resolve_model_path(model_name)
    model = YOLO(path)
    config.sweep_stray_weights()
    _ensure_device(model)
    return model


@st.cache_resource
def load_world_model(model_name: str) -> YOLOWorld:
    """Load a YOLO World v2 model for open-vocabulary detection.

    Uses the ``YOLOWorld`` class which supports natural language
    text prompts like "person in black", "red car", etc.
    """
    path = config.resolve_model_path(model_name)
    model = YOLOWorld(path)
    config.sweep_stray_weights()
    return model


@st.cache_resource
def load_yoloe_model(model_name: str) -> YOLOE:
    """Load a YOLOE model for text-prompted detection + segmentation.

    Uses the ``YOLOE`` class which supports category-level text prompts
    and produces both bounding boxes and instance segmentation masks.
    """
    path = config.resolve_model_path(model_name)
    model = YOLOE(path)
    config.sweep_stray_weights()
    return model


@st.cache_resource(max_entries=4, show_spinner="Encoding text prompt…")
def _load_prompted_model(
    task: str, model_name: str, classes: tuple[str, ...]
) -> YOLOWorld | YOLOE:
    """Load an open-vocabulary model with *classes* already embedded.

    The prompt is part of the cache key, so the text encoder runs once per distinct
    prompt instead of once per rerun. That matters more than it sounds: applying a
    prompt is CPU → ``set_classes`` → CUDA, a full round trip of the weights, measured
    at ~3.6 s on this machine. Doing it unconditionally in ``get_model_for_task`` meant
    every confidence-slider nudge froze the app for those 3.6 s.

    ``max_entries`` caps how many prompt variants stay resident — each entry is a whole
    model on the GPU.
    """
    path = config.resolve_model_path(model_name)
    model = YOLOE(path) if task == config.TASK_YOLOE else YOLOWorld(path)
    config.sweep_stray_weights()

    # CPU first so set_classes() builds the text embeddings on the device the weights
    # are on, then move weights *and* embeddings together. See _set_world_classes.
    model.to("cpu")
    model.set_classes(list(classes))
    _ensure_device(model)
    return model


def _ensure_device(model) -> None:
    """Move model to the best available device.

    This fixes the "index is on cpu, different from cuda:0" error
    that occurs when ``set_classes()`` creates CPU tensors while the
    model weights live on CUDA.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        # Move the inner nn.Module (catches buffers + non-parameter tensors)
        if hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to(device)
        model.to(device)
    except Exception as exc:
        # Not fatal — Ultralytics places the model itself at predict time — but a silent
        # ``pass`` here once hid a CPU/CUDA mismatch for a whole debugging session.
        warnings.warn(f"Could not move model to {device}: {exc}", RuntimeWarning)


def _set_world_classes(model: YOLOWorld, classes: list[str]) -> None:
    """Safely set classes on a YOLOWorld model, avoiding CPU/CUDA mismatch.

    ``set_classes()`` internally creates text-embedding tensors on CPU.
    If the model is already on CUDA (e.g. after a prior ``predict()``),
    calling ``set_classes()`` directly causes a device mismatch crash.

    Fix: move model → CPU → set_classes → move to **best** device.
    We always move to the best available device (CUDA if present) after
    setting classes, because freshly loaded models start on CPU and
    ``predict()`` alone may not move the text-embedding tensors.
    """
    # 1. CPU so set_classes creates embeddings on the same device as weights
    model.to("cpu")
    model.set_classes(classes)

    # 2. Move everything (weights + fresh text embeddings) to best device
    best = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(best)


def _set_yoloe_classes(model: YOLOE, classes: list[str]) -> None:
    """Safely set classes on a YOLOE model, avoiding CPU/CUDA mismatch.

    Same pattern as ``_set_world_classes`` but for YOLOE.
    YOLOE supports category-level labels (not descriptive phrases).
    """
    model.to("cpu")
    model.set_classes(classes)
    best = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(best)


def get_model_for_task(
    task: str,
    world_classes: list[str] | None = None,
    model_name: str | None = None,
) -> YOLO | YOLOWorld | None:
    """Return the appropriate model for *task*.

    Parameters
    ----------
    task : str
        One of the ``config.TASKS_LIST`` values.
    world_classes : list[str] | None
        Required when *task* is ``config.TASK_WORLD``.
    model_name : str | None
        Specific model filename chosen by the user via sidebar.
        Falls back to the default for the task if ``None``.
    """
    _DEFAULTS = {
        config.TASK_DETECT: config.DETECTION_MODEL,
        config.TASK_SEGMENT: config.SEGMENTATION_MODEL,
        config.TASK_POSE: config.POSE_MODEL,
        config.TASK_WORLD: config.YOLO_WORLD_MODEL,
        config.TASK_YOLOE: config.YOLOE_MODEL,
    }

    name = model_name or _DEFAULTS.get(task, config.DETECTION_MODEL)

    try:
        if task in (config.TASK_WORLD, config.TASK_YOLOE):
            if not world_classes:
                # No prompt yet — hand back the bare model rather than embedding "".
                loader = (
                    load_yoloe_model if task == config.TASK_YOLOE else load_world_model
                )
                return loader(name)
            # tuple() so the prompt is hashable and the cache key is order-sensitive
            return _load_prompted_model(task, name, tuple(world_classes))
        return load_model(name)
    except Exception as exc:
        st.error(f"❌ Failed to load model for **{task}**: {exc}")
        return None


def load_fresh_model(
    task: str,
    world_classes: list[str] | None = None,
    model_name: str | None = None,
) -> YOLO | YOLOWorld:
    """Load a **fresh** (uncached) model instance.

    Used by multi-video mode so each video gets isolated tracking
    state (ByteTrack / BoTSORT state lives on the model).
    """
    _DEFAULTS = {
        config.TASK_DETECT: config.DETECTION_MODEL,
        config.TASK_SEGMENT: config.SEGMENTATION_MODEL,
        config.TASK_POSE: config.POSE_MODEL,
        config.TASK_WORLD: config.YOLO_WORLD_MODEL,
        config.TASK_YOLOE: config.YOLOE_MODEL,
    }

    name = model_name or _DEFAULTS.get(task, config.DETECTION_MODEL)
    path = config.resolve_model_path(name)

    if task == config.TASK_WORLD:
        m = YOLOWorld(path)
        config.sweep_stray_weights()
        if world_classes:
            _set_world_classes(m, world_classes)
        return m

    if task == config.TASK_YOLOE:
        m = YOLOE(path)
        config.sweep_stray_weights()
        if world_classes:
            _set_yoloe_classes(m, world_classes)
        return m

    m = YOLO(path)
    config.sweep_stray_weights()
    return m


def get_session_model(
    task: str,
    world_classes: list[str] | None = None,
    model_name: str | None = None,
) -> YOLO | YOLOWorld | YOLOE:
    """A model owned by **this browser session** — used for tracking.

    Tracker state (``persist=True``) lives on the model's predictor. On the shared
    ``@st.cache_resource`` model that state was shared by every visitor: two people
    playing videos at once fed one tracker, and one run's end-of-video reset wiped the
    other's IDs mid-stream. A session-owned copy isolates all of it.

    Only one is kept per session (each is a full copy of the weights); switching model,
    task or prompt replaces it.
    """
    store: dict = st.session_state.setdefault("_session_models", {})
    key = (task, model_name, tuple(world_classes or ()))
    model = store.get(key)
    if model is None:
        store.clear()
        model = load_fresh_model(task, world_classes, model_name=model_name)
        _ensure_device(model)
        setattr(model, _OWNED_ATTR, True)
        store[key] = model
    return model


def mark_session_owned(model):
    """Flag a model that no other session can reach (e.g. multi-video's per-video copies)."""
    setattr(model, _OWNED_ATTR, True)
    return model
