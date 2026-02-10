"""
Centralized model loading with Streamlit caching.
Models are loaded once and reused across reruns.
"""

import torch
import streamlit as st
from ultralytics import YOLO, YOLOWorld

import config


@st.cache_resource
def load_model(model_name: str) -> YOLO:
    """Load a YOLO / RT-DETR model (detection / segmentation / pose).

    Checks the local ``weights/`` directory first; falls back to
    ultralytics auto-download, then sweeps stray weights into ``weights/``.
    """
    path = config.resolve_model_path(model_name)
    model = YOLO(path)
    config.sweep_stray_weights()
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
    except Exception:
        pass


def _set_world_classes(model: YOLOWorld, classes: list[str]) -> None:
    """Safely set classes on a YOLOWorld model, avoiding CPU/CUDA mismatch.

    The trick: move entire model to CPU → set_classes (creates CPU
    text features) → move everything to CUDA uniformly.
    """
    try:
        # Get the device the model is currently on
        current_device = next(model.model.parameters()).device
        # Move to CPU so set_classes() creates embeddings on the same device
        model.to("cpu")
        model.set_classes(classes)
        # Move everything (weights + new text embeddings) to target device
        model.to(current_device)
    except StopIteration:
        # No parameters found — just set classes directly
        model.set_classes(classes)
    except Exception:
        # Fallback: set classes then brute-force device sync
        model.set_classes(classes)
        _ensure_device(model)


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
    }

    name = model_name or _DEFAULTS.get(task, config.DETECTION_MODEL)

    try:
        if task == config.TASK_WORLD:
            model = load_world_model(name)
            if world_classes:
                _set_world_classes(model, world_classes)
            return model
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
    }

    name = model_name or _DEFAULTS.get(task, config.DETECTION_MODEL)
    path = config.resolve_model_path(name)

    if task == config.TASK_WORLD:
        m = YOLOWorld(path)
        config.sweep_stray_weights()
        if world_classes:
            _set_world_classes(m, world_classes)
        return m

    m = YOLO(path)
    config.sweep_stray_weights()
    return m
