"""
Centralized model loading with Streamlit caching.
Models are loaded once and reused across reruns.
"""

import streamlit as st
from ultralytics import YOLO, YOLOWorld

import config


@st.cache_resource
def load_model(model_name: str) -> YOLO:
    """Load a YOLO model (detection / segmentation / pose).

    Checks the local ``weights/`` directory first; falls back to
    ultralytics auto-download.
    """
    path = config.resolve_model_path(model_name)
    return YOLO(path)


@st.cache_resource
def load_world_model(model_name: str) -> YOLOWorld:
    """Load a YOLO World v2 model for open-vocabulary detection.

    Uses the ``YOLOWorld`` class which supports natural language
    text prompts like "person in black", "red car", etc.
    The model is cached, but ``set_classes()`` must be called
    before each inference because the class list can change.
    """
    path = config.resolve_model_path(model_name)
    return YOLOWorld(path)


def get_model_for_task(task: str, world_classes: list[str] | None = None) -> YOLO | YOLOWorld | None:
    """Return the appropriate model for *task*.

    Parameters
    ----------
    task : str
        One of the ``config.TASKS_LIST`` values.
    world_classes : list[str] | None
        Required when *task* is ``config.TASK_WORLD``.

    Returns
    -------
    YOLO | YOLOWorld | None
        The loaded model, or ``None`` on error.
    """
    try:
        if task == config.TASK_DETECT:
            return load_model(config.DETECTION_MODEL)
        if task == config.TASK_SEGMENT:
            return load_model(config.SEGMENTATION_MODEL)
        if task == config.TASK_POSE:
            return load_model(config.POSE_MODEL)
        if task == config.TASK_WORLD:
            model = load_world_model(config.YOLO_WORLD_MODEL)
            if world_classes:
                model.set_classes(world_classes)
            return model
    except Exception as exc:
        st.error(f"❌ Failed to load model for **{task}**: {exc}")
        return None
    return None
