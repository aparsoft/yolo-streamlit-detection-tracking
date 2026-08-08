"""
Image inference service — Detection, Segmentation, YOLO World & Pose.

Provides a single ``render()`` entry-point called from *app.py*.
"""

from __future__ import annotations

import PIL.Image
import pandas as pd
import streamlit as st

import config
from model_loader import get_model_for_task

# ── Public API ────────────────────────────────────────────────────────────────


def render(task: str, confidence: float, selected_model: str | None = None) -> None:
    """Render the full image-inference page for the chosen *task*."""
    st.header(f"📷 Image · {task}")

    # YOLO World / YOLOE need a text prompt before anything else
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

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload",
    )

    col1, col2 = st.columns(2)

    with col1:
        if uploaded:
            image = PIL.Image.open(uploaded)
            st.image(image, caption="Original Image", width="stretch")
        elif config.DEFAULT_IMAGE.exists():
            st.image(
                str(config.DEFAULT_IMAGE),
                caption="Default Image",
                width="stretch",
            )

    with col2:
        if uploaded:
            if st.button(f"🚀 Run {task}", type="primary", width="stretch"):
                _run_inference(model, image, confidence, task)
        elif config.DEFAULT_DETECT_IMAGE.exists():
            st.image(
                str(config.DEFAULT_DETECT_IMAGE),
                caption="Detected Image",
                width="stretch",
            )


# ── YOLOE helpers ─────────────────────────────────────────────────────────────


def _yoloe_class_input() -> list[str] | None:
    """Show a text-area for the user to type category-level object classes."""
    st.markdown(
        "💡 **Tip**: YOLOE supports **category-level** labels like `person`, `car`, `dog`. "
        "It does **NOT** support descriptive phrases like *person in red shirt*. "
        "Results include both bounding boxes **and** segmentation masks."
    )
    text = st.text_area(
        "🔍 Enter object categories to detect & segment (comma-separated)",
        value=config.DEFAULT_YOLOE_CLASSES,
        help=(
            "YOLOE performs open-vocabulary detection + instance segmentation. "
            "Use simple category names only (e.g. person, car, laptop)."
        ),
    )
    classes = [c.strip() for c in text.split(",") if c.strip()]
    if classes:
        st.info(f"🎯 Detecting & segmenting: **{', '.join(classes)}**")
    else:
        st.warning("⚠️ Please enter at least one object category.")
    return classes or None


# ── YOLO World helpers ────────────────────────────────────────────────────────


def _world_class_input() -> list[str] | None:
    """Show a text-area for the user to type object classes / descriptive prompts."""
    st.markdown(
        "💡 **Tip**: YOLO World v2 supports natural language prompts! "
        "Try descriptive phrases like `person in black`, `red car`, `wooden chair`."
    )
    text = st.text_area(
        "🔍 Enter object classes or descriptions (comma-separated)",
        value=config.DEFAULT_WORLD_CLASSES,
        help=(
            "YOLO World v2 performs open-vocabulary detection based on your text prompt. "
            "You can use simple class names OR descriptive phrases."
        ),
    )
    classes = [c.strip() for c in text.split(",") if c.strip()]
    if classes:
        st.info(f"🎯 Detecting: **{', '.join(classes)}**")
    else:
        st.warning("⚠️ Please enter at least one class to detect.")
    return classes or None


# ── Inference logic ───────────────────────────────────────────────────────────


def _run_inference(model, image: PIL.Image.Image, confidence: float, task: str) -> None:
    with st.spinner(f"Running {task}…"):
        results = model.predict(image, conf=confidence)
        result = results[0]

        annotated = result.plot()[:, :, ::-1]  # BGR → RGB
        st.image(annotated, caption=f"{task} Result", width="stretch")

    _display_results(result, task)


def _display_results(result, task: str) -> None:
    """Show structured results below the annotated image."""
    with st.expander("📊 Results", expanded=True):

        # ── Boxes (Detection / Segmentation / World) ──────────────────────
        if result.boxes is not None and len(result.boxes):
            names = result.names
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            # Summary metrics
            summary: dict[str, dict] = {}
            for cls_id, conf in zip(classes, confs):
                name = names[int(cls_id)]
                entry = summary.setdefault(name, {"count": 0, "total_conf": 0.0})
                entry["count"] += 1
                entry["total_conf"] += float(conf)

            cols = st.columns(min(len(summary), 4) or 1)
            for idx, (name, data) in enumerate(summary.items()):
                avg = data["total_conf"] / data["count"]
                cols[idx % len(cols)].metric(
                    label=name.capitalize(),
                    value=data["count"],
                    delta=f"{avg:.0%} avg conf",
                )

            # Detailed table
            rows = [
                {"Class": names[int(c)], "Confidence": f"{cf:.2%}"}
                for c, cf in zip(classes, confs)
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch")

        # ── Keypoints (Pose) ──────────────────────────────────────────────
        if task == config.TASK_POSE:
            kp = getattr(result, "keypoints", None)
            if kp is not None and len(kp):
                st.success(f"✅ Detected **{len(kp)}** pose(s)")
            else:
                st.info("No poses detected — try lowering the confidence threshold.")

        # ── Masks (Segmentation / YOLOE) ──────────────────────────────────
        if task in (config.TASK_SEGMENT, config.TASK_YOLOE):
            masks = getattr(result, "masks", None)
            if masks is not None and len(masks):
                st.success(f"✅ Segmented **{len(masks)}** object(s)")

        # ── Empty state ───────────────────────────────────────────────────
        if result.boxes is None or len(result.boxes) == 0:
            st.info("No objects detected. Try lowering the confidence threshold.")
