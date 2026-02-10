"""
YOLO Vision Studio — Main Application
======================================
Real-time Object Detection, Segmentation, Pose Estimation & Tracking
powered by YOLO26, YOLO World v2 and Streamlit.
"""

import streamlit as st
import config
import image_service
import video_service

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.caption(f"v{config.APP_VERSION}")
    st.markdown("---")

    # 1️⃣  Inference mode ─────────────────────────────────────────────────────
    mode = st.radio("Inference Mode", config.MODES_LIST, key="mode")

    # 2️⃣  Task selection ─────────────────────────────────────────────────────
    task = st.radio("Task", config.TASKS_LIST, key="task")

    st.markdown("---")

    # 3️⃣  Confidence slider ──────────────────────────────────────────────────
    confidence = (
        st.slider(
            "Model Confidence (%)",
            min_value=config.MIN_CONFIDENCE,
            max_value=config.MAX_CONFIDENCE,
            value=int(config.DEFAULT_CONFIDENCE * 100),
        )
        / 100.0
    )

# ── Main content ──────────────────────────────────────────────────────────────
st.title(config.APP_TITLE)
st.markdown(f"*{config.APP_DESCRIPTION}*")

if mode == config.MODE_IMAGE:
    image_service.render(task, confidence)
elif mode == config.MODE_VIDEO:
    video_service.render(task, confidence)
else:
    st.error("Please select a valid inference mode.")
