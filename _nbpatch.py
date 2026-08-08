"""One-shot patcher for docs/yolo26_playground.ipynb (deleted after use)."""

import json
from pathlib import Path

NB = Path("docs/yolo26_playground.ipynb")
nb = json.loads(NB.read_text())
cells = nb["cells"]


def src(i):
    return "".join(cells[i]["source"])


def set_src(i, text):
    cells[i]["source"] = text.splitlines(keepends=True)


def sub(i, old, new):
    s = src(i)
    assert old in s, f"cell {i}: anchor not found:\n{old!r}"
    set_src(i, s.replace(old, new, 1))


# ── Cell 3: reload config so a long-lived kernel picks up config.py changes ──
sub(
    3,
    """import cv2
import numpy as np""",
    """import importlib

import cv2
import numpy as np""",
)
sub(
    3,
    """import config  # the repo's own config module

DEVICE""",
    """import config  # the repo's own config module

# A kernel started before config.py last changed still holds the old module object:
# `import config` is a no-op once it is in sys.modules, so anything added since shows
# up as AttributeError. Reloading makes this cell idempotent across an edit.
config = importlib.reload(config)

DEVICE""",
)

# ── Cell 7: w() must never hand a bare name back to Ultralytics ──────────────
sub(
    7,
    """    if not local.exists():
        print(f"↓ {name} not in weights/ — downloading it there")
        with weights_cwd():
            attempt_download_asset(name)
    return config.resolve_model_path(name)""",
    """    if not local.exists():
        print(f"↓ {name} not in weights/ — downloading it there")
        with weights_cwd():
            attempt_download_asset(name)
        config.sweep_stray_weights()  # in case it still landed beside the CWD
    if not local.exists():
        raise FileNotFoundError(
            f"{name} is not in {config.WEIGHTS_DIR} and could not be downloaded there. "
            "Returning the bare name would make Ultralytics fetch it into the CWD, "
            "which is exactly what this notebook refuses to do."
        )
    return str(local)""",
)

# ── Cell 37: the CLIP text encoder must land in weights/, not the repo root ──
sub(
    37,
    """    m.to("cpu")  # 1. weights to CPU so embeddings land on the same device
    m.set_classes(classes)  # 2. text embeddings created — on CPU
    config.sweep_stray_weights()  #    ↑ first call downloads the CLIP text encoder
    m.to(DEVICE)  # 3. weights + embeddings together to the real device""",
    """    m.to("cpu")  # 1. weights to CPU so embeddings land on the same device
    with weights_cwd():  #    the CLIP encoder downloads into the CWD, so make the
        m.set_classes(classes)  # 2. CWD weights/ — text embeddings created, on CPU
    config.sweep_stray_weights()  #    belt and braces for older ultralytics
    m.to(DEVICE)  # 3. weights + embeddings together to the real device""",
)

# ── Cell 41: same encoder, same treatment ────────────────────────────────────
sub(
    41,
    """    yoloe.to("cpu")
    yoloe.set_classes(names, yoloe.get_text_pe(names))  # ← two arguments
    config.sweep_stray_weights()  # get_text_pe() pulls the same CLIP encoder""",
    """    yoloe.to("cpu")
    with weights_cwd():  # get_text_pe() pulls the same CLIP encoder — into weights/
        yoloe.set_classes(names, yoloe.get_text_pe(names))  # ← two arguments
    config.sweep_stray_weights()""",
)

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("patched")
