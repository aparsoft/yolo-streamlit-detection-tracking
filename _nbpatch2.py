import json
from pathlib import Path

NB = Path("docs/yolo26_playground.ipynb")
nb = json.loads(NB.read_text())
cells = nb["cells"]

def sub(i, old, new):
    s = "".join(cells[i]["source"])
    assert old in s, f"cell {i}: anchor not found:\n{old!r}"
    cells[i]["source"] = s.replace(old, new, 1).splitlines(keepends=True)

# ── 9.4 · `half` was renamed to `quantize` in ultralytics 8.4 ───────────────
sub(64, "### 9.4 Half precision", """### 9.4 Half precision

The argument is `quantize`, not `half`. Ultralytics 8.4 folded the old `half` and `int8`
switches into one parameter: `quantize=16` for FP16, `32` for FP32, and — on `export()` only —
`8` for INT8. `half=True` still works and still warns; it is scheduled for removal.""")

sub(65, """    fp16 = bench(lambda: det.predict(img, conf=0.4, half=True, verbose=False), n=15)
    r32 = det.predict(img, conf=0.4, verbose=False)[0]
    r16 = det.predict(img, conf=0.4, half=True, verbose=False)[0]""",
"""    fp16 = bench(
        lambda: det.predict(img, conf=0.4, quantize=16, verbose=False), n=15
    )
    r32 = det.predict(img, conf=0.4, verbose=False)[0]
    r16 = det.predict(img, conf=0.4, quantize=16, verbose=False)[0]""")

# ── 11 · same rename on the export path ─────────────────────────────────────
sub(76, 'exported = Path(m.export(format="onnx", imgsz=640, half=False))',
       'exported = Path(m.export(format="onnx", imgsz=640, quantize=32))')
sub(76, '# engine_path = m.export(format="engine", imgsz=640, half=True)',
       '# engine_path = m.export(format="engine", imgsz=640, quantize=16)')

# ── 13 · CLI: keep every weight in weights/, and use the current arg name ───
sub(83, "yolo train model=yolo26n.pt data=custom.yaml",
       "yolo train model=weights/yolo26n.pt data=custom.yaml")
sub(83, "yolo export model=weights/yolo26n.pt format=engine half=True",
       "yolo export model=weights/yolo26n.pt format=engine quantize=16")

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("patched")
