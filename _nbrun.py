import sys, nbformat
from nbclient import NotebookClient

src = sys.argv[1] if len(sys.argv) > 1 else "docs/yolo26_playground.ipynb"
out = sys.argv[2] if len(sys.argv) > 2 else "_playground_out/_run.ipynb"

nb = nbformat.read(src, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": "docs/"}},
                        allow_errors=True)
client.execute()
nbformat.write(nb, out)

fails = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            fails += 1
            print("=" * 72)
            print(f"CELL {i}  {o['ename']}: {o['evalue']}")
            print("\n".join(o["traceback"][-12:]))
print("=" * 72)
print("FAILED CELLS:", fails)
