"""Execute selected notebook cells in a fresh kernel and write outputs back."""
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from pathlib import Path
import copy, sys, time

NB_PATH = Path('segment_level_analysis.ipynb')
SETUP_CELLS = [1, 3, 5, 70, 78]   # produce globals needed by §18
TARGET_CELLS = [95, 96, 97, 99]   # cells we want executed and updated

nb = nbformat.read(NB_PATH, as_version=4)
all_idx = SETUP_CELLS + TARGET_CELLS
print(f"Will run cells: {all_idx}")

# Build a temp notebook with just those cells, in order
mini = copy.deepcopy(nb)
mini.cells = [copy.deepcopy(nb.cells[i]) for i in all_idx]

client = NotebookClient(
    mini,
    timeout=900,
    kernel_name='python3',
    allow_errors=False,
    resources={'metadata': {'path': str(NB_PATH.parent.absolute())}},
)

t0 = time.time()
try:
    client.execute()
except CellExecutionError as e:
    print(f"!! cell errored: {e}")
    raise
print(f"All cells executed in {time.time()-t0:.0f}s")

# Copy outputs/execution_count back into the original notebook
for src_pos, orig_idx in enumerate(all_idx):
    src_cell = mini.cells[src_pos]
    dst_cell = nb.cells[orig_idx]
    if dst_cell.cell_type != 'code':
        continue
    dst_cell.outputs = src_cell.get('outputs', [])
    dst_cell.execution_count = src_cell.get('execution_count')

nbformat.write(nb, NB_PATH)
print(f"Wrote outputs back to {NB_PATH}")
