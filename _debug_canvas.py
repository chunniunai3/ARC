import sys; sys.path.insert(0, '.')
import torch
import numpy as np
from src.varc import VARCConfig, VARCModel, load_checkpoint
from src.data_loader import load_tasks, task_to_arrays
from src.varc.data import place_on_canvas_input, place_on_canvas_target, BG_CLASS, BD_CLASS
from src.varc.evaluate import recover_shape_from_bd, extract_grid_from_canvas

device = torch.device('cuda')
config = VARCConfig()
model = VARCModel(config).to(device)
load_checkpoint(model, 'results/checkpoints/varc_exp3_base.pt', device)
model.eval()

tasks = load_tasks(split='training', max_tasks=50)

# Debug first task
for task_idx in [0, 3, 21, 28]:
    task = tasks[task_idx]
    arrays = task_to_arrays(task)
    test_in = arrays["test"][0]["input"]
    test_out = arrays["test"][0].get("output")
    print(f"\n{'='*60}")
    print(f"Task {task_idx}: {task['id']}")
    print(f"Input shape: {test_in.shape}")
    print(f"Output shape: {test_out.shape}")

    inp, inh, inw, iy, ix = place_on_canvas_input(test_in, 64)
    print(f"Canvas: offset=({iy},{ix}), content_size=({inh},{inw})")

    inp_t = torch.tensor(inp).unsqueeze(0).to(device)
    tids = torch.full((1,), task_idx, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(inp_t, task_ids=tids)
        pred = logits[0].argmax(dim=0).cpu().numpy()

    print(f"Canvas unique: {np.unique(pred)}")
    print(f"BG count: {(pred == BG_CLASS).sum()}, BD count: {(pred == BD_CLASS).sum()}")

    top, left, bottom, right = recover_shape_from_bd(pred)
    print(f"BD recovery: top={top}, left={left}, bottom={bottom}, right={right}")

    # Check non-BG positions
    non_bg = np.where(pred != BG_CLASS)
    if len(non_bg[0]) > 0:
        print(f"Non-BG range: rows [{non_bg[0].min()}, {non_bg[0].max()}], cols [{non_bg[1].min()}, {non_bg[1].max()}]")
        # Sample some non-BG values
        sample_ys = sorted(set(non_bg[0].tolist()))[:5]
        for y in sample_ys:
            row = pred[y, non_bg[1][non_bg[0] == y]]
            print(f"  row {y}: {row[:12].tolist()}")

    if top == 0 and left == 0 and bottom == 0 and right == 0:
        h, w = test_out.shape
        grid = extract_grid_from_canvas(pred, h, w)
        print(f"Fallback grid shape={grid.shape}, unique={np.unique(grid)}")
    else:
        grid_area = pred[top:bottom, left:right]
        print(f"Grid area shape={grid_area.shape}, unique={np.unique(grid_area)}")
        grid_area[grid_area == BG_CLASS] = 0
        grid_area[grid_area == BD_CLASS] = 0

    print(f"Expected: {test_out.tolist()}")
