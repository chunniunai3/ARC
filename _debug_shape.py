import sys; sys.path.insert(0, '.')
import torch
import numpy as np
from src.varc import VARCConfig, VARCModel, load_checkpoint
from src.data_loader import load_tasks, task_to_arrays
from src.varc.data import place_on_canvas_input, place_on_canvas_target, BG_CLASS, BD_CLASS
from src.varc.evaluate import recover_shape_from_bd

device = torch.device('cuda')
config = VARCConfig()
model = VARCModel(config).to(device)
load_checkpoint(model, 'results/checkpoints/varc_exp3_base.pt', device)
model.eval()

# Test on a task with non-zero prediction
tasks = load_tasks(split='training', max_tasks=30)
task = tasks[28]  # 228f6490
print(f"Task: {task['id']}")
arrays = task_to_arrays(task)
test_in = arrays["test"][0]["input"]
print(f"Test input shape: {test_in.shape}")
test_out = arrays["test"][0].get("output")
print(f"Test output shape: {test_out.shape}")

# Prepare input
inp, inh, inw, iy, ix = place_on_canvas_input(test_in, 64, scale=1)
print(f"Input canvas shape: {inp.shape}, offset=({iy},{ix}), size=({inh},{inw})")

# Run inference
inp_t = torch.tensor(inp).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(inp_t)
    pred = logits[0].argmax(dim=0).cpu().numpy()

print(f"\nPred unique values: {np.unique(pred)}")
print(f"BG count: {(pred == BG_CLASS).sum()}")
print(f"BD count: {(pred == BD_CLASS).sum()}")

# Check BD positions
bd_pos = np.where(pred == BD_CLASS)
print(f"BD positions: {list(zip(bd_pos[0][:5], bd_pos[1][:5]))}")

# Recover shape from BD
top, left, bottom, right = recover_shape_from_bd(pred)
print(f"\nBD shape recovery: top={top}, left={left}, bottom={bottom}, right={right}")

if top == 0 and left == 0 and bottom == 0 and right == 0:
    print("BD recovery failed, using fallback")
    from src.varc.evaluate import extract_grid_from_canvas
    h, w = len(test_out), len(test_out[0])
    grid = extract_grid_from_canvas(pred, h, w)
    print(f"Fallback grid shape: {grid.shape}")
else:
    grid_area = pred[top:bottom, left:right]
    print(f"Grid area shape: {grid_area.shape}")
    print(f"Grid area unique: {np.unique(grid_area)}")

# Print full predicted grid
print(f"\nExpected:\n{test_out}")
pred_first = pred[:20, :20] if pred.shape[0] > 20 and pred.shape[1] > 20 else pred
print(f"\nPredicted canvas (top-left 20x20):\n{pred_first}")
