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

task_idx = 0
task = load_tasks(split='training', max_tasks=50)[task_idx]
arrays = task_to_arrays(task)
test_in = arrays["test"][0]["input"]
test_out = arrays["test"][0].get("output")

inp, inh, inw, iy, ix = place_on_canvas_input(test_in, 64)
print(f"Input offset=({iy},{ix}), size=({inh},{inw})")
print(f"Output shape: {test_out.shape}")

# Also get target
target = place_on_canvas_target(test_out, 64, inh, inw, iy, ix)
print(f"Target unique: {np.unique(target)}")
print(f"Target BD at right (col {ix+inw-1}): {target[iy:iy+inw, ix+inw-1][:10]}")
print(f"Target BD at bottom (row {iy+inh-1}): {target[iy+inh-1, ix:ix+inw][:10]}")

inp_t = torch.tensor(inp).unsqueeze(0).to(device)
tids = torch.full((1,), task_idx, dtype=torch.long, device=device)
with torch.no_grad():
    logits = model(inp_t, task_ids=tids)
    pred = logits[0].argmax(dim=0).cpu().numpy()

print(f"\nPred unique: {np.unique(pred)}")
print(f"Target unique: {np.unique(target)}")

# Compare target vs pred in the grid area
grid_area = pred[iy:iy+inh, ix:ix+inw]
print(f"\nPred grid area ({iy}:{iy+inh}, {ix}:{ix+inw}) shape={grid_area.shape}")
print(f"Grid area unique: {np.unique(grid_area)}")
print(f"First 3 rows:")
for r in range(min(3, grid_area.shape[0])):
    print(f"  row {r}: {grid_area[r, :12].tolist()}")

# Check where BD is predicted
bd_mask = pred == BD_CLASS
bd_rows = np.where(bd_mask.any(axis=1))[0]
bd_cols = np.where(bd_mask.any(axis=0))[0]
print(f"\nBD rows: {bd_rows[:20]}")
print(f"BD cols: {bd_cols[:20]}")

# Recover shape and resize
top, left, bottom, right = recover_shape_from_bd(pred)
print(f"\nRecovered: top={top}, left={left}, bottom={bottom}, right={right}")
print(f"Expected grid: {test_out.tolist()}")

# Direct comparison: what if we just crop the grid area?
from PIL import Image
h_exp, w_exp = test_out.shape
scale = max(1, (64 - 4) // max(h_exp, w_exp))
out_h, out_w = h_exp * scale, w_exp * scale
y_off = (64 - out_h) // 2
x_off = (64 - out_w) // 2
print(f"\nExpected placement: offset=({y_off},{x_off}), size=({out_h},{out_w})")
crop = pred[y_off:y_off+out_h, x_off:x_off+out_w]
print(f"Crop shape: {crop.shape}")
crop[crop == BG_CLASS] = 0
crop[crop == BD_CLASS] = 0
if crop.shape != (h_exp, w_exp):
    img = Image.fromarray(crop.astype(np.uint8))
    crop = np.array(img.resize((w_exp, h_exp), Image.NEAREST))
print(f"Resized crop: {crop.tolist()}")
