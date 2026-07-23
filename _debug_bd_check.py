import sys; sys.path.insert(0, '.')
import torch
import numpy as np
from src.varc import VARCConfig, VARCModel, load_checkpoint
from src.data_loader import load_tasks, task_to_arrays
from src.varc.data import place_on_canvas_input, place_on_canvas_target, BG_CLASS, BD_CLASS
from src.varc.evaluate import extract_grid_from_canvas

device = torch.device('cuda')
config = VARCConfig()
model = VARCModel(config).to(device)
load_checkpoint(model, 'results/checkpoints/varc_exp3_base.pt', device)
model.eval()

# Check task 06df4c85 which should benefit from BD boundary fix
idx = 8
task = load_tasks(split='training', max_tasks=50)[idx]
arrays = task_to_arrays(task)
test_out = arrays["test"][0].get("output")
test_in = arrays["test"][0]["input"]
h, w = test_out.shape

inp, inh, inw, iy, ix = place_on_canvas_input(test_in, 64)
target = place_on_canvas_target(test_out, 64, inh, inw, iy, ix)

inp_t = torch.tensor(inp).unsqueeze(0).to(device)
tids = torch.full((1,), idx, dtype=torch.long, device=device)
with torch.no_grad():
    logits = model(inp_t, task_ids=tids)
    pred = logits[0].argmax(dim=0).cpu().numpy()

# Check the crop area
scale = max(1, (64-4)//max(h,w))
new_h, new_w = h*scale, w*scale
y_off = (64-new_h)//2
x_off = (64-new_w)//2
crop = pred[y_off:y_off+new_h, x_off:x_off+new_w]
print(f"Task {task['id']}: {h}x{w}")
print(f"scale={scale}, crop=({new_h},{new_w}) at ({y_off},{x_off})")

# Check BD in right col and bottom row
right_col = crop[:, -1]
bottom_row = crop[-1, :]
print(f"Right col BD%: {(right_col==BD_CLASS).mean():.2%}, unique:{np.unique(right_col)}")
print(f"Bottom row BD%: {(bottom_row==BD_CLASS).mean():.2%}, unique:{np.unique(bottom_row)}")

# The extract_grid result
result = extract_grid_from_canvas(pred, h, w)
print(f"Extracted shape: {result.shape}")

expected = test_out.tolist()
pred_list = result.tolist()
print(f"\nExpected first 2 rows: {expected[:2]}")
print(f"Predicted first 2 rows: {pred_list[:2]}")
print(f"Expected last 2 rows: {expected[-2:]}")
print(f"Predicted last 2 rows: {pred_list[-2:]}")

if len(pred_list) == len(expected) and len(pred_list[0]) == len(expected[0]):
    diff = sum(1 for i in range(h) for j in range(w) if pred_list[i][j] != expected[i][j])
    print(f"Diff: {diff}/{h*w}")
