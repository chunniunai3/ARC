import sys; sys.path.insert(0, '.')
import torch
import numpy as np
from src.varc import VARCConfig, VARCModel, load_checkpoint
from src.data_loader import load_tasks, task_to_arrays
from src.varc.evaluate import predict_grid, extract_grid_from_canvas, recover_shape_from_bd

device = torch.device('cuda')
config = VARCConfig()
model = VARCModel(config).to(device)
load_checkpoint(model, 'results/checkpoints/varc_exp3_base.pt', device)
model.eval()

tasks = load_tasks(split='training', max_tasks=50)

# Tasks that appear correct in first row
for idx in [2, 8, 12, 39, 43, 44]:
    task = tasks[idx]
    arrays = task_to_arrays(task)
    test_out = arrays["test"][0].get("output")
    if test_out is None:
        continue
    h, w = test_out.shape
    pred = predict_grid(model, task, device, 64, task_id=idx, known_out_shape=(h, w))
    expected = test_out.tolist()
    
    is_correct = pred == expected
    print(f"\n{task['id']} (idx={idx}): {'OK' if is_correct else 'XX'}")
    print(f"  Expected ({h}x{w}): {expected}")
    print(f"  Predicted ({len(pred)}x{len(pred[0])}): {pred}")
    
    if not is_correct and len(pred) == len(expected) and len(pred[0]) == len(expected[0]):
        diff = sum(1 for i in range(h) for j in range(w) if pred[i][j] != expected[i][j])
        print(f"  Same shape, {diff}/{h*w} pixels differ")
