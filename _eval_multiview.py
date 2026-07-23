import sys; sys.path.insert(0, '.')
import torch
from src.varc import VARCConfig, VARCModel, load_checkpoint, evaluate_on_tasks, compute_accuracy
from src.data_loader import load_tasks

device = torch.device('cuda')
config = VARCConfig()
model = VARCModel(config).to(device)
load_checkpoint(model, 'results/checkpoints/varc_exp3_base.pt', device)
model.eval()

tasks = load_tasks(split='training', max_tasks=50)
print(f"=== Multi-view eval (8 views) on {len(tasks)} tasks ===")
results = evaluate_on_tasks(model, tasks, device, use_ttt=False, use_multiview=True, num_views=8)
acc = compute_accuracy(results)
for r in results:
    flat = [v for row in r['predicted'] for v in row]
    az = all(v == 0 for v in flat)
    mark = "OK" if r['correct'] else "XX"
    print(f"  {r['task_id']}: {mark} all0={az}")
print(f"Accuracy: {acc}")
