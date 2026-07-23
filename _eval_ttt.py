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
print(f"Evaluating {len(tasks)} tasks...")

print("=== BASE (no TTT) ===")
results_base = evaluate_on_tasks(model, tasks, device, use_ttt=False)
acc_base = compute_accuracy(results_base)
print(f"Accuracy BASE: {acc_base}")

print("\n=== TTT (100 steps) ===")
results_ttt = evaluate_on_tasks(model, tasks, device, use_ttt=True, ttt_steps=100, ttt_lr=3e-4)
acc_ttt = compute_accuracy(results_ttt)
print(f"Accuracy TTT: {acc_ttt}")

print("\nDetailed comparison:")
for rb, rt in zip(results_base, results_ttt):
    base_ok = rb['correct']
    ttt_ok = rt['correct']
    if base_ok or ttt_ok:
        print(f"  {rb['task_id']}: BASE={'+' if base_ok else '-'} TTT={'+' if ttt_ok else '-'}")
        if not base_ok:
            exp = rb['expected'][0]
            pred_base = rb['predicted'][0]
            pred_ttt = rt['predicted'][0]
            print(f"    exp={exp} base={pred_base[:len(exp)]} ttt={pred_ttt[:len(exp)]}")
