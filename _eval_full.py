import sys; sys.path.insert(0, '.')
import torch
import json
from src.varc import VARCConfig, VARCModel, load_checkpoint, evaluate_on_tasks, compute_accuracy
from src.data_loader import load_tasks

device = torch.device('cuda')
config = VARCConfig()
model = VARCModel(config).to(device)
load_checkpoint(model, 'results/checkpoints/varc_exp3_base.pt', device)
model.eval()

print("=== Training set (400 tasks) ===")
train_tasks = load_tasks(split='training', max_tasks=400)
print(f"Loaded {len(train_tasks)} training tasks")
train_results = evaluate_on_tasks(model, train_tasks, device, use_ttt=False)
train_acc = compute_accuracy(train_results)
print(f"\nTrain Accuracy: {train_acc}")
with open('results/logs/exp3_varc_full_train.json', 'w') as f:
    json.dump({'accuracy': train_acc, 'results': train_results}, f)

print("\n=== Evaluation set (100 tasks) ===")
try:
    eval_tasks = load_tasks(source='arc-agi-1', split='evaluation', max_tasks=100)
    print(f"Loaded {len(eval_tasks)} evaluation tasks")
    eval_results = evaluate_on_tasks(model, eval_tasks, device, use_ttt=False)
    eval_acc = compute_accuracy(eval_results)
    print(f"\nEval Accuracy: {eval_acc}")
    with open('results/logs/exp3_varc_full_eval.json', 'w') as f:
        json.dump({'accuracy': eval_acc, 'results': eval_results}, f)
except Exception as e:
    print(f"Evaluation set failed: {e}")
