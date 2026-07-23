import sys; sys.path.insert(0, '.')
import torch
from src.varc import VARCConfig, VARCModel, train_model, save_checkpoint, load_checkpoint, ARCDataset
from pathlib import Path

device = torch.device('cuda')
config = VARCConfig()
model = VARCModel(config).to(device)
print(f'Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

ckpt_path = 'results/checkpoints/varc_exp3_base.pt'
if Path(ckpt_path).exists():
    print('Loading existing checkpoint...')
    load_checkpoint(model, ckpt_path, device)
else:
    print('Loading datasets...')
    train_dataset = ARCDataset(split='training', max_tasks=400)
    val_dataset = ARCDataset(split='training', max_tasks=40)
    print(f'Train: {len(train_dataset)} tasks, Val: {len(val_dataset)} tasks')
    print('Training...')
    train_model(model, train_dataset, val_dataset,
                num_epochs=100, batch_size=32, device=device,
                checkpoint_path=ckpt_path, log_interval=10, warmup_epochs=10)
    save_checkpoint(model, ckpt_path)

print('Done')
