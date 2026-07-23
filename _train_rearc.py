import sys; sys.path.insert(0, '.')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.varc import (VARCConfig, VARCModel, ARCDataset, save_checkpoint,
                      load_checkpoint, BG_CLASS, BD_CLASS)
from pathlib import Path
import math
import collections.abc as abc

def rearc_collate(batch):
    max_pairs = max(b['train_input'].shape[0] for b in batch)
    train_in, train_target, task_idx = [], [], []
    for b in batch:
        n = b['train_input'].shape[0]
        ti = b['train_input']
        tt = b['train_target']
        if n < max_pairs:
            pad = max_pairs - n
            pad_ti = torch.full((pad, *ti.shape[1:]), BG_CLASS, dtype=ti.dtype)
            pad_tt = torch.full((pad, *tt.shape[1:]), BG_CLASS, dtype=tt.dtype)
            ti = torch.cat([ti, pad_ti], dim=0)
            tt = torch.cat([tt, pad_tt], dim=0)
        train_in.append(ti.unsqueeze(0))
        train_target.append(tt.unsqueeze(0))
        task_idx.append(torch.tensor([[b['task_idx']]]))
    return {
        'train_input': torch.cat(train_in, dim=0),
        'train_target': torch.cat(train_target, dim=0),
        'task_idx': torch.cat(task_idx, dim=0),
    }

device = torch.device('cuda')
config = VARCConfig()
model = VARCModel(config).to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

ckpt_path = 'results/checkpoints/varc_rearc.pt'
resume = Path(ckpt_path).exists()

if resume:
    print("Resuming from checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'epoch' in ckpt:
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed from epoch {ckpt['epoch']}")
    else:
        model.load_state_dict(ckpt)
        start_epoch = 14
        print("  Loaded state_dict (epoch unknown, starting from 14)")
else:
    print("Training from scratch...")
    start_epoch = 1

train_dataset = ARCDataset(split='training', max_tasks=400, rearc_prob=1.0)
val_dataset = ARCDataset(split='training', max_tasks=40, rearc_prob=0.0)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss(ignore_index=BG_CLASS)
T_max = 100
num_epochs = 100

batch_size = 32

for epoch in range(start_epoch, num_epochs + 1):
    if epoch <= 5:
        lr = 1e-3 * epoch / 5
    else:
        progress = (epoch - 5) / T_max
        lr = 1e-3 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr

    model.train()
    total_loss = 0.0
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=rearc_collate)
    for batch in loader:
        train_in = batch["train_input"].to(device)
        train_target = batch["train_target"].to(device)
        task_ids = batch.get("task_idx")
        B, n_pairs = train_in.shape[0], train_in.shape[1]
        train_in = train_in.view(B * n_pairs, *train_in.shape[2:])
        train_target = train_target.view(B * n_pairs, *train_target.shape[2:])
        if task_ids is not None:
            task_ids = task_ids.to(device).view(B, 1).expand(B, n_pairs).reshape(B * n_pairs)
        pred = model(train_in, task_ids=task_ids)
        loss = criterion(pred, train_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(loader)

    model.eval()
    val_loss = 0.0
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=rearc_collate)
    with torch.no_grad():
        for batch in val_loader:
            ti = batch["train_input"].to(device)
            to = batch["train_target"].to(device)
            tids = batch.get("task_idx")
            B, N = ti.shape[0], ti.shape[1]
            ti = ti.view(B * N, *ti.shape[2:])
            to = to.view(B * N, *to.shape[2:])
            if tids is not None:
                tids = tids.to(device).view(B, 1).expand(B, N).reshape(B * N)
            val_loss += criterion(model(ti, task_ids=tids), to).item()
    val_loss /= len(val_loader)

    print(f"  Epoch {epoch:3d}/{num_epochs}  lr={lr:.2e}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
    torch.save({'epoch': epoch, 'model': model.state_dict()}, ckpt_path)

print("Done!")
