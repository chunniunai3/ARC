import sys; sys.path.insert(0, '.')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.varc import (VARCConfig, VARCModel, ARCDataset, load_checkpoint,
                      BG_CLASS)
from pathlib import Path
import math
import collections.abc as abc

def rearc_collate(batch):
    train_in_list, train_target_list, task_idx_list = [], [], []
    for b in batch:
        n = b['train_input'].shape[0]
        train_in_list.append(b['train_input'])
        train_target_list.append(b['train_target'])
        task_idx_list.append(torch.tensor(b['task_idx']).expand(n))
    return {
        'train_input': torch.cat(train_in_list, dim=0),
        'train_target': torch.cat(train_target_list, dim=0),
        'task_idx': torch.cat(task_idx_list, dim=0),
    }

device = torch.device('cuda')
config = VARCConfig()
model = VARCModel(config).to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

base_ckpt = 'results/checkpoints/varc_exp3_base.pt'
print("Loading base checkpoint...")
load_checkpoint(model, base_ckpt, device)
model.train()

start_epoch = 1
num_epochs = 50

train_dataset = ARCDataset(split='training', max_tasks=400, rearc_prob=0.5)
val_dataset = ARCDataset(split='training', max_tasks=40, rearc_prob=0.0)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss(ignore_index=BG_CLASS)
T_max = num_epochs
batch_size = 2

for epoch in range(start_epoch, num_epochs + 1):
    if epoch <= 5:
        lr = 3e-4 * epoch / 5
    else:
        progress = (epoch - 5) / T_max
        lr = 3e-4 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr

    model.train()
    total_loss = 0.0
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=rearc_collate)
    for batch in loader:
        train_in = batch["train_input"].to(device)
        train_target = batch["train_target"].to(device)
        task_ids = batch.get("task_idx")
        if task_ids is not None:
            task_ids = task_ids.to(device)
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
            if tids is not None:
                tids = tids.to(device)
            val_loss += criterion(model(ti, task_ids=tids), to).item()
    val_loss /= len(val_loader)

    print(f"  Epoch {epoch:3d}/{num_epochs}  lr={lr:.2e}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
    torch.save(model.state_dict(), 'results/checkpoints/varc_rearc_ft.pt')

print("Done!")
