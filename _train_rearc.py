import sys; sys.path.insert(0, '.')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.varc import (VARCConfig, VARCModel, ARCDataset, save_checkpoint,
                      load_checkpoint, train_epoch, BG_CLASS)
from pathlib import Path
import math

device = torch.device('cuda')
config = VARCConfig()
model = VARCModel(config).to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

ckpt_path = 'results/checkpoints/varc_rearc.pt'
resume = Path(ckpt_path).exists()

if resume:
    print("Resuming from checkpoint...")
    load_checkpoint(model, ckpt_path, device)
    start_epoch = 11  # was at epoch 10, resume from 11
else:
    print("Training from scratch...")
    start_epoch = 1

train_dataset = ARCDataset(split='training', max_tasks=400, rearc_prob=1.0)
val_dataset = ARCDataset(split='training', max_tasks=40, rearc_prob=0.0)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss(ignore_index=BG_CLASS)
T_max = 50
num_epochs = 50

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
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
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
    val_loader = DataLoader(val_dataset, batch_size=1)
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
    torch.save(model.state_dict(), ckpt_path)

print("Done!")
