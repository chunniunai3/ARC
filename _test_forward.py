import sys; sys.path.insert(0, '.')
import torch
from src.varc import VARCConfig, VARCModel, ARCDataset
from src.varc.data import BG_CLASS, BD_CLASS, NUM_CLASSES
from torch.utils.data import DataLoader

config = VARCConfig()
print(f"Config: d_model={config.d_model}, num_colors={config.num_colors}, pixel_embed_dim={config.pixel_embed_dim}")
print(f"  num_tasks={config.num_tasks}, dim_feedforward={config.dim_feedforward}")

model = VARCModel(config)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"Param keys: {[k for k,v in model.named_parameters()][:5]}...")

ds = ARCDataset(split='training', max_tasks=2)
loader = DataLoader(ds, batch_size=1)
batch = next(iter(loader))

train_in = batch['train_input']
train_target = batch['train_target']
print(f"\ntrain_in shape: {train_in.shape}, dtype: {train_in.dtype}")
print(f"train_in values: min={train_in.min()}, max={train_in.max()}")
print(f"train_target shape: {train_target.shape}, dtype: {train_target.dtype}")
print(f"train_target unique: {torch.unique(train_target)}")
print(f"BG_CLASS present in target: {(train_target == BG_CLASS).any()}")
print(f"BD_CLASS present in target: {(train_target == BD_CLASS).any()}")

B, n = train_in.shape[0], train_in.shape[1]
x = train_in.view(B * n, *train_in.shape[2:])
y = train_target.view(B * n, *train_target.shape[2:])

task_ids = batch['task_idx'].to('cpu').view(B, 1).expand(B, n).reshape(B * n)
print(f"\ntask_ids: {task_ids}")

with torch.no_grad():
    out = model(x, task_ids=task_ids)
print(f"output shape: {out.shape}")
print(f"output min/max: {out.min():.2f}/{out.max():.2f}")
print(f"output argmax unique: {torch.unique(out.argmax(dim=1))}")

criterion = torch.nn.CrossEntropyLoss(ignore_index=BG_CLASS)
loss = criterion(out, y)
print(f"loss: {loss.item():.4f}")

print("\n=== FORWARD PASS OK ===")
