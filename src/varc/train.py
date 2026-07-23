import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.varc.data import ARCDataset, BG_CLASS
from src.varc.model import VARCModel, VARCConfig


def train_epoch(
    model: VARCModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    warmup_epochs: int = 10,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=BG_CLASS)
    for batch in dataloader:
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
        torch.cuda.empty_cache()
    return total_loss / len(dataloader)


def train_model(
    model: VARCModel,
    train_dataset: ARCDataset,
    val_dataset: ARCDataset | None = None,
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 3e-4,
    warmup_epochs: int = 10,
    device: torch.device = torch.device("cuda"),
    checkpoint_path: str = "results/checkpoints/varc_best.pt",
    log_interval: int = 10,
) -> dict:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        if epoch <= warmup_epochs:
            for g in optimizer.param_groups:
                g["lr"] = lr * epoch / warmup_epochs

        loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        train_loss = train_epoch(model, loader, optimizer, device, epoch, warmup_epochs)
        history["train_loss"].append(train_loss)

        if epoch > warmup_epochs:
            scheduler.step()

        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=1)
            val_loss = 0.0
            model.eval()
            criterion = nn.CrossEntropyLoss(ignore_index=BG_CLASS)
            with torch.no_grad():
                for batch in val_loader:
                    ti = batch["train_input"].to(device)
                    to = batch["train_target"].to(device)
                    task_ids = batch.get("task_idx")
                    B, N = ti.shape[0], ti.shape[1]
                    ti = ti.view(B * N, *ti.shape[2:])
                    to = to.view(B * N, *to.shape[2:])
                    if task_ids is not None:
                        task_ids = task_ids.to(device).view(B, 1).expand(B, N).reshape(B * N)
                    val_loss += criterion(model(ti, task_ids=task_ids), to).item()
            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)

        if epoch == 1 or epoch % log_interval == 0 or epoch == num_epochs:
            val_str = f", val_loss={history['val_loss'][-1]:.4f}" if val_dataset else ""
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{num_epochs}  lr={lr_now:.2e}  train_loss={train_loss:.4f}{val_str}")

    return history


def ttt_finetune(
    model: VARCModel,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    device: torch.device,
    steps: int = 100,
    lr: float = 3e-4,
    task_id: int | None = None,
) -> VARCModel:
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss(ignore_index=BG_CLASS)
    tids = None
    if task_id is not None:
        tids = torch.full((train_inputs.shape[0],), task_id, dtype=torch.long, device=device)
    for _ in range(steps):
        pred = model(train_inputs.to(device), task_ids=tids)
        loss = criterion(pred, train_targets.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def save_checkpoint(model: VARCModel, path: str) -> str:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": model.config}, path)
    return path


def load_checkpoint(model: VARCModel, path: str, device: torch.device) -> VARCModel:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model
