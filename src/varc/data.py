import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_loader import load_tasks, task_to_arrays
from src.varc.augment import augment_task_rearc


COLOR_MAP = np.array([
    [0, 0, 0],       # 0 black
    [0, 0, 255],     # 1 blue
    [255, 0, 0],     # 2 red
    [0, 255, 0],     # 3 green
    [255, 255, 0],   # 4 yellow
    [128, 128, 128], # 5 gray
    [255, 0, 255],   # 6 magenta
    [255, 165, 0],   # 7 orange
    [0, 255, 255],   # 8 cyan
    [165, 42, 42],   # 9 brown
], dtype=np.float32)

DEFAULT_CANVAS_SIZE = 64
BG_CLASS = 10
BD_CLASS = 11
NUM_CLASSES = 12


def grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for color_idx in range(10):
        mask = grid == color_idx
        rgb[mask] = COLOR_MAP[color_idx]
    return rgb


def _scale_grid(grid: np.ndarray, scale: int) -> np.ndarray:
    h, w = grid.shape
    if scale == 1:
        return grid
    scaled = np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)
    return scaled


def place_on_canvas_input(
    grid: np.ndarray,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    scale: int | None = None,
    offset: tuple[int, int] | None = None,
) -> tuple[np.ndarray, int, int, int, int]:
    h, w = grid.shape
    if scale is None:
        max_s = (canvas_size - 4) // max(h, w)
        scale = max(1, max_s)
    new_h, new_w = h * scale, w * scale
    grid_scaled = _scale_grid(grid, scale)
    canvas = np.full((canvas_size, canvas_size), BG_CLASS, dtype=np.int64)
    if offset is not None:
        y_off, x_off = offset
    else:
        y_off = (canvas_size - new_h) // 2
        x_off = (canvas_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = grid_scaled
    return canvas, new_h, new_w, y_off, x_off


def place_on_canvas_target(
    grid: np.ndarray,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    new_h: int | None = None,
    new_w: int | None = None,
    y_off: int | None = None,
    x_off: int | None = None,
) -> np.ndarray:
    h, w = grid.shape
    if new_h is None or new_w is None:
        scale = min((canvas_size - 4) // h, (canvas_size - 4) // w, 10)
        scale = max(1, scale)
        new_h, new_w = h * scale, w * scale
    grid_scaled = _scale_grid(grid, new_h // h) if new_h // h == new_w // w and new_h // h >= 1 else grid
    if new_h != h or new_w != w:
        from PIL import Image
        img = Image.fromarray(grid.astype(np.uint8))
        grid_scaled = np.array(img.resize((new_w, new_h), Image.NEAREST)).astype(np.int64)
    canvas = np.full((canvas_size, canvas_size), BG_CLASS, dtype=np.int64)
    if y_off is None or x_off is None:
        y_off = (canvas_size - new_h) // 2
        x_off = (canvas_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = grid_scaled
    canvas[y_off:y_off + new_h, x_off + new_w - 1] = BD_CLASS
    canvas[y_off + new_h - 1, x_off:x_off + new_w] = BD_CLASS
    return canvas


def task_to_canvas(
    task: dict[str, Any],
    canvas_size: int = DEFAULT_CANVAS_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_inputs, train_targets, test_inputs, test_targets = [], [], [], []
    train_scales_offsets = []
    for pair in task["train"]:
        inp, inh, inw, iy, ix = place_on_canvas_input(pair["input"], canvas_size)
        train_inputs.append(inp)
        out = place_on_canvas_target(pair["output"], canvas_size, inh, inw, iy, ix)
        train_targets.append(out)
        train_scales_offsets.append((inh, inw, iy, ix))
    for pair in task["test"]:
        inp, inh, inw, iy, ix = place_on_canvas_input(pair["input"], canvas_size)
        test_inputs.append(inp)
        if "output" in pair:
            out = place_on_canvas_target(pair["output"], canvas_size, inh, inw, iy, ix)
            test_targets.append(out)
    return (
        torch.tensor(np.stack(train_inputs)),
        torch.tensor(np.stack(train_targets)),
        torch.tensor(np.stack(test_inputs)),
        torch.tensor(np.stack(test_targets)) if test_targets else torch.empty(0, dtype=torch.long),
        train_scales_offsets,
    )


AUG_IDENTITY = 0
AUG_ROT90 = 1
AUG_ROT180 = 2
AUG_ROT270 = 3
AUG_FLIP_H = 4
AUG_FLIP_V = 5
AUG_ROT90_FLIP_H = 6
AUG_ROT90_FLIP_V = 7
NUM_AUGS = 8


def augment_grid(grid: np.ndarray, aug_idx: int) -> np.ndarray:
    if aug_idx == AUG_IDENTITY:
        return grid
    elif aug_idx == AUG_ROT90:
        return np.rot90(grid, k=1)
    elif aug_idx == AUG_ROT180:
        return np.rot90(grid, k=2)
    elif aug_idx == AUG_ROT270:
        return np.rot90(grid, k=3)
    elif aug_idx == AUG_FLIP_H:
        return np.fliplr(grid)
    elif aug_idx == AUG_FLIP_V:
        return np.flipud(grid)
    elif aug_idx == AUG_ROT90_FLIP_H:
        return np.fliplr(np.rot90(grid, k=1))
    elif aug_idx == AUG_ROT90_FLIP_V:
        return np.flipud(np.rot90(grid, k=1))
    return grid


def augment_task(task: dict[str, Any], aug_idx: int) -> dict[str, Any]:
    out: dict[str, Any] = {"id": task["id"], "split": task.get("split", "training"), "train": [], "test": []}
    for pair in task["train"]:
        out["train"].append({
            "input": augment_grid(pair["input"], aug_idx),
            "output": augment_grid(pair["output"], aug_idx),
        })
    for pair in task["test"]:
        entry = {"input": augment_grid(pair["input"], aug_idx)}
        if "output" in pair:
            entry["output"] = augment_grid(pair["output"], aug_idx)
        out["test"].append(entry)
    return out


class ARCDataset(Dataset):
    def __init__(
        self,
        split: str = "training",
        canvas_size: int = DEFAULT_CANVAS_SIZE,
        max_tasks: int | None = None,
        source: str = "arc-agi-1",
        rearc_prob: float = 1.0,
    ):
        self.canvas_size = canvas_size
        self.rearc_prob = rearc_prob
        raw_tasks = load_tasks(source, split, max_tasks)
        self.tasks = [task_to_arrays(t) for t in raw_tasks]

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        task = self.tasks[idx]
        rng = random.Random(random.randint(0, 2**31))
        if rng.random() < self.rearc_prob:
            aug_task = augment_task_rearc(task, rng, augment_prob=0.8)
        else:
            aug_task = task
        train_in, train_target, test_in, test_target, _ = task_to_canvas(aug_task, self.canvas_size)
        return {
            "train_input": train_in,
            "train_target": train_target,
            "test_input": test_in,
            "test_target": test_target,
            "task_id": task["id"],
            "task_idx": idx,
        }
