import random
from typing import Any

import numpy as np

BG_COLOR = 0


def find_objects(grid: np.ndarray) -> list[dict[str, Any]]:
    non_bg = grid != BG_COLOR
    h, w = grid.shape
    parent = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra < rb:
            parent[rb] = ra
        elif rb < ra:
            parent[ra] = rb
    label = np.zeros_like(grid, dtype=np.int32)
    next_label = 1
    for y in range(h):
        for x in range(w):
            if not non_bg[y, x]:
                continue
            up = label[y-1, x] if y > 0 else 0
            left = label[y, x-1] if x > 0 else 0
            if up == 0 and left == 0:
                label[y, x] = next_label
                parent[next_label] = next_label
                next_label += 1
            elif up == 0:
                label[y, x] = left
            elif left == 0:
                label[y, x] = up
            else:
                label[y, x] = min(up, left)
                if up != left:
                    union(up, left)
    for y in range(h):
        for x in range(w):
            if label[y, x] > 0:
                label[y, x] = find(label[y, x])
    _, inverse = np.unique(label, return_inverse=True)
    label = inverse.reshape(h, w)
    num_features = label.max()
    objects = []
    for obj_id in range(1, num_features + 1):
        mask = label == obj_id
        ys, xs = np.where(mask)
        obj = {
            "mask": mask,
            "pixels": grid * mask,
            "colors": np.unique(grid[mask]).tolist(),
            "y1": int(ys.min()),
            "y2": int(ys.max()),
            "x1": int(xs.min()),
            "x2": int(xs.max()),
            "h": int(ys.max() - ys.min() + 1),
            "w": int(xs.max() - xs.min() + 1),
            "cy": float(ys.mean()),
            "cx": float(xs.mean()),
        }
        objects.append(obj)
    return objects


def paste_object(canvas: np.ndarray, mask: np.ndarray, pixels: np.ndarray, y: int, x: int) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return canvas
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    obj_h = y2 - y1 + 1
    obj_w = x2 - x1 + 1
    H, W = canvas.shape
    src_y1 = max(0, -y)
    src_x1 = max(0, -x)
    dst_y1 = max(0, y)
    dst_x1 = max(0, x)
    copy_h = min(obj_h - src_y1, H - dst_y1)
    copy_w = min(obj_w - src_x1, W - dst_x1)
    if copy_h <= 0 or copy_w <= 0:
        return canvas
    mask_src = mask[y1+src_y1:y1+src_y1+copy_h, x1+src_x1:x1+src_x1+copy_w]
    pix_src = pixels[y1+src_y1:y1+src_y1+copy_h, x1+src_x1:x1+src_x1+copy_w]
    dst = canvas[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w]
    canvas[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = np.where(mask_src, pix_src, dst)
    return canvas


def transform_color_replace(pair: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    in_grid = np.array(pair["input"])
    out_grid = np.array(pair["output"])
    all_colors = sorted(set(in_grid.flatten().tolist() + out_grid.flatten().tolist()) - {BG_COLOR})
    if len(all_colors) < 2:
        return pair
    src_color = rng.choice(all_colors)
    dst_color = rng.choice([c for c in range(1, 10) if c != src_color])
    in_grid[in_grid == src_color] = dst_color
    out_grid[out_grid == src_color] = dst_color
    return {"input": in_grid, "output": out_grid}


def transform_noise(pair: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    in_grid = np.array(pair["input"])
    out_grid = np.array(pair["output"])
    h, w = in_grid.shape
    noise_count = max(1, h * w // 20)
    for _ in range(rng.randint(0, noise_count)):
        y, x = rng.randint(0, h - 1), rng.randint(0, w - 1)
        in_grid[y, x] = rng.randint(0, 9)
        if y < out_grid.shape[0] and x < out_grid.shape[1]:
            out_grid[y, x] = rng.randint(0, 9)
    return {"input": in_grid, "output": out_grid}


def transform_flip_rotate(pair: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    in_grid = np.array(pair["input"])
    out_grid = np.array(pair["output"])
    ops = [
        lambda g: g,
        lambda g: np.rot90(g, k=1),
        lambda g: np.rot90(g, k=2),
        lambda g: np.rot90(g, k=3),
        lambda g: np.fliplr(g),
        lambda g: np.flipud(g),
    ]
    op = rng.choice(ops)
    return {"input": op(in_grid), "output": op(out_grid)}


def transform_move_object(pair: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    in_grid = np.array(pair["input"])
    out_grid = np.array(pair["output"])
    in_objs = find_objects(in_grid)
    out_objs = find_objects(out_grid)
    if len(in_objs) == 0 or len(out_objs) == 0:
        return pair
    target_obj = rng.choice(out_objs)
    dy = rng.randint(-3, 3)
    dx = rng.randint(-3, 3)
    if dy == 0 and dx == 0:
        dy, dx = 1, 1
    new_out = np.full_like(out_grid, BG_COLOR)
    for obj in out_objs:
        y = obj["y1"] + (dy if obj is target_obj else 0)
        x = obj["x1"] + (dx if obj is target_obj else 0)
        paste_object(new_out, obj["mask"], obj["pixels"], y, x)
    new_in = in_grid.copy()
    if len(in_objs) > 0:
        closest = min(in_objs, key=lambda o: abs(o["cy"] - target_obj["cy"]) + abs(o["cx"] - target_obj["cx"]))
        new_in = np.full_like(in_grid, BG_COLOR)
        for obj in in_objs:
            y = obj["y1"] + (dy if obj is closest else 0)
            x = obj["x1"] + (dx if obj is closest else 0)
            paste_object(new_in, obj["mask"], obj["pixels"], y, x)
    return {"input": new_in, "output": new_out}


def transform_copy_object(pair: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    in_grid = np.array(pair["input"])
    out_grid = np.array(pair["output"])
    in_objs = find_objects(in_grid)
    out_objs = find_objects(out_grid)
    if len(in_objs) == 0 or len(out_objs) == 0:
        return pair
    obj = rng.choice(out_objs)
    dy = rng.randint(0, out_grid.shape[0] - obj["y2"] - 1)
    dx = rng.randint(0, out_grid.shape[1] - obj["x2"] - 1)
    if dy == 0 and dx == 0:
        dy = max(1, out_grid.shape[0] // 4)
    new_out = out_grid.copy()
    paste_object(new_out, obj["mask"], obj["pixels"], obj["y1"] + dy, obj["x1"] + dx)
    new_in = in_grid.copy()
    if len(in_objs) > 0:
        closest = min(in_objs, key=lambda o: abs(o["cy"] - obj["cy"]) + abs(o["cx"] - obj["cx"]))
        paste_object(new_in, closest["mask"], closest["pixels"], closest["y1"] + dy, closest["x1"] + dx)
    return {"input": new_in, "output": new_out}


def transform_scale_object(pair: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    from PIL import Image
    in_grid = np.array(pair["input"])
    out_grid = np.array(pair["output"])
    in_objs = find_objects(in_grid)
    out_objs = find_objects(out_grid)
    if len(in_objs) == 0 or len(out_objs) == 0:
        return pair
    obj = rng.choice(out_objs)
    scale = rng.choice([0.5, 2.0])
    new_h = max(1, int(obj["h"] * scale))
    new_w = max(1, int(obj["w"] * scale))
    crop = out_grid[obj["y1"]:obj["y2"] + 1, obj["x1"]:obj["x2"] + 1]
    scaled = np.array(Image.fromarray(crop.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST))
    new_out = out_grid.copy()
    y_end_out = min(obj["y1"] + new_h, new_out.shape[0])
    x_end_out = min(obj["x1"] + new_w, new_out.shape[1])
    ch_out = y_end_out - obj["y1"]
    cw_out = x_end_out - obj["x1"]
    new_out[obj["y1"]:y_end_out, obj["x1"]:x_end_out] = scaled[:ch_out, :cw_out]
    new_in = in_grid.copy()
    if len(in_objs) > 0:
        closest = min(in_objs, key=lambda o: abs(o["cy"] - obj["cy"]) + abs(o["cx"] - obj["cx"]))
        crop_in = in_grid[closest["y1"]:closest["y2"] + 1, closest["x1"]:closest["x2"] + 1]
        scaled_in = np.array(Image.fromarray(crop_in.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST))
        y_end_in = min(closest["y1"] + ch_out, new_in.shape[0])
        x_end_in = min(closest["x1"] + cw_out, new_in.shape[1])
        new_in[closest["y1"]:y_end_in, closest["x1"]:x_end_in] = scaled_in[:y_end_in - closest["y1"], :x_end_in - closest["x1"]]
    return {"input": new_in, "output": new_out}


TRANSFORMS = [
    ("color_replace", transform_color_replace),
    ("noise", transform_noise),
    ("flip_rotate", transform_flip_rotate),
    ("move_object", transform_move_object),
    ("copy_object", transform_copy_object),
    ("scale_object", transform_scale_object),
]


def augment_pair_rearc(pair: dict[str, Any], rng: random.Random, num_transforms: int = 1) -> dict[str, Any]:
    result = {"input": np.array(pair["input"]), "output": np.array(pair["output"])}
    chosen = rng.sample(TRANSFORMS, min(num_transforms, len(TRANSFORMS)))
    for name, fn in chosen:
        result = fn(result, rng)
    return result


def augment_task_rearc(task: dict[str, Any], rng: random.Random, augment_prob: float = 0.8) -> dict[str, Any]:
    out: dict[str, Any] = {"id": task["id"], "split": task.get("split", "training"), "train": [], "test": []}
    for pair in task["train"]:
        if rng.random() < augment_prob:
            pair = augment_pair_rearc(pair, rng, num_transforms=rng.randint(1, 3))
        out["train"].append(pair)
    for pair in task["test"]:
        out["test"].append({"input": pair["input"]})
        if "output" in pair:
            out["test"][-1]["output"] = pair["output"]
    return out
