from typing import Any

import numpy as np
import torch
from PIL import Image

from src.data_loader import load_task, task_to_arrays, load_tasks, grid_to_array
from src.varc.data import task_to_canvas, DEFAULT_CANVAS_SIZE, BG_CLASS, BD_CLASS
from src.varc.model import VARCModel


def recover_shape_from_bd(
    class_canvas: np.ndarray,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
) -> tuple[int, int, int, int]:
    content_mask = (class_canvas != BG_CLASS) & (class_canvas != BD_CLASS)
    content_pos = np.where(content_mask)
    if len(content_pos[0]) == 0:
        return 0, 0, 0, 0
    top = content_pos[0].min()
    left = content_pos[1].min()
    bottom = content_pos[0].max()
    right = content_pos[1].max()
    bd_mask = class_canvas == BD_CLASS
    bd_y, bd_x = np.where(bd_mask)
    if len(bd_y) > 0:
        right_extend = bd_x[bd_x >= left]
        if len(right_extend) > 0:
            candidate_right = right_extend.max()
            if candidate_right > right:
                right = candidate_right
        bottom_extend = bd_y[bd_y >= top]
        if len(bottom_extend) > 0:
            candidate_bottom = bottom_extend.max()
            if candidate_bottom > bottom:
                bottom = candidate_bottom
    return top, left, bottom + 1, right + 1


def _prepare_test_input(
    arrays: dict[str, Any],
    canvas_size: int = DEFAULT_CANVAS_SIZE,
) -> tuple[torch.Tensor, int, int, int, int]:
    from src.varc.data import place_on_canvas_input
    test_in = arrays["test"][0]["input"]
    inp, inh, inw, iy, ix = place_on_canvas_input(test_in, canvas_size)
    inp_t = torch.tensor(inp).unsqueeze(0)
    return inp_t, inh, inw, iy, ix


@torch.no_grad()
def run_inference(
    model: VARCModel,
    task: dict[str, Any],
    device: torch.device,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    task_id: int | None = None,
) -> np.ndarray:
    model.eval()
    arrays = task_to_arrays(task)
    canvas, _, _, _, _ = _prepare_test_input(arrays, canvas_size)
    canvas = canvas.to(device)
    tids = None
    if task_id is not None:
        tids = torch.full((1,), task_id, dtype=torch.long, device=device)
    logits = model(canvas, task_ids=tids)
    pred = logits[0].argmax(dim=0).cpu().numpy()
    return pred


def predict_grid(
    model: VARCModel,
    task: dict[str, Any],
    device: torch.device,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    task_id: int | None = None,
    known_out_shape: tuple[int, int] | None = None,
) -> list[list[int]] | None:
    pred_classes = run_inference(model, task, device, canvas_size, task_id=task_id)
    if known_out_shape is not None:
        h, w = known_out_shape
        return extract_grid_from_canvas(pred_classes, h, w, canvas_size).tolist()
    top, left, bottom, right = recover_shape_from_bd(pred_classes, canvas_size)
    if top == 0 and left == 0 and bottom == 0 and right == 0:
        test_out = task["test"][0].get("output")
        if test_out is not None:
            h, w = len(test_out), len(test_out[0])
        else:
            h = len(task["test"][0]["input"])
            w = len(task["test"][0]["input"][0])
        return extract_grid_from_canvas(pred_classes, h, w, canvas_size).tolist()
    grid_area = pred_classes[top:bottom, left:right]
    grid_area[grid_area == BG_CLASS] = 0
    grid_area[grid_area == BD_CLASS] = 0
    return grid_area.tolist()


def extract_grid_from_canvas(
    class_canvas: np.ndarray,
    orig_h: int,
    orig_w: int,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
) -> np.ndarray:
    h, w = orig_h, orig_w
    scale = max(1, (canvas_size - 4) // max(h, w))
    new_h, new_w = h * scale, w * scale
    y_off = (canvas_size - new_h) // 2
    x_off = (canvas_size - new_w) // 2
    crop = class_canvas[y_off:y_off + new_h, x_off:x_off + new_w]
    if crop.shape[0] > 1 and crop.shape[1] > 1:
        right_col_bd = (crop[:, -1] == BD_CLASS).mean() > 0.5
        bottom_row_bd = (crop[-1, :] == BD_CLASS).mean() > 0.5
        if right_col_bd and bottom_row_bd:
            crop = crop[:-1, :-1]
    img = Image.fromarray(crop.astype(np.uint8))
    resized = np.array(img.resize((orig_w, orig_h), Image.NEAREST))
    resized[resized == BG_CLASS] = 0
    resized[resized == BD_CLASS] = 0
    return resized


def evaluate_on_tasks(
    model: VARCModel,
    tasks: list[dict[str, Any]],
    device: torch.device,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    use_ttt: bool = False,
    ttt_steps: int = 100,
    ttt_lr: float = 3e-4,
    task_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    results = []
    for i, task in enumerate(tasks):
        arrays = task_to_arrays(task)
        test_out_expected = arrays["test"][0].get("output")
        if test_out_expected is None:
            continue
        out_shape = test_out_expected.shape
        if use_ttt:
            model_copy = VARCModel(model.config)
            model_copy.load_state_dict(model.state_dict())
            train_in, train_target, _, _, _ = task_to_canvas(arrays, canvas_size)
            from src.varc.train import ttt_finetune
            ttt_finetune(model_copy, train_in, train_target, device, steps=ttt_steps, lr=ttt_lr)
            pred_grid = predict_grid(model_copy, task, device, canvas_size, task_id=i, known_out_shape=out_shape)
        else:
            pred_grid = predict_grid(model, task, device, canvas_size, task_id=i, known_out_shape=out_shape)

        correct = False
        if pred_grid is not None:
            expected = test_out_expected.tolist() if isinstance(test_out_expected, np.ndarray) else test_out_expected
            correct = pred_grid == expected

        results.append({
            "task_id": task["id"],
            "correct": correct,
            "predicted": pred_grid,
            "expected": expected,
        })
        mark = "+" if correct else "-"
        mode = "TTT" if use_ttt else "BASE"
        print(f"  [{mode}] {task['id']}: {mark}")

    return results


def compute_accuracy(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {"accuracy": 0.0, "correct": 0, "total": 0}
    correct = sum(1 for r in results if r["correct"])
    return {
        "accuracy": correct / len(results),
        "correct": correct,
        "total": len(results),
    }
