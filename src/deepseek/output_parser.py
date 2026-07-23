import json
from typing import Any

import numpy as np


def _balanced_bracket_indices(text: str) -> list[tuple[int, int]]:
    results = []
    i = 0
    while i < len(text):
        if text[i] != "[":
            i += 1
            continue
        depth = 0
        start = i
        while i < len(text):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    results.append((start, i + 1))
                    break
            i += 1
        i += 1
    return results


def find_all_json_arrays(text: str) -> list[list[list[int]]]:
    candidates = []
    for start, end in _balanced_bracket_indices(text):
        chunk = text[start:end]
        try:
            data = json.loads(chunk)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                candidates.append((end - start, start, data))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    return [c[2] for c in candidates]


def parse_from_json_mode(response: str) -> np.ndarray | None:
    try:
        obj = json.loads(response)
        if not isinstance(obj, dict):
            return None
        grid = obj.get("output_grid") or obj.get("grid") or obj.get("output")
        if grid is None:
            for v in obj.values():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                    grid = v
                    break
        if grid is None:
            return None
        arr = np.array(grid, dtype=np.int32)
        if arr.ndim == 2 and arr.size > 0:
            return arr
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        return None
    return None


def parse_output_grid(response: str) -> np.ndarray | None:
    result = parse_from_json_mode(response)
    if result is not None:
        return result
    candidates = find_all_json_arrays(response)
    for data in candidates:
        try:
            arr = np.array(data, dtype=np.int32)
            if arr.ndim == 2 and arr.size > 0:
                return arr
        except (ValueError, TypeError):
            continue
    return None


def grids_match(predicted: np.ndarray, expected: np.ndarray) -> bool:
    if predicted.shape != expected.shape:
        return False
    return bool(np.all(predicted == expected))


def evaluate_predictions(predictions: list[dict[str, Any]]) -> dict[str, float]:
    correct = sum(1 for p in predictions if p["correct"])
    total = len(predictions)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }
