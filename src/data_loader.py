import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.config import ARC_AGI1_TRAINING_DIR, ARC_AGI1_EVALUATION_DIR

Split = Literal["training", "evaluation"]
Source = Literal["arc-agi-1"]


def _get_split_dir(source: Source, split: Split) -> Path:
    if source == "arc-agi-1":
        return ARC_AGI1_TRAINING_DIR if split == "training" else ARC_AGI1_EVALUATION_DIR
    msg = f"Unknown source: {source}"
    raise ValueError(msg)


def load_task_from_file(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def list_task_ids(source: Source = "arc-agi-1", split: Split | None = None) -> list[str]:
    if split is not None:
        paths = sorted(_get_split_dir(source, split).glob("*.json"))
        return [p.stem for p in paths]
    all_ids = []
    for s in ("training", "evaluation"):
        all_ids.extend(list_task_ids(source, s))
    return all_ids


def load_task(
    task_id: str,
    source: Source = "arc-agi-1",
    split: Split | None = None,
) -> dict[str, Any] | None:
    if split is not None:
        path = _get_split_dir(source, split) / f"{task_id}.json"
        if path.exists():
            return {"id": task_id, "split": split, **load_task_from_file(path)}
        return None
    for s in ("training", "evaluation"):
        result = load_task(task_id, source, s)
        if result is not None:
            return result
    return None


def load_tasks(
    source: Source = "arc-agi-1",
    split: Split = "training",
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    task_dir = _get_split_dir(source, split)
    paths = sorted(task_dir.glob("*.json"))
    if max_tasks is not None:
        paths = paths[:max_tasks]
    tasks = []
    for path in paths:
        tasks.append({"id": path.stem, "split": split, **load_task_from_file(path)})
    return tasks


def grid_to_array(grid: list[list[int]]) -> np.ndarray:
    return np.array(grid, dtype=np.int32)


def task_to_arrays(task: dict[str, Any]) -> dict[str, Any]:
    result = {"id": task["id"], "split": task["split"], "train": [], "test": []}
    for pair in task["train"]:
        result["train"].append(
            {
                "input": grid_to_array(pair["input"]),
                "output": grid_to_array(pair["output"]),
            }
        )
    for pair in task["test"]:
        entry: dict[str, Any] = {"input": grid_to_array(pair["input"])}
        if "output" in pair:
            entry["output"] = grid_to_array(pair["output"])
        result["test"].append(entry)
    return result
