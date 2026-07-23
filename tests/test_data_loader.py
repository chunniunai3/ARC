import json
from pathlib import Path

import numpy as np

from src.data_loader import (
    load_task_from_file,
    load_task,
    load_tasks,
    list_task_ids,
    grid_to_array,
)


def test_load_task_from_file(tmp_path: Path) -> None:
    task_data = {"train": [], "test": []}
    f = tmp_path / "abc123.json"
    f.write_text(json.dumps(task_data))
    result = load_task_from_file(f)
    assert result == task_data


def test_load_task_found(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.data_loader.ARC_AGI1_TRAINING_DIR",
        tmp_path / "training",
    )
    monkeypatch.setattr(
        "src.data_loader.ARC_AGI1_EVALUATION_DIR",
        tmp_path / "evaluation",
    )
    train_dir = tmp_path / "training"
    train_dir.mkdir()
    task = {"train": [{"input": [[0]], "output": [[1]]}], "test": [{"input": [[0]]}]}
    (train_dir / "test_id.json").write_text(json.dumps(task))
    result = load_task("test_id")
    assert result is not None
    assert result["id"] == "test_id"
    assert result["split"] == "training"


def test_load_task_not_found(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.data_loader.ARC_AGI1_TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr("src.data_loader.ARC_AGI1_EVALUATION_DIR", tmp_path / "evaluation")
    result = load_task("nonexistent")
    assert result is None


def test_list_task_ids(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.data_loader.ARC_AGI1_TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr("src.data_loader.ARC_AGI1_EVALUATION_DIR", tmp_path / "evaluation")
    d = tmp_path / "training"
    d.mkdir()
    for name in ["a.json", "b.json", "c.json"]:
        (d / name).write_text("{}")
    ids = list_task_ids(split="training")
    assert ids == ["a", "b", "c"]


def test_grid_to_array() -> None:
    grid = [[0, 1], [2, 3]]
    arr = grid_to_array(grid)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)
    assert arr.dtype == np.int32
    assert arr[0, 0] == 0
    assert arr[1, 1] == 3


def test_load_tasks_max(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.data_loader.ARC_AGI1_TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr("src.data_loader.ARC_AGI1_EVALUATION_DIR", tmp_path / "evaluation")
    d = tmp_path / "training"
    d.mkdir()
    for i in range(5):
        (d / f"task{i}.json").write_text('{"train": [], "test": []}')
    tasks = load_tasks(max_tasks=3)
    assert len(tasks) == 3
