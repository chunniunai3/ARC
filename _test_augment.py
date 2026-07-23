import sys; sys.path.insert(0, '.')
import random
import numpy as np
from src.data_loader import load_tasks
from src.varc.augment import *

tasks = load_tasks(split='training', max_tasks=5)
task = tasks[0]
task_id = task["id"]
print(f"Task: {task_id}")
pair = task["train"][0]
inp, out = np.array(pair["input"]), np.array(pair["output"])
print(f"Input: {inp.shape}\n{inp}")
print(f"Output: {out.shape}\n{out}")

rng = random.Random(42)
for name, fn in TRANSFORMS:
    result = fn({"input": inp, "output": out}, rng)
    print(f"\n{name}:")
    print(f"  Input:\n{result['input']}")
    print(f"  Output:\n{result['output']}")
