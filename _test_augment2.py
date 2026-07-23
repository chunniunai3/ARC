import sys; sys.path.insert(0, '.')
import random
import numpy as np
from src.data_loader import load_tasks
from src.varc.augment import *

tasks = load_tasks(split='training', max_tasks=10)
for ti in range(min(10, len(tasks))):
    task = tasks[ti]
    tid = task["id"]
    print(f"Task {ti}: {tid}")
    for pi, pair in enumerate(task["train"][:1]):
        inp = np.array(pair["input"])
        out = np.array(pair["output"])
        print(f"  train pair: inp={inp.shape} out={out.shape}")
        rng = random.Random(ti * 10 + pi)
        aug = augment_pair_rearc(pair, rng, num_transforms=2)
        changed_in = not np.array_equal(aug["input"], inp)
        changed_out = not np.array_equal(aug["output"], out)
        changed = "CHANGED" if (changed_in or changed_out) else "IDENTITY"
        print(f"  -> {changed} inp_shape={np.array(aug['input']).shape} out_shape={np.array(aug['output']).shape}")
