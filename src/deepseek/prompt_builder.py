import json
from typing import Any

import numpy as np

COLOR_MAP_RGB = {
    0: (0, 0, 0),
    1: (0, 120, 240),
    2: (255, 0, 0),
    3: (0, 216, 0),
    4: (255, 255, 0),
    5: (160, 160, 160),
    6: (255, 0, 255),
    7: (255, 160, 0),
    8: (0, 255, 255),
    9: (139, 69, 19),
}

COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "brown",
}

# ---------------------------------------------------------------------------
# Few-shot exemplars — complete CoT demonstrations for ARC tasks
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "train": [
            {"input": [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
             "output": [[0, 0, 0], [0, 1, 0], [0, 0, 0]]},
            {"input": [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]],
             "output": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]},
        ],
        "test_in": [[3, 0, 0], [0, 3, 0], [0, 0, 0]],
        "reasoning": "Each example replaces all red(2) pixels with blue(1). Other pixels stay unchanged. Grid dimensions stay the same.",
        "output": [[3, 0, 0], [0, 1, 0], [0, 0, 0]],
    },
    {
        "train": [
            {"input": [[1, 0], [0, 0]],
             "output": [[1, 1], [1, 1]]},
            {"input": [[0, 2], [0, 0]],
             "output": [[2, 2], [2, 2]]},
        ],
        "test_in": [[0, 0], [3, 0]],
        "reasoning": "The single non-black pixel's row and column are filled with that color. Output same dimensions as input.",
        "output": [[3, 3], [3, 3]],
    },
]


def _grid_to_text(grid: np.ndarray) -> str:
    h, w = grid.shape
    rows = [" ".join(str(int(c)) for c in row) for row in grid]
    return f"[{h}x{w}] " + " | ".join(rows)


def _format_examples(examples: list[dict]) -> str:
    parts = []
    for i, pair in enumerate(examples):
        inp = np.array(pair["input"])
        out = np.array(pair["output"])
        parts.append(f"Example {i + 1}:")
        parts.append(_grid_to_text(inp))
        parts.append("-> transforms to:")
        parts.append(_grid_to_text(out))
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt — role + constraints
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a top abstract visual reasoning expert. Your job is to analyze input-output grid examples, infer the transformation rule, and apply it to new test inputs.

Rules:
- Grid values are integers 0-9, each representing a color.
- The output grid may have different dimensions than the input.
- Be precise: every cell must be correct.

You must respond in valid JSON with this exact structure:
{
  "reasoning": "Your step-by-step reasoning",
  "output_grid": [[...]]
}
"""


def _build_few_shot_section() -> str:
    parts = ["---\n\n## Few-shot Examples\n\nBelow are example ARC tasks with step-by-step reasoning and correct outputs.\n"]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES):
        parts.append(f"### Example {i + 1}")
        parts.append(_format_examples(ex["train"]))
        inp = np.array(ex["test_in"])
        parts.append(f"Test Input:\n{_grid_to_text(inp)}")
        parts.append(f"\nReasoning: {ex['reasoning']}")
        parts.append(f"Output: {json.dumps(ex['output'])}\n")
    return "\n".join(parts)


FEW_SHOT_SECTION = _build_few_shot_section()


def build_arc_prompt(
    task: dict[str, Any],
    text_only: bool = False,
    use_few_shot: bool = True,
) -> str:
    parts = []
    parts.append("<think>\n")
    parts.append("## Task\n\nAnalyze the input-output examples and infer the transformation rule.\n")
    parts.append("## Steps\n")
    parts.append("1. Examine each example pair. Note changes in dimensions, colors, and spatial arrangement.")
    parts.append("2. Describe the rule precisely.")
    parts.append("3. Apply the rule to the test input.")
    parts.append("4. Verify: does the rule hold for all training examples?\n")
    if use_few_shot:
        parts.append(FEW_SHOT_SECTION)
    parts.append("## Task to Solve\n")
    parts.append("Now solve the following task. Follow the same reasoning pattern.\n")
    parts.append("### Examples\n")

    for i, pair in enumerate(task["train"]):
        inp = np.array(pair["input"])
        out = np.array(pair["output"])
        parts.append(f"--- Example {i + 1} ---")
        if text_only:
            h, w = inp.shape
            counts_in = {COLOR_NAMES.get(int(v), str(v)): int((inp == v).sum()) for v in np.unique(inp)}
            counts_out = {COLOR_NAMES.get(int(v), str(v)): int((out == v).sum()) for v in np.unique(out)}
            parts.append(f"Input: {h}x{w}, colors: {counts_in}")
            parts.append(f"Output: {out.shape[0]}x{out.shape[1]}, colors: {counts_out}")
        else:
            parts.append(_grid_to_text(inp))
            parts.append("->")
            parts.append(_grid_to_text(out))
        parts.append("")

    test_inp = np.array(task["test"][0]["input"])
    parts.append("--- Test Input ---")
    if text_only:
        h, w = test_inp.shape
        counts = {COLOR_NAMES.get(int(v), str(v)): int((test_inp == v).sum()) for v in np.unique(test_inp)}
        parts.append(f"{h}x{w}, colors: {counts}")
    else:
        parts.append(_grid_to_text(test_inp))

    parts.append("\nFollow the steps above. Think step by step inside <think> tags.")
    parts.append('Return valid JSON with keys "reasoning" and "output_grid".')
    return "\n".join(parts)



