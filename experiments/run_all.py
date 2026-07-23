"""Run ARC comparison experiments.

Usage:
    python -m experiments.run_all --exp 1                  # DeepSeek: structured prompt + JSON mode
    python -m experiments.run_all --exp 1 --describe       # DeepSeek: color counts only
    python -m experiments.run_all --exp 1 --no-few-shot    # DeepSeek: without few-shot examples
    python -m experiments.run_all --exp 2                  # VARC no TTT
    python -m experiments.run_all --exp 3                  # VARC with TTT
    python -m experiments.run_all --exp all                # Run all experiments
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src import RESULTS_DIR, load_tasks, grid_to_array

try:
    from src.deepseek import DeepSeekClient, build_arc_prompt, parse_output_grid, parse_from_json_mode, SYSTEM_PROMPT
except ImportError:
    DeepSeekClient = None
    build_arc_prompt = None
    parse_output_grid = None
    parse_from_json_mode = None
    SYSTEM_PROMPT = None

try:
    from src.varc import (
        VARCConfig, VARCModel,
        train_model, evaluate_on_tasks, compute_accuracy,
        save_checkpoint, load_checkpoint,
        ARCDataset,
    )
except ImportError:
    VARCConfig = None
    VARCModel = None
    train_model = None
    evaluate_on_tasks = None
    compute_accuracy = None
    save_checkpoint = None
    load_checkpoint = None
    ARCDataset = None


EXPERIMENTS = {
    "1": {"name": "deepseek_text_only", "method": "deepseek"},
    "2": {"name": "varc_no_ttt", "method": "varc"},
    "3": {"name": "varc_with_ttt", "method": "varc"},
}


def run_exp1(tasks: list[dict], text_only: bool = False, use_few_shot: bool = True) -> list[dict]:
    results = []
    client = DeepSeekClient()
    for task in tasks:
        prompt = build_arc_prompt(task, text_only=text_only, use_few_shot=use_few_shot)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        t0 = time.time()
        response = client.chat_json(messages=messages)
        elapsed = time.time() - t0
        predicted = parse_output_grid(response)
        expected = grid_to_array(task["test"][0]["output"]) if "output" in task["test"][0] else None
        correct = (
            predicted is not None and expected is not None
            and predicted.shape == expected.shape
            and bool((predicted == expected).all())
        )
        results.append({
            "task_id": task["id"],
            "correct": bool(correct) if predicted is not None else False,
            "time_s": round(elapsed, 2),
            "response_preview": response[:200],
        })
        print(f"  [Exp1] {task['id']}: {'+' if correct else '-'} ({elapsed:.1f}s)")
    return results


def run_exp2(tasks: list[dict]) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = VARCConfig()
    model = VARCModel(config)
    ckpt_path = str(RESULTS_DIR / "checkpoints" / "varc_exp2.pt")
    if Path(ckpt_path).exists():
        print("  Loading existing checkpoint...")
        load_checkpoint(model, ckpt_path, device)
    else:
        print("  Training VARC on 400 training tasks (no TTT)...")
        train_dataset = ARCDataset(split="training")
        val_dataset = ARCDataset(split="training", max_tasks=40)
        train_model(model, train_dataset, val_dataset,
                    num_epochs=50, batch_size=16, device=device,
                    checkpoint_path=ckpt_path)
    print("  Evaluating on eval tasks...")
    results = evaluate_on_tasks(model, tasks, device, use_ttt=False)
    return results


def run_exp3(tasks: list[dict]) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = VARCConfig()
    model = VARCModel(config)
    ckpt_path = str(RESULTS_DIR / "checkpoints" / "varc_exp3_base.pt")
    if Path(ckpt_path).exists():
        print("  Loading pre-trained checkpoint...")
        load_checkpoint(model, ckpt_path, device)
    else:
        print("  Training base VARC on training tasks...")
        train_dataset = ARCDataset(split="training")
        val_dataset = ARCDataset(split="training", max_tasks=40)
        train_model(model, train_dataset, val_dataset,
                    num_epochs=50, batch_size=16, device=device,
                    checkpoint_path=ckpt_path)
    print("  Evaluating with TTT (20 steps per task)...")
    results = evaluate_on_tasks(model, tasks, device, use_ttt=True, ttt_steps=20, ttt_lr=1e-4)
    return results


def save_results(results: list[dict], name: str, exp_num: str) -> None:
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"exp{exp_num}_{name}_{int(time.time())}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {path}")


def print_metrics(results: list[dict]) -> dict:
    if not results:
        acc = {"accuracy": 0.0, "correct": 0, "total": 0}
    else:
        correct = sum(1 for r in results if r.get("correct"))
        acc = {"accuracy": correct / len(results), "correct": correct, "total": len(results)}
    print(f"  Accuracy: {acc['accuracy']:.1%} ({acc['correct']}/{acc['total']})")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARC comparison experiments")
    parser.add_argument("--exp", type=str, default="all",
                        help="Experiment: 1, 2, 3, or all")
    parser.add_argument("--tasks", type=int, default=20,
                        help="Number of evaluation tasks to test")
    parser.add_argument("--train-tasks", type=int, default=400,
                        help="Number of training tasks (for VARC)")
    parser.add_argument("--describe", action="store_true",
                        help="Exp 1: use color counts only (no grid numbers)")
    parser.add_argument("--no-few-shot", action="store_true",
                        help="Exp 1: disable few-shot examples")
    parser.add_argument("--shuffle", action="store_true",
                        help="Randomly shuffle tasks instead of sequential")
    args = parser.parse_args()

    exps = ["1", "2", "3"] if args.exp == "all" else [args.exp]

    for exp_num in exps:
        if exp_num not in EXPERIMENTS:
            print(f"Unknown experiment: {exp_num}")
            continue

        info = EXPERIMENTS[exp_num]
        print(f"\n{'='*60}")
        print(f"  Experiment {exp_num}: {info['name']}")
        print(f"{'='*60}")

        tasks = load_tasks(split="training", max_tasks=None if args.shuffle else args.tasks)
        if args.shuffle:
            import random
            random.shuffle(tasks)
            tasks = tasks[:args.tasks]
        print(f"  Loaded {len(tasks)} tasks\n")

        if info["method"] == "deepseek":
            if DeepSeekClient is None:
                print("  SKIP: openai not installed.")
                continue
            use_few_shot = not args.no_few_shot
            results = run_exp1(tasks, text_only=args.describe, use_few_shot=use_few_shot)
        else:
            if VARCModel is None:
                print("  SKIP: torch not installed. Use conda env.")
                continue
            if exp_num == "2":
                results = run_exp2(tasks)
            else:
                results = run_exp3(tasks)

        metrics = print_metrics(results)
        save_results(results, info["name"], exp_num)


if __name__ == "__main__":
    main()
