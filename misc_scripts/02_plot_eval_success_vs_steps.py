#!/usr/bin/env python3
"""
Plot sim eval success rate vs train steps from an eval sweep directory.

Expected eval layout:
  <eval_root>/<checkpoint_name>/<split>/<task>/eval_log.json

Example:
  python misc_scripts/02_plot_eval_success_vs_steps.py \
    --eval_root /home/phan07/diffusion_policy/data/outputs/no_force_baseline_rinsesinkbasin_target151/evals/target_sweep_20260306 \
    --task RinseSinkBasin \
    --split target \
    --steps_per_epoch 2400
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
from typing import Any

import matplotlib.pyplot as plt


EPOCH_RE = re.compile(r"epoch=(\d+)")


def parse_epoch(checkpoint_name: str) -> int | None:
    match = EPOCH_RE.search(checkpoint_name)
    if match is None:
        return None
    return int(match.group(1))


def load_metric(eval_log_path: pathlib.Path, task: str) -> float | None:
    with eval_log_path.open("r", encoding="utf-8") as f:
        obj: dict[str, Any] = json.load(f)

    success_key = f"success_rate/{task}"
    if success_key in obj and obj[success_key] is not None:
        return float(obj[success_key])
    if "test/mean_score" in obj and obj["test/mean_score"] is not None:
        return float(obj["test/mean_score"])
    return None


def collect_rows(
    eval_root: pathlib.Path,
    task: str,
    split: str,
    steps_per_epoch: int,
    include_latest: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = f"*/{split}/{task}/eval_log.json"

    for eval_log_path in sorted(eval_root.glob(pattern)):
        checkpoint_name = eval_log_path.parents[2].name
        epoch = parse_epoch(checkpoint_name)

        if epoch is None:
            if include_latest and checkpoint_name == "latest":
                # No exact epoch encoded in latest.ckpt eval directory name.
                # Keep row for visibility but without x-axis step.
                metric = load_metric(eval_log_path, task)
                if metric is None:
                    continue
                rows.append(
                    {
                        "checkpoint": checkpoint_name,
                        "epoch": None,
                        "train_steps": None,
                        "success_rate": metric,
                        "eval_log": str(eval_log_path),
                    }
                )
            continue

        metric = load_metric(eval_log_path, task)
        if metric is None:
            continue

        rows.append(
            {
                "checkpoint": checkpoint_name,
                "epoch": epoch,
                "train_steps": epoch * steps_per_epoch,
                "success_rate": metric,
                "eval_log": str(eval_log_path),
            }
        )

    rows.sort(key=lambda r: (-1 if r["epoch"] is None else int(r["epoch"])))
    return rows


def write_csv(rows: list[dict[str, Any]], out_csv: pathlib.Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["checkpoint", "epoch", "train_steps", "success_rate", "eval_log"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: list[dict[str, Any]], out_json: pathlib.Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def make_plot(
    rows: list[dict[str, Any]],
    task: str,
    split: str,
    steps_per_epoch: int,
    out_png: pathlib.Path,
) -> None:
    x = [int(r["train_steps"]) for r in rows if r["train_steps"] is not None]
    y = [float(r["success_rate"]) for r in rows if r["train_steps"] is not None]

    if not x:
        raise RuntimeError("No epoch-based eval rows found to plot.")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xlabel("Train Steps")
    plt.ylabel("Success Rate")
    plt.title(f"{task} ({split}) success vs train steps  |  {steps_per_epoch} steps/epoch")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", required=True, help="Eval sweep directory root.")
    parser.add_argument("--task", required=True, help="Task name, e.g. RinseSinkBasin.")
    parser.add_argument("--split", default="target", choices=["target", "pretrain"])
    parser.add_argument("--steps_per_epoch", type=int, default=2400)
    parser.add_argument("--include_latest", action="store_true")
    parser.add_argument("--out_png", default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    eval_root = pathlib.Path(args.eval_root).resolve()
    if not eval_root.exists():
        raise FileNotFoundError(f"Missing eval root: {eval_root}")

    rows = collect_rows(
        eval_root=eval_root,
        task=args.task,
        split=args.split,
        steps_per_epoch=args.steps_per_epoch,
        include_latest=args.include_latest,
    )
    if not rows:
        raise RuntimeError(
            f"No eval logs found under {eval_root} with split={args.split}, task={args.task}."
        )

    stem = f"success_vs_steps_{args.task}_{args.split}"
    out_png = pathlib.Path(args.out_png).resolve() if args.out_png else (eval_root / f"{stem}.png")
    out_csv = pathlib.Path(args.out_csv).resolve() if args.out_csv else (eval_root / f"{stem}.csv")
    out_json = pathlib.Path(args.out_json).resolve() if args.out_json else (eval_root / f"{stem}.json")

    make_plot(rows=rows, task=args.task, split=args.split, steps_per_epoch=args.steps_per_epoch, out_png=out_png)
    write_csv(rows, out_csv)
    write_json(rows, out_json)

    print(f"Saved plot: {out_png}")
    print(f"Saved csv : {out_csv}")
    print(f"Saved json: {out_json}")


if __name__ == "__main__":
    main()
