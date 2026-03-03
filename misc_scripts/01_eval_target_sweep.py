#!/usr/bin/env python3
"""
Run target-split eval for all checkpoints in a run directory, then summarize.

Output structure:
  <output_root>/<checkpoint_stem>/target/<task>/eval_log.json
"""

import argparse
import glob
import json
import pathlib
import subprocess
import sys
from datetime import datetime


def find_checkpoints(checkpoint_dir: pathlib.Path, include_latest: bool) -> list[pathlib.Path]:
    ckpts = sorted(checkpoint_dir.glob("epoch=*.ckpt"))
    if include_latest:
        latest = checkpoint_dir / "latest.ckpt"
        if latest.exists():
            ckpts.append(latest)
    return ckpts


def run_eval(
    eval_script: pathlib.Path,
    ckpt_path: pathlib.Path,
    task: str,
    split: str,
    num_rollouts: int,
    num_envs: int,
    device: str,
    output_dir: pathlib.Path,
    overwrite: bool,
) -> None:
    cmd = [
        sys.executable,
        str(eval_script),
        "--checkpoint",
        str(ckpt_path),
        "--task",
        task,
        "--split",
        split,
        "--num_rollouts",
        str(num_rollouts),
        "--num_envs",
        str(num_envs),
        "--device",
        device,
        "--output_dir",
        str(output_dir),
    ]
    if overwrite:
        cmd.append("--overwrite")

    print(f"\n[eval] {ckpt_path.name}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def load_result(eval_log_path: pathlib.Path, task: str) -> tuple[float | None, float | None]:
    if not eval_log_path.exists():
        return None, None
    with open(eval_log_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    success = obj.get(f"success_rate/{task}", None)
    mean_score = obj.get("test/mean_score", None)
    return success, mean_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Training run directory (contains checkpoints/).",
    )
    parser.add_argument("--task", default="CloseFridge")
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="target", choices=["target"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_latest", action="store_true", help="Skip latest.ckpt.")
    parser.add_argument(
        "--output_root",
        default=None,
        help="If unset, uses <run_dir>/evals/target_sweep_<YYYYMMDD>.",
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    eval_script = root / "misc_scripts" / "00_eval_single_task.py"
    run_dir = pathlib.Path(args.run_dir).resolve()
    checkpoint_dir = run_dir / "checkpoints"

    if not eval_script.exists():
        raise FileNotFoundError(f"Missing eval script: {eval_script}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint dir: {checkpoint_dir}")

    if args.output_root is None:
        day = datetime.now().strftime("%Y%m%d")
        output_root = run_dir / "evals" / f"target_sweep_{day}"
    else:
        output_root = pathlib.Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ckpts = find_checkpoints(checkpoint_dir, include_latest=(not args.no_latest))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {checkpoint_dir}")

    rows = []
    for ckpt in ckpts:
        ckpt_stem = ckpt.stem
        this_out = output_root / ckpt_stem / args.split
        run_eval(
            eval_script=eval_script,
            ckpt_path=ckpt,
            task=args.task,
            split=args.split,
            num_rollouts=args.num_rollouts,
            num_envs=args.num_envs,
            device=args.device,
            output_dir=this_out,
            overwrite=args.overwrite,
        )

        eval_log = this_out / args.task / "eval_log.json"
        success, mean_score = load_result(eval_log, args.task)
        rows.append(
            {
                "checkpoint": ckpt.name,
                "eval_log": str(eval_log),
                "success_rate": success,
                "mean_score": mean_score,
            }
        )

    summary_path = output_root / f"summary_{args.task}_{args.split}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("\n=== Summary ===")
    print(f"{'checkpoint':45s}  {'success':>8s}  {'mean_score':>10s}")
    for r in rows:
        s = "None" if r["success_rate"] is None else f"{r['success_rate']:.4f}"
        m = "None" if r["mean_score"] is None else f"{r['mean_score']:.4f}"
        print(f"{r['checkpoint'][:45]:45s}  {s:>8s}  {m:>10s}")
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
