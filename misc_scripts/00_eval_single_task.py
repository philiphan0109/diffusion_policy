#!/usr/bin/env python3
"""
Single-task RoboCasa eval wrapper for diffusion_policy.

This avoids the default task-set loop in eval_robocasa.py and forces
AsyncVectorEnv to run without shared memory.
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_robocasa import eval_task
import diffusion_policy.env_runner.robomimic_image_runner as rr
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv as _AsyncVectorEnv
import diffusion_policy.gym_util.async_vector_env as _ave


_ORIG_RESET_ASYNC = _AsyncVectorEnv.reset_async
_ORIG_RESET_WAIT = _AsyncVectorEnv.reset_wait
_ORIG_CONCATENATE = _ave.concatenate
_ORIG_RUNNER_CLOSE = rr.RobomimicImageRunner.close


def _reset_async_compat(self, seed=None, options=None):
    # gym==0.26 passes seed/options to reset_async, but this repo's
    # AsyncVectorEnv implements the older signature with no kwargs.
    return _ORIG_RESET_ASYNC(self)


def _reset_wait_compat(self, timeout=None, seed=None, options=None):
    # gym==0.26 also passes seed/options to reset_wait.
    return _ORIG_RESET_WAIT(self, timeout=timeout)


def _concatenate_compat(items, out, space):
    # This repo calls concatenate(items, out, space), but gym==0.26 expects
    # concatenate(space, items, out).
    return _ORIG_CONCATENATE(space, items, out)


def _async_safe(*args, **kwargs):
    kwargs.setdefault("shared_memory", False)
    kwargs.setdefault("context", "spawn")
    return _AsyncVectorEnv(*args, **kwargs)


def _runner_close_compat(self):
    # Stock close() early-returns for AsyncVectorEnv, which leaves worker
    # processes open and later triggers BrokenPipe during interpreter teardown.
    try:
        _ORIG_RUNNER_CLOSE(self)
    except Exception:
        pass

    env = getattr(self, "env", None)
    if env is not None:
        try:
            env.close(terminate=True)
        except (BrokenPipeError, EOFError, OSError):
            pass
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True, help="RoboCasa task name, e.g. CloseFridge")
    parser.add_argument("--split", required=True, choices=["pretrain", "target"])
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    _AsyncVectorEnv.reset_async = _reset_async_compat
    _AsyncVectorEnv.reset_wait = _reset_wait_compat
    _ave.concatenate = _concatenate_compat
    rr.AsyncVectorEnv = _async_safe
    rr.RobomimicImageRunner.close = _runner_close_compat

    eval_task(
        checkpoint=args.checkpoint,
        base_output_dir=args.output_dir,
        device=args.device,
        task=args.task,
        num_rollouts=args.num_rollouts,
        num_envs=args.num_envs,
        split=args.split,
        overwrite=args.overwrite,
    )
    print("done")


if __name__ == "__main__":
    main()
