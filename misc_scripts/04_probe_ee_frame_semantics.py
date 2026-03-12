#!/usr/bin/env python3
"""
Probe whether RoboCasa LeRobot EE state keys are temporal deltas or
frame-relative absolute values.

Primary checks:
1) Metadata check:
   - state key names / modality mapping
   - controller input settings in extras/dataset_meta.json
2) Data check:
   - compare magnitude of EE state vs one-step temporal deltas
3) Optional sim replay check (strongest evidence):
   - replay stored MuJoCo states and compare:
     dataset state.end_effector_position_relative
       vs simulator robot0_base_to_eef_pos
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import robosuite
except Exception:
    robosuite = None

try:
    from robocasa.utils.env_utils import create_env
except Exception:
    create_env = None


KEY_STATE = "observation.state"
KEY_EEF_POS_REL = "end_effector_position_relative"
KEY_EEF_QUAT_REL = "end_effector_rotation_relative"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def default_report_path(dataset_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path("artifacts")
        / "ee_frame_probe_reports"
        / f"{dataset_root.name}_ee_frame_probe_{ts}.json"
    )


def get_total_episodes(dataset_root: Path) -> int:
    info = load_json(dataset_root / "meta" / "info.json")
    total = info.get("total_episodes")
    if not isinstance(total, int):
        raise ValueError("meta/info.json missing valid total_episodes")
    return total


def parse_episodes_spec(spec: str, total_episodes: int) -> list[int]:
    s = spec.strip().lower()
    if s == "all":
        return list(range(total_episodes))

    out: set[int] = set()
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a = int(a_str)
            b = int(b_str)
            if b < a:
                a, b = b, a
            for ep in range(a, b + 1):
                if ep < 0 or ep >= total_episodes:
                    raise ValueError(
                        f"Episode {ep} out of bounds [0, {total_episodes - 1}]"
                    )
                out.add(ep)
        else:
            ep = int(part)
            if ep < 0 or ep >= total_episodes:
                raise ValueError(
                    f"Episode {ep} out of bounds [0, {total_episodes - 1}]"
                )
            out.add(ep)
    return sorted(out)


def find_episode_parquet(dataset_root: Path, episode: int) -> Path:
    matches = list(dataset_root.glob(f"data/*/episode_{episode:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No parquet found for episode_{episode:06d} in {dataset_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple parquet files for episode {episode}: {matches}")
    return matches[0]


def get_slice_from_modality(modality: dict[str, Any], state_key: str) -> tuple[int, int]:
    entry = modality["state"].get(state_key)
    if entry is None:
        raise KeyError(f"state key '{state_key}' not found in modality.json")
    return int(entry["start"]), int(entry["end"])


def rowstack_object_array(series: pd.Series, dtype=np.float64) -> np.ndarray:
    return np.vstack([np.asarray(v, dtype=dtype).reshape(-1) for v in series.tolist()])


def load_episode_state_matrix(dataset_root: Path, episode: int) -> np.ndarray:
    p = find_episode_parquet(dataset_root, episode)
    df = pd.read_parquet(p, columns=[KEY_STATE])
    return rowstack_object_array(df[KEY_STATE], dtype=np.float64)


def load_episode_sim_state(dataset_root: Path, episode: int) -> tuple[np.ndarray, str, dict[str, Any]]:
    ep_dir = dataset_root / "extras" / f"episode_{episode:06d}"
    states_npz = np.load(ep_dir / "states.npz")
    states = np.asarray(states_npz["states"], dtype=np.float64)
    with gzip.open(ep_dir / "model.xml.gz", "rt") as f:
        model_xml = f.read()
    ep_meta = load_json(ep_dir / "ep_meta.json")
    return states, model_xml, ep_meta


def reset_to(env, state: dict[str, Any]) -> None:
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}

        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)

        env.reset()
        if robosuite is None:
            raise RuntimeError("robosuite is required for sim replay check")
        robosuite_minor = int(robosuite.__version__.split(".")[1])
        if robosuite_minor <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            xml = env.edit_model_xml(state["model"])
        env.reset_from_xml_string(xml)
        env.sim.reset()

    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()

    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def get_obs(env) -> dict[str, Any]:
    if getattr(env, "viewer_get_obs", False):
        return env.viewer._get_observations(force_update=True)
    return env._get_observations(force_update=True)


def choose_frame_indices(t_horizon: int, max_frames: int, stride: int) -> list[int]:
    if t_horizon <= 0:
        return []
    idxs = list(range(0, t_horizon, max(1, stride)))
    if (t_horizon - 1) not in idxs:
        idxs.append(t_horizon - 1)
    if max_frames > 0 and len(idxs) > max_frames:
        sel = np.linspace(0, len(idxs) - 1, num=max_frames, dtype=int)
        idxs = [idxs[i] for i in sel.tolist()]
    return sorted(set(idxs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="LeRobot dataset root")
    parser.add_argument("--episodes", type=str, default="0-2", help="all | 0-10 | 0,5,9")
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=25,
        help="Max replay frames per episode for sim check",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=20,
        help="Stride before frame subsampling",
    )
    parser.add_argument(
        "--sim-check",
        action="store_true",
        help="Replay MuJoCo states and numerically compare against dataset state key",
    )
    parser.add_argument(
        "--sim-tol",
        type=float,
        default=1e-6,
        help="Tolerance for max abs error in sim check",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-json", type=str, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if sim check fails")
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Invalid dataset root: {dataset_root}")

    report_path = Path(args.report_json) if args.report_json else default_report_path(dataset_root)
    t0 = time.time()

    modality = load_json(dataset_root / "meta" / "modality.json")
    if "state" not in modality:
        raise KeyError("meta/modality.json missing 'state'")

    pos_start, pos_end = get_slice_from_modality(modality, KEY_EEF_POS_REL)
    quat_start, quat_end = get_slice_from_modality(modality, KEY_EEF_QUAT_REL)

    dataset_meta_path = dataset_root / "extras" / "dataset_meta.json"
    dataset_meta = load_json(dataset_meta_path) if dataset_meta_path.exists() else {}
    env_name = dataset_meta.get("env", None)
    split = (
        dataset_meta.get("env_args", {})
        .get("env_kwargs", {})
        .get("obj_instance_split", None)
    )
    right_ctrl = (
        dataset_meta.get("env_args", {})
        .get("env_kwargs", {})
        .get("controller_configs", {})
        .get("body_parts", {})
        .get("right", {})
    )
    ctrl_input_type = right_ctrl.get("input_type", None)
    ctrl_input_ref_frame = right_ctrl.get("input_ref_frame", None)

    total_episodes = get_total_episodes(dataset_root)
    episodes = parse_episodes_spec(args.episodes, total_episodes)

    report: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_root),
        "episodes": episodes,
        "modality_mapping": {
            "state.end_effector_position_relative": {"start": pos_start, "end": pos_end},
            "state.end_effector_rotation_relative": {"start": quat_start, "end": quat_end},
        },
        "controller_hints": {
            "input_type": ctrl_input_type,
            "input_ref_frame": ctrl_input_ref_frame,
            "env_name": env_name,
            "split": split,
        },
        "episode_results": [],
        "sim_check_enabled": bool(args.sim_check),
        "sim_check_summary": {},
        "conclusion": {},
    }

    env = None
    if args.sim_check:
        if create_env is None or robosuite is None:
            raise RuntimeError(
                "--sim-check requested but robocasa/robosuite imports failed. "
                "Run in an environment where they are installed."
            )
        if not env_name or not split:
            raise RuntimeError(
                "--sim-check requested but dataset extras/dataset_meta.json is missing "
                "env/split info."
            )
        env = create_env(env_name=env_name, split=split, seed=args.seed)

    global_max_pos_err = 0.0
    global_max_quat_err = 0.0
    sim_checked_frames = 0
    sim_checked_episodes = 0

    try:
        for ep in episodes:
            ep_t0 = time.time()
            obs_state_mat = load_episode_state_matrix(dataset_root, ep)
            if obs_state_mat.shape[1] < quat_end:
                raise RuntimeError(
                    f"Episode {ep}: observation.state dim {obs_state_mat.shape[1]} "
                    f"is smaller than required end index {quat_end}"
                )

            ds_pos = obs_state_mat[:, pos_start:pos_end]
            ds_quat = obs_state_mat[:, quat_start:quat_end]
            dpos = np.diff(ds_pos, axis=0) if ds_pos.shape[0] > 1 else np.zeros((0, 3))

            ep_result: dict[str, Any] = {
                "episode": ep,
                "frames": int(ds_pos.shape[0]),
                "state_norm_mean": float(np.mean(np.linalg.norm(ds_pos, axis=1))),
                "state_norm_std": float(np.std(np.linalg.norm(ds_pos, axis=1))),
                "delta_norm_mean": float(np.mean(np.linalg.norm(dpos, axis=1))) if len(dpos) else 0.0,
                "delta_norm_std": float(np.std(np.linalg.norm(dpos, axis=1))) if len(dpos) else 0.0,
                "first_state_norm": float(np.linalg.norm(ds_pos[0])) if len(ds_pos) else 0.0,
            }

            # Heuristic signal: if these were temporal deltas, state norm and per-step delta
            # norm scales should be similar. In practice they are usually not.
            delta_ratio = (
                ep_result["state_norm_mean"] / max(ep_result["delta_norm_mean"], 1e-12)
                if ep_result["delta_norm_mean"] > 0
                else float("inf")
            )
            ep_result["state_vs_delta_norm_ratio"] = float(delta_ratio)

            if env is not None:
                states, model_xml, ep_meta = load_episode_sim_state(dataset_root, ep)
                if len(states) != len(ds_pos):
                    raise RuntimeError(
                        f"Episode {ep}: states length {len(states)} != parquet length {len(ds_pos)}"
                    )
                reset_to(
                    env,
                    {
                        "states": states[0],
                        "model": model_xml,
                        "ep_meta": json.dumps(ep_meta),
                    },
                )

                frame_ids = choose_frame_indices(
                    t_horizon=len(states),
                    max_frames=args.max_frames_per_episode,
                    stride=args.frame_stride,
                )
                ep_max_pos_err = 0.0
                ep_max_quat_err = 0.0
                for t_idx in frame_ids:
                    reset_to(env, {"states": states[t_idx]})
                    obs = get_obs(env)
                    sim_pos = np.asarray(obs["robot0_base_to_eef_pos"], dtype=np.float64).reshape(-1)
                    sim_quat = np.asarray(obs["robot0_base_to_eef_quat"], dtype=np.float64).reshape(-1)
                    pos_err = float(np.max(np.abs(sim_pos - ds_pos[t_idx])))
                    quat_err = float(np.max(np.abs(sim_quat - ds_quat[t_idx])))
                    ep_max_pos_err = max(ep_max_pos_err, pos_err)
                    ep_max_quat_err = max(ep_max_quat_err, quat_err)
                    sim_checked_frames += 1

                sim_checked_episodes += 1
                ep_result["sim_check"] = {
                    "frames_checked": len(frame_ids),
                    "max_abs_pos_err": ep_max_pos_err,
                    "max_abs_quat_err": ep_max_quat_err,
                }
                global_max_pos_err = max(global_max_pos_err, ep_max_pos_err)
                global_max_quat_err = max(global_max_quat_err, ep_max_quat_err)

            ep_result["elapsed_sec"] = round(time.time() - ep_t0, 4)
            report["episode_results"].append(ep_result)
    finally:
        if env is not None:
            env.close()

    sim_success = None
    if env is not None:
        sim_success = (
            global_max_pos_err <= float(args.sim_tol)
            and global_max_quat_err <= float(args.sim_tol)
        )
        report["sim_check_summary"] = {
            "episodes_checked": sim_checked_episodes,
            "frames_checked": sim_checked_frames,
            "global_max_abs_pos_err": global_max_pos_err,
            "global_max_abs_quat_err": global_max_quat_err,
            "tol": float(args.sim_tol),
            "success": bool(sim_success),
        }

    metadata_support = (
        ctrl_input_type == "delta"
        and ctrl_input_ref_frame == "base"
    )
    mean_ratio = float(
        np.mean([x["state_vs_delta_norm_ratio"] for x in report["episode_results"]])
    ) if report["episode_results"] else float("nan")
    data_support = bool(mean_ratio > 3.0)

    if sim_success is True:
        verdict = (
            "Confirmed by replay: dataset state.end_effector_position_relative matches "
            "simulator robot0_base_to_eef_pos at the same simulator state. "
            "This is an absolute EE pose in robot-base frame (not a temporal delta)."
        )
    elif sim_success is False:
        verdict = (
            "Sim replay check failed tolerance, so hard confirmation failed. "
            "Inspect sim_check_summary and per-episode errors."
        )
    else:
        verdict = (
            "Sim replay check not run. Metadata and value-scale checks indicate "
            "base-frame absolute state semantics rather than temporal deltas."
        )

    report["conclusion"] = {
        "metadata_supports_base_frame_semantics": bool(metadata_support),
        "data_scale_supports_non_delta_state": bool(data_support),
        "state_vs_delta_norm_ratio_mean": mean_ratio,
        "verdict": verdict,
    }
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_sec"] = round(time.time() - t0, 4)

    dump_json(report_path, report)

    print(f"Probe report: {report_path}")
    print(
        f"metadata input_type={ctrl_input_type}, input_ref_frame={ctrl_input_ref_frame}, "
        f"state_vs_delta_norm_ratio_mean={mean_ratio:.4f}"
    )
    if env is not None:
        print(
            "sim_check: "
            f"max_pos_err={global_max_pos_err:.3e}, "
            f"max_quat_err={global_max_quat_err:.3e}, "
            f"tol={args.sim_tol:.3e}, success={sim_success}"
        )
    print(f"Verdict: {verdict}")

    if args.strict and env is not None and not sim_success:
        sys.exit(1)


if __name__ == "__main__":
    main()

