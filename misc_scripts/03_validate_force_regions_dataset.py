#!/usr/bin/env python3
"""
Validate force-region augmented LeRobot dataset exported by force-region scripts.

This reproduces the script-13 style validation logic in this repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


KEY_QFRC_ARM = "observation.force.qfrc_constraint_arm"
KEY_QFRC_ARM_L2 = "observation.force.qfrc_constraint_arm_l2"

KEY_PHASE_LABEL = "diagnostic.force_phase.label"
KEY_PHASE_CONTACT = "diagnostic.force_phase.contact_mask"
KEY_PHASE_PRE = "diagnostic.force_phase.precontact_mask"
KEY_PHASE_SIGNAL_NORM = "diagnostic.force_phase.signal_norm"
KEY_PHASE_SIGNAL_SMOOTH = "diagnostic.force_phase.signal_smooth"

REQ_KEYS = [
    KEY_PHASE_LABEL,
    KEY_PHASE_CONTACT,
    KEY_PHASE_PRE,
    KEY_PHASE_SIGNAL_NORM,
    KEY_PHASE_SIGNAL_SMOOTH,
]

LABEL_FREE = 0
LABEL_PRE = 1
LABEL_CONTACT = 2


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


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


def get_total_episodes(dataset_root: Path) -> int:
    info = load_json(dataset_root / "meta" / "info.json")
    total = info.get("total_episodes")
    if not isinstance(total, int):
        raise ValueError("meta/info.json missing valid total_episodes")
    return total


def find_episode_parquet(dataset_root: Path, episode: int) -> Path:
    matches = list(dataset_root.glob(f"data/*/episode_{episode:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No parquet found for episode_{episode:06d} in {dataset_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple parquet files for episode {episode}: {matches}")
    return matches[0]


def col_to_scalar_vector(df: pd.DataFrame, key: str, dtype=np.float64) -> np.ndarray:
    return np.asarray(
        [np.asarray(v, dtype=dtype).reshape(-1)[0] for v in df[key].tolist()],
        dtype=dtype,
    )


def infer_force_signal(df: pd.DataFrame) -> np.ndarray:
    if KEY_QFRC_ARM_L2 in df.columns:
        sig = col_to_scalar_vector(df, KEY_QFRC_ARM_L2, dtype=np.float64)
    elif KEY_QFRC_ARM in df.columns:
        mat = np.vstack(
            [np.asarray(v, dtype=np.float64).reshape(-1) for v in df[KEY_QFRC_ARM].tolist()]
        )
        sig = np.linalg.norm(mat, axis=1)
    else:
        raise RuntimeError(
            f"Missing force keys. Need at least one of: {KEY_QFRC_ARM_L2}, {KEY_QFRC_ARM}"
        )
    return np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_contact_1d(contact: np.ndarray, mode: str) -> np.ndarray:
    c = np.asarray(contact, dtype=np.float64)
    if mode == "none":
        return c
    if mode != "episode":
        raise ValueError(f"Unsupported normalize mode: {mode}")
    lo = float(np.min(c))
    hi = float(np.max(c))
    if hi > lo:
        return (c - lo) / (hi - lo)
    return np.ones_like(c) if hi > 0 else np.zeros_like(c)


def ema_smooth(signal: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    out = np.empty_like(x)
    out[0] = x[0]
    a = float(alpha)
    for i in range(1, x.shape[0]):
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
    return out


def hysteresis_contact_binary(
    signal_1d: np.ndarray, threshold_high: float, threshold_low: float
) -> np.ndarray:
    x = np.asarray(signal_1d, dtype=np.float64)
    out = np.zeros((x.shape[0],), dtype=np.int8)
    on = False
    for i in range(x.shape[0]):
        s = x[i]
        if on:
            if s < threshold_low:
                on = False
        else:
            if s > threshold_high:
                on = True
        out[i] = 1 if on else 0
    return out


def build_pre_mask(con_mask: np.ndarray, precontact_frames: int) -> np.ndarray:
    c = np.asarray(con_mask, dtype=np.int8).reshape(-1)
    pre = np.zeros_like(c, dtype=np.int8)
    n = c.shape[0]
    for i in range(n):
        if c[i] == 1 and (i == 0 or c[i - 1] == 0):
            a = max(0, i - int(precontact_frames))
            pre[a:i] = 1
    # Enforce strict exclusivity: never precontact where already in contact.
    pre[c == 1] = 0
    return pre


def build_labels(con_mask: np.ndarray, pre_mask: np.ndarray) -> np.ndarray:
    labels = np.zeros((con_mask.shape[0],), dtype=np.int8)
    labels[con_mask == 1] = LABEL_CONTACT
    labels[(pre_mask == 1) & (con_mask == 0)] = LABEL_PRE
    return labels


def compute_expected_regions(
    raw_force_signal: np.ndarray,
    normalize_mode: str,
    ema_alpha: float,
    threshold_high: float,
    threshold_low: float,
    precontact_frames: int,
) -> dict[str, np.ndarray]:
    sig_norm = normalize_contact_1d(raw_force_signal, mode=normalize_mode)
    sig_smooth = ema_smooth(sig_norm, alpha=ema_alpha)
    hi = float(threshold_high)
    lo = float(threshold_low)
    if hi < lo:
        hi, lo = lo, hi

    con_mask = hysteresis_contact_binary(sig_smooth, threshold_high=hi, threshold_low=lo)
    pre_mask = build_pre_mask(con_mask=con_mask, precontact_frames=precontact_frames)
    labels = build_labels(con_mask=con_mask, pre_mask=pre_mask)
    return {
        "signal_norm": sig_norm,
        "signal_smooth": sig_smooth,
        "contact_mask": con_mask.astype(np.int8),
        "precontact_mask": pre_mask.astype(np.int8),
        "label": labels.astype(np.int8),
    }


def validate_metadata(dataset_root: Path) -> list[str]:
    errors: list[str] = []

    info = load_json(dataset_root / "meta" / "info.json")
    features = info.get("features", {})
    exp_shapes = {
        KEY_PHASE_LABEL: [1],
        KEY_PHASE_CONTACT: [1],
        KEY_PHASE_PRE: [1],
        KEY_PHASE_SIGNAL_NORM: [1],
        KEY_PHASE_SIGNAL_SMOOTH: [1],
    }
    exp_dtypes = {
        KEY_PHASE_LABEL: "int8",
        KEY_PHASE_CONTACT: "bool",
        KEY_PHASE_PRE: "bool",
        KEY_PHASE_SIGNAL_NORM: "float64",
        KEY_PHASE_SIGNAL_SMOOTH: "float64",
    }
    for key in REQ_KEYS:
        if key not in features:
            errors.append(f"meta/info.json missing feature {key}")
            continue
        feat = features[key]
        if feat.get("dtype") != exp_dtypes[key]:
            errors.append(
                f"meta/info.json dtype mismatch for {key}: {feat.get('dtype')} vs {exp_dtypes[key]}"
            )
        if feat.get("shape") != exp_shapes[key]:
            errors.append(
                f"meta/info.json shape mismatch for {key}: {feat.get('shape')} vs {exp_shapes[key]}"
            )

    modality = load_json(dataset_root / "meta" / "modality.json")
    diagnostic = modality.get("diagnostic", {})
    mod_expect = {
        "force_phase_label": KEY_PHASE_LABEL,
        "force_phase_contact_mask": KEY_PHASE_CONTACT,
        "force_phase_precontact_mask": KEY_PHASE_PRE,
        "force_phase_signal_norm": KEY_PHASE_SIGNAL_NORM,
        "force_phase_signal_smooth": KEY_PHASE_SIGNAL_SMOOTH,
    }
    for mkey, okey in mod_expect.items():
        if diagnostic.get(mkey, {}).get("original_key") != okey:
            errors.append(f"meta/modality.json missing diagnostic.{mkey} mapping")

    stats = load_json(dataset_root / "meta" / "stats.json")
    for key in REQ_KEYS:
        if key not in stats:
            errors.append(f"meta/stats.json missing key {key}")
            continue
        row = stats[key]
        for stat_name in ["mean", "std", "min", "max", "q01", "q99"]:
            if stat_name not in row:
                errors.append(f"meta/stats.json missing {stat_name} for {key}")

    schema_path = dataset_root / "meta" / "force_region_schema.json"
    if not schema_path.exists():
        errors.append("meta/force_region_schema.json missing")

    return errors


def default_report_path(dataset_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path("artifacts")
        / "force_region_validation_reports"
        / f"{dataset_root.name}_force_region_validation_{ts}.json"
    )


def maybe_plot_episode(
    dataset_root: Path,
    episode: int,
    output_png: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for --plot-episode") from exc

    p = find_episode_parquet(dataset_root, episode)
    df = pd.read_parquet(
        p,
        columns=[
            KEY_PHASE_LABEL,
            KEY_PHASE_SIGNAL_SMOOTH,
        ],
    )
    labels = col_to_scalar_vector(df, KEY_PHASE_LABEL, dtype=np.int8)
    sig = col_to_scalar_vector(df, KEY_PHASE_SIGNAL_SMOOTH, dtype=np.float64)

    t = np.arange(sig.shape[0])
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=130)
    ax.plot(t, sig, linewidth=1.8, color="#1f77b4", label="signal_smooth")
    for i in range(sig.shape[0]):
        if labels[i] == LABEL_PRE:
            ax.axvspan(i, i + 1, color="#2ca02c", alpha=0.30, linewidth=0)
        elif labels[i] == LABEL_CONTACT:
            ax.axvspan(i, i + 1, color="#d62728", alpha=0.30, linewidth=0)
    ax.set_title(f"Episode {episode:06d} force-region sanity plot")
    ax.set_xlabel("frame")
    ax.set_ylabel("signal")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Augmented dataset root")
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episode spec: all | 0-103 | 0,1,2",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Return non-zero if any validation error is found",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Always exit zero and rely on report contents",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-8,
        help="Relative tolerance for value checks",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for value checks",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Validation report path",
    )
    parser.add_argument(
        "--plot-episode",
        type=int,
        default=None,
        help="Optional episode index to export a sanity PNG",
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default=None,
        help="Optional output PNG path for --plot-episode",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Invalid dataset root: {dataset_root}")

    schema_path = dataset_root / "meta" / "force_region_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema: {schema_path}")

    report_path = (
        Path(args.report_json) if args.report_json else default_report_path(dataset_root)
    )
    total_episodes = get_total_episodes(dataset_root)
    episodes = parse_episodes_spec(args.episodes, total_episodes)

    schema = load_json(schema_path)
    method = schema.get("method", {})
    normalize_mode = str(method.get("normalize_mode", "episode"))
    ema_alpha = float(method.get("ema_alpha", 0.3))
    threshold_high = float(method.get("effective_threshold_high", 0.15))
    threshold_low = float(method.get("effective_threshold_low", 0.05))
    precontact_frames = int(method.get("precontact_frames", 30))

    report: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_root),
        "episodes": episodes,
        "strict": bool(args.strict),
        "method_from_schema": {
            "normalize_mode": normalize_mode,
            "ema_alpha": ema_alpha,
            "threshold_high": threshold_high,
            "threshold_low": threshold_low,
            "precontact_frames": precontact_frames,
        },
        "episode_results": [],
        "metadata_errors": [],
        "loader_check": {},
        "errors": [],
    }

    t0 = time.time()
    ep_iter: Any = episodes
    pbar: Any | None = None
    if tqdm is not None:
        pbar = tqdm(
            episodes,
            total=len(episodes),
            desc="Validate force regions",
            unit="ep",
            dynamic_ncols=True,
        )
        ep_iter = pbar
    else:
        print(f"Validating {len(episodes)} episodes (install tqdm for progress bar).")

    try:
        for ep in ep_iter:
            ep_errs: list[str] = []
            ep_t0 = time.time()
            try:
                p = find_episode_parquet(dataset_root, ep)
                df = pd.read_parquet(p)

                for key in REQ_KEYS:
                    if key not in df.columns:
                        ep_errs.append(f"missing column {key}")
                if ep_errs:
                    report["episode_results"].append(
                        {
                            "episode": ep,
                            "ok": False,
                            "errors": ep_errs,
                            "elapsed_sec": round(time.time() - ep_t0, 4),
                        }
                    )
                    continue

                labels = col_to_scalar_vector(df, KEY_PHASE_LABEL, dtype=np.int8)
                contact = col_to_scalar_vector(df, KEY_PHASE_CONTACT, dtype=bool).astype(bool)
                pre = col_to_scalar_vector(df, KEY_PHASE_PRE, dtype=bool).astype(bool)
                sig_norm = col_to_scalar_vector(df, KEY_PHASE_SIGNAL_NORM, dtype=np.float64)
                sig_smooth = col_to_scalar_vector(df, KEY_PHASE_SIGNAL_SMOOTH, dtype=np.float64)
                t_horizon = len(df)

                if labels.shape[0] != t_horizon:
                    ep_errs.append("label length mismatch")
                if contact.shape[0] != t_horizon:
                    ep_errs.append("contact length mismatch")
                if pre.shape[0] != t_horizon:
                    ep_errs.append("precontact length mismatch")
                if sig_norm.shape[0] != t_horizon:
                    ep_errs.append("signal_norm length mismatch")
                if sig_smooth.shape[0] != t_horizon:
                    ep_errs.append("signal_smooth length mismatch")

                if not np.all(np.isin(labels, [LABEL_FREE, LABEL_PRE, LABEL_CONTACT])):
                    ep_errs.append("labels contain values outside {0,1,2}")

                if np.any(contact & pre):
                    ep_errs.append("contact_mask and precontact_mask overlap")

                if np.any((labels == LABEL_CONTACT) & (~contact)):
                    ep_errs.append("label=contact but contact_mask is False")
                if np.any((labels == LABEL_PRE) & (~pre)):
                    ep_errs.append("label=precontact but precontact_mask is False")
                if np.any((labels == LABEL_FREE) & (contact | pre)):
                    ep_errs.append("label=free but contact/precontact mask is True")
                if np.any((labels == LABEL_PRE) & contact):
                    ep_errs.append("label=precontact while contact_mask is True")

                raw_sig = infer_force_signal(df)
                exp = compute_expected_regions(
                    raw_force_signal=raw_sig,
                    normalize_mode=normalize_mode,
                    ema_alpha=ema_alpha,
                    threshold_high=threshold_high,
                    threshold_low=threshold_low,
                    precontact_frames=precontact_frames,
                )

                if not np.array_equal(labels, exp["label"]):
                    ep_errs.append("label values mismatch vs recomputed values")
                if not np.array_equal(contact.astype(np.int8), exp["contact_mask"]):
                    ep_errs.append("contact_mask mismatch vs recomputed values")
                if not np.array_equal(pre.astype(np.int8), exp["precontact_mask"]):
                    ep_errs.append("precontact_mask mismatch vs recomputed values")
                if not np.allclose(
                    sig_norm, exp["signal_norm"], rtol=args.rtol, atol=args.atol
                ):
                    ep_errs.append("signal_norm mismatch vs recomputed values")
                if not np.allclose(
                    sig_smooth, exp["signal_smooth"], rtol=args.rtol, atol=args.atol
                ):
                    ep_errs.append("signal_smooth mismatch vs recomputed values")

                report["episode_results"].append(
                    {
                        "episode": ep,
                        "ok": len(ep_errs) == 0,
                        "errors": ep_errs,
                        "frames": t_horizon,
                        "elapsed_sec": round(time.time() - ep_t0, 4),
                    }
                )

                if pbar is not None and hasattr(pbar, "write"):
                    if ep_errs:
                        pbar.write(
                            f"[fail] episode_{ep:06d}: {len(ep_errs)} validation error(s)"
                        )
                    else:
                        pbar.write(f"[ok] episode_{ep:06d}")
                else:
                    if ep_errs:
                        print(
                            f"[fail] episode_{ep:06d}: {len(ep_errs)} validation error(s)"
                        )
                    else:
                        print(f"[ok] episode_{ep:06d}")

            except Exception as e:
                report["episode_results"].append(
                    {
                        "episode": ep,
                        "ok": False,
                        "errors": [str(e)],
                        "elapsed_sec": round(time.time() - ep_t0, 4),
                    }
                )
                if pbar is not None and hasattr(pbar, "write"):
                    pbar.write(f"[fail] episode_{ep:06d}: {e}")
                else:
                    print(f"[fail] episode_{ep:06d}: {e}")

            if pbar is not None and hasattr(pbar, "set_postfix"):
                ep_failures_now = sum(
                    1 for x in report["episode_results"] if not bool(x.get("ok", False))
                )
                pbar.set_postfix(
                    ok=len(report["episode_results"]) - ep_failures_now,
                    fail=ep_failures_now,
                    last=f"{ep:06d}",
                )
    finally:
        if pbar is not None:
            pbar.close()

    report["metadata_errors"] = validate_metadata(dataset_root)
    report["errors"].extend(report["metadata_errors"])

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id="robocasa365", root=str(dataset_root))
        idx0 = int(ds.episode_data_index["from"][episodes[0]])
        sample = ds[idx0]
        missing = [k for k in REQ_KEYS if k not in sample]
        if missing:
            report["loader_check"] = {
                "ok": False,
                "error": f"missing sample keys: {missing}",
            }
            report["errors"].append(f"Loader check missing sample keys: {missing}")
        else:
            report["loader_check"] = {"ok": True, "error": None}
    except Exception as e:
        report["loader_check"] = {"ok": False, "error": str(e)}
        report["errors"].append(f"Loader check failed: {e}")

    if args.plot_episode is not None:
        plot_path = (
            Path(args.plot_out)
            if args.plot_out
            else Path("artifacts")
            / "force_region_validation_plots"
            / f"{dataset_root.name}_ep{args.plot_episode:06d}_sanity.png"
        )
        try:
            maybe_plot_episode(dataset_root, int(args.plot_episode), plot_path)
            report["plot"] = {"ok": True, "path": str(plot_path)}
        except Exception as e:
            report["plot"] = {"ok": False, "error": str(e)}
            report["errors"].append(f"Plot generation failed: {e}")

    ep_failures = [x for x in report["episode_results"] if not x["ok"]]
    report["summary"] = {
        "validated_count": len(report["episode_results"]),
        "episode_failures": len(ep_failures),
        "metadata_failures": len(report["metadata_errors"]),
        "has_errors": len(report["errors"]) > 0 or len(ep_failures) > 0,
        "elapsed_sec": round(time.time() - t0, 4),
    }
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()

    dump_json(report_path, report)
    print(f"Validation report: {report_path}")
    print(
        f"Validated {report['summary']['validated_count']} episodes; "
        f"episode failures={report['summary']['episode_failures']}; "
        f"metadata failures={report['summary']['metadata_failures']}"
    )

    if args.strict and report["summary"]["has_errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
