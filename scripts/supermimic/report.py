#!/usr/bin/env python3
"""Aggregate completed, matched-seed suite results without mixing code revisions."""

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

METRICS = (
    "success_rate",
    "progress",
    "object_pos_error_m",
    "object_rot_error_rad",
    "finger_mae_rad",
    "contact_f1",
    "contact_point_error_m",
)


def matched_pairs(results):
    """A pair requires equal revision, clip, seed, embodiment and real-step budget."""
    grouped = {}
    for result in results:
        manifest = json.loads((Path(result["run"]) / "manifest.json").read_text())
        if result.get("commit") != manifest["git_sha"]:
            raise ValueError(f"Result/manifest source mismatch: {result['run']}")
        key = (
            result["commit"],
            manifest["reference_sha256"],
            manifest["num_envs"],
            result["seed"],
            result["metrics"]["real_steps"],
        )
        grouped.setdefault(key, {}).setdefault(result["algorithm"], []).append(result)
    pairs = []
    for values in grouped.values():
        if len(values.get("ppo", [])) != 1:
            continue
        ppo = values["ppo"][0]
        for sm in values.get("supermimic", []):
            pe, se = ppo["evaluation"], sm["evaluation"]
            for field in (
                "seed",
                "episodes",
                "num_envs",
                "thresholds",
                "episode_allocation",
                "metric_protocol",
            ):
                if pe[field] != se[field]:
                    raise ValueError(f"Evaluation protocol mismatch: {field}")
            gaps = {
                name: se[name] - pe[name]
                for name in METRICS
                if pe.get(name) is not None and se.get(name) is not None
            }
            quality = gaps["success_rate"] >= -0.05 and gaps["progress"] >= -0.05
            times = {}
            for algorithm, result in (("ppo", ppo), ("supermimic", sm)):
                curve = Path(result["run"]) / "curves.json"
                if not curve.exists():
                    continue
                rows = json.loads(curve.read_text())["rows"]
                passing = [
                    row
                    for row in rows
                    if row["evaluation"]["success_rate"] >= pe["success_rate"] - 0.05
                    and row["evaluation"]["progress"] >= pe["progress"] - 0.05
                ]
                times[algorithm] = min(
                    (row["training_process_wall_seconds"] for row in passing), default=None
                )
            pairs.append(
                {
                    "clip": Path(sm["data"]).stem,
                    "seed": sm["seed"],
                    "preset": sm["preset"],
                    "ppo_run": ppo["run"],
                    "supermimic_run": sm["run"],
                    "real_steps": sm["metrics"]["real_steps"],
                    "gaps_supermimic_minus_ppo": gaps,
                    "quality_within_tolerance": quality,
                    "ppo_training_process_wall_seconds": ppo["training_process_wall_seconds"],
                    "supermimic_training_process_wall_seconds": sm["training_process_wall_seconds"],
                    "wall_ratio_ppo_over_supermimic": ppo["training_process_wall_seconds"]
                    / sm["training_process_wall_seconds"],
                    "first_observed_seconds_to_ppo_final_success_progress": times,
                    "ppo": pe,
                    "supermimic": se,
                }
            )
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--paper-reference", type=Path)
    args = parser.parse_args()
    identity = json.loads((args.suite_dir / "suite_manifest.json").read_text())
    results = []
    for path in args.suite_dir.glob("*/*/*/completion.json"):
        result = json.loads(path.read_text())
        if result.get("state") == "evaluated":
            if result.get("commit") != identity["commit"]:
                raise ValueError(f"Mixed revisions: {path}")
            results.append(result)
    # Calibration includes four presets and a different budget; report it
    # separately from pilot/full and never pool its quality or timing estimates.
    stages = {
        stage: matched_pairs([r for r in results if Path(r["run"]).parents[1].name == stage])
        for stage in ("calibration", "pilot", "full")
    }
    out = args.output or args.suite_dir / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "commit": identity["commit"],
        "completed_runs": len(results),
        "stages": stages,
        "paper_reference": json.loads(args.paper_reference.read_text())
        if args.paper_reference
        else None,
        "limitations": [
            "Ratios include application startup, compilation and checkpoint saving.",
            "Time to quality uses coarse pilot checkpoint samples; earlier unsampled crossings are unknown.",
            "Published baselines are contextual references, not runs on this machine.",
            "Hardware and released-code observation differences must accompany paper comparisons.",
        ],
    }
    gate = args.suite_dir / "pilot_gate.json"
    if gate.exists():
        report["pilot_gate"] = json.loads(gate.read_text())
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    for stage, pairs in stages.items():
        for p in pairs:
            row = {
                "stage": stage,
                "clip": p["clip"],
                "seed": p["seed"],
                "preset": p["preset"],
                "real_steps": p["real_steps"],
                "quality_within_tolerance": p["quality_within_tolerance"],
                "wall_ratio_ppo_over_supermimic": p["wall_ratio_ppo_over_supermimic"],
            }
            for algorithm in ("ppo", "supermimic"):
                row.update({algorithm + "_" + name: p[algorithm].get(name) for name in METRICS})
            rows.append(row)
    if rows:
        with (out / "paired_results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        f"ConTrack / SuperMimic comparison at {identity['commit']}",
        "",
        f"Completed formal runs: {len(results)}. Matched pairs: {sum(map(len, stages.values()))}.",
        "",
        f"Pilot quality gate: {report.get('pilot_gate', {}).get('passed', 'pending')}.",
        "",
        "Published values are references; measured values below use the same local evaluation protocol.",
        "",
        "| Stage | Clip | Preset | Seeds | Success gap | Progress gap | PPO / SM wall time |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for stage, pairs in stages.items():
        groups = {}
        for p in pairs:
            groups.setdefault((p["clip"], p["preset"]), []).append(p)
        for (clip, preset), group in sorted(groups.items()):

            def avg(field):
                return statistics.mean(p[field] for p in group)

            success = statistics.mean(p["gaps_supermimic_minus_ppo"]["success_rate"] for p in group)
            progress = statistics.mean(p["gaps_supermimic_minus_ppo"]["progress"] for p in group)
            ratio = avg("ppo_training_process_wall_seconds") / avg(
                "supermimic_training_process_wall_seconds"
            )
            lines.append(
                f"| {stage} | {clip} | {preset} | {len(group)} | {success:+.3f} | {progress:+.3f} | {ratio:.3f} |"
            )
    lines.extend(["", "Limitations:", ""] + ["- " + v for v in report["limitations"]])
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps({"output": str(out), "completed_runs": len(results), "matched_pairs": len(rows)})
    )


if __name__ == "__main__":
    main()
