#!/usr/bin/env python3
"""Submit the preregistered cases and summarize them at hourly intervals."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from multitask_case import (
    PLAN_PATH,
    SOURCE,
    check_source,
    digest,
    git,
    last_metrics,
    stamp,
    write_json,
)

TERMINAL = {"Succeeded", "Failed", "Stopped", "SucceededReserving", "FailedReserving"}
IMAGE = "dsw-registry-vpc.cn-beijing.cr.aliyuncs.com/pai-training-algorithm/isaaclab:2.3.2-isaacsim5.1.0-py311-ubuntu24.04"


def api(method, *args):
    command = ["aliyun", "pai-dlc", method, "--region", "cn-beijing", *args]
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"{method} failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def short_task(task):
    if task.startswith("dexterhand-"):
        return task.split("-")[1].lower()
    return task.split("_")[1].lower()


def submit(root, plan, selected):
    commit = git(SOURCE, "rev-parse", "HEAD")
    check_source(SOURCE, plan, commit)
    receipt = root / "submissions.json"
    submissions = json.loads(receipt.read_text()) if receipt.exists() else []
    for task in plan["tasks"]:
        if selected and task["task"] not in selected:
            continue
        for algorithm in ("supermimic", "ppo"):
            label = f"{task['task']}__{algorithm}"
            existing = [item for item in submissions if item["case"] == label]
            if existing:
                if existing[0]["source_commit"] != commit:
                    previous_repo = Path(existing[0]["case_root"]) / "repo"
                    check_source(previous_repo, plan, existing[0]["source_commit"])
                    entry = "scripts/supermimic/multitask_case.py"
                    if digest(previous_repo / entry) != digest(SOURCE / entry):
                        raise RuntimeError("Existing case used a different GPU training harness")
                continue
            prefix = plan.get("job_prefix", "smmt1109")
            display = f"{prefix}_{short_task(task['task'])}_{algorithm}_{commit[:7]}"
            case_root = root / "cases" / f"{label}__{commit[:7]}"
            command = [
                "python3",
                str(SOURCE / "scripts/supermimic/multitask_case.py"),
                "--root",
                str(case_root),
                "--task",
                task["task"],
                "--algorithm",
                algorithm,
                "--expected-commit",
                commit,
                "--resources",
                str(root.parents[1] / "releases/Supermimic-dex"),
                "--plan",
                str(SOURCE / plan.get("plan_file", str(PLAN_PATH.relative_to(SOURCE)))),
            ]
            # Every argument is a known path/name without whitespace; use shlex
            # to preserve those arguments when DLC passes UserCommand to a shell.
            import shlex

            request = dict(
                WorkspaceId="464179",
                ResourceId=plan["resource_id"],
                DisplayName=display,
                JobType="PyTorchJob",
                JobMaxRunningTimeMinutes=720,
                Accessibility="PRIVATE",
                Priority=1,
                JobSpecs=[
                    dict(
                        Type="Worker",
                        Image=IMAGE,
                        PodCount=1,
                        ResourceConfig=dict(CPU="16", GPU="1", Memory="128Gi", SharedMemory="32Gi"),
                    )
                ],
                DataSources=[
                    dict(
                        DataSourceId="d-w56t9d68yj06l09u5p",
                        DataSourceVersion="v1",
                        MountPath="/mnt/project/public",
                        MountAccess="RW",
                    )
                ],
                UserCommand=shlex.join(command),
            )
            if (
                any("AssignNodeSpec" in spec for spec in request["JobSpecs"])
                or "Settings" in request
            ):
                raise RuntimeError("Obsolete driver/node scheduling overrides are forbidden")
            write_json(root / "requests" / f"{label}.json", request)
            response = api("CreateJob", "--body", json.dumps(request))
            if not response.get("JobId"):
                write_json(root / "requests" / f"{label}.response.json", response)
                raise RuntimeError("CreateJob returned no JobId; inspect before retrying")
            item = dict(
                case=label,
                task=task["task"],
                algorithm=algorithm,
                job_id=response["JobId"],
                display_name=display,
                source_commit=commit,
                submitted_utc=stamp(),
                case_root=str(case_root),
                resource=plan["resource"],
            )
            submissions.append(item)
            write_json(receipt, submissions)
            print(json.dumps(item), flush=True)
    return submissions


def summarize(root, submissions):
    saved_plan = root / "suite_plan.json"
    plan = json.loads((saved_plan if saved_plan.exists() else PLAN_PATH).read_text())
    expected_cases = len(plan["tasks"]) * 2
    response = api(
        "ListJobs",
        "--WorkspaceId",
        "464179",
        "--ResourceId",
        plan["resource_id"],
        "--DisplayName",
        plan.get("job_prefix", "smmt1109") + "_",
        "--PageSize",
        "100",
    )
    statuses = {item["JobId"]: item["Status"] for item in response.get("Jobs", [])}
    rows = []
    for item in submissions:
        case = Path(item["case_root"])
        status = statuses.get(item["job_id"], "Unknown")
        completion_path = case / "completion.json"
        completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
        state_path = case / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        metrics = last_metrics(case / "run")
        evaluation_path = case / "run/evaluation.json"
        evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
        failure_path = case / "failure.json"
        failure = json.loads(failure_path.read_text()) if failure_path.exists() else None
        wm_path = case / "run/wm_evaluation.json"
        wm = json.loads(wm_path.read_text()) if wm_path.exists() else None
        # A cloud job success alone cannot stand in for evaluated policy artifacts.
        evaluated = completion.get("state") == "evaluated" and evaluation.get("episodes") == 4096
        row = dict(
            task=item["task"],
            algorithm=item["algorithm"],
            job_id=item["job_id"],
            cloud_status=status,
            evaluated=evaluated,
            phase=state.get("phase"),
            metrics=metrics,
            gpu=state.get("gpu"),
            last_check=state.get("checked_utc"),
            estimated_remaining_seconds=metrics.get(
                "estimated_remaining_seconds",
                metrics.get("wall_seconds", 0)
                * max(0, plan["budget_updates"] * 16 / metrics["vector_steps"] - 1)
                if metrics.get("vector_steps")
                else state.get("estimated_remaining_seconds"),
            ),
            evaluation=evaluation,
            wm_evaluation=wm,
            completion=completion,
            failure=failure,
        )
        rows.append(row)
    done = len(rows) == expected_cases and all(row["cloud_status"] in TERMINAL for row in rows)
    report = dict(
        checked_utc=stamp(),
        submitted=len(rows),
        expected_cases=expected_cases,
        terminal=sum(row["cloud_status"] in TERMINAL for row in rows),
        evaluated=sum(row["evaluated"] for row in rows),
        state="terminal" if done else "active",
        interval_seconds=3600,
        rows=rows,
    )
    write_json(root / "status.json", report)
    with (root / "hourly_checks.jsonl").open("a") as stream:
        stream.write(json.dumps(report) + "\n")
    lines = [
        "# 新任务 BPTT/PPO 对照",
        "",
        f"检查时间：{report['checked_utc']}",
        "",
        f"固定 Hammer e3 配置，从头训练；{plan.get('embodiment', 'xhand')}、单 seed {plan['seed']}，每项 {plan['real_steps_per_run'] / 1000000:g}M 真实交互，最终评估 {plan['evaluation_episodes']} episodes。",
        "训练期 completion fraction 与最终成功率分开记录。世界模型误差是各自策略数据上的诊断，不能直接作跨任务公平排名。",
        "",
        "| 任务 | 方法 | DLC 状态 | 阶段 | PPO 等效 iteration | 最终成功率 |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        success = row["evaluation"].get("success_rate")
        text = f"{success * 100:.2f}%" if success is not None else "待评估"
        lines.append(
            f"| {short_task(row['task'])} | {row['algorithm']} | {row['cloud_status']} | "
            f"{row['phase'] or '排队/启动'} | {row['metrics'].get('equivalent_ppo_iteration', '—')} | {text} |"
        )
    (root / "report_zh.md").write_text("\n".join(lines) + "\n")
    if done:
        write_json(root / "completed.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("checked_utc", "submitted", "terminal", "evaluated", "state")
            }
        ),
        flush=True,
    )
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--task", action="append")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "requests").mkdir(exist_ok=True)
    plan = json.loads(args.plan.read_text())
    write_json(args.root / "suite_plan.json", plan)
    if args.submit:
        submit(args.root, plan, args.task)
    if args.watch:
        write_json(
            args.root / "monitor_process.json",
            dict(
                pid=os.getpid(),
                started_utc=stamp(),
                interval_seconds=3600,
                source_commit=git(SOURCE, "rev-parse", "HEAD"),
            ),
        )
        while True:
            try:
                submissions = json.loads((args.root / "submissions.json").read_text())
                if summarize(args.root, submissions):
                    break
            except Exception as error:
                with (args.root / "monitor_errors.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(checked_utc=stamp(), error=repr(error))) + "\n")
            time.sleep(3600)


if __name__ == "__main__":
    main()
