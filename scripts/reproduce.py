"""Run demonstration collection, ACT training, evaluation and analysis.

--dry-run prints the job list without executing it. Completed outputs can be
reused with --skip-existing when source code, settings and input data match.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import os
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("prepare", "train", "evaluate", "analyze")


@dataclass
class Job:
    args: list[str]
    outputs: tuple[str, ...] = ()


def jobs(stage: str):
    main_states = "exp/003_demo_diversity/evaluation_states.csv"
    wide_states = "exp/004_wide_range/states.csv"
    if stage == "prepare":
        yield Job(["exp/000_smoke/run_checks.py"], ("exp/000_smoke/results/checks.json",))
        yield Job(["scripts/make_states.py"], (main_states, "exp/002_obstacle/pilot_states.csv"))
        yield Job(["scripts/make_wide_states.py"], (wide_states,))
        yield Job(["scripts/collect_demos.py"], tuple(
            f"artifacts/003_demo_diversity/datasets/crane_reach_{c}/meta/info.json" for c in "ABC"))
    elif stage == "train":
        for condition in "ABC":
            for seed in range(5):
                run = f"003_{condition}_seed{seed}"
                yield Job(["scripts/train_act.py", "--dataset", condition, "--seed", str(seed), "--run-id", run],
                          tuple(f"artifacts/003_demo_diversity/runs/{run}/{p}" for p in
                                ("run_info.json", "step_002000/model.safetensors", "step_020000/model.safetensors")))
    elif stage == "evaluate":
        for condition in "ABC":
            for seed in range(5):
                run = f"003_{condition}_seed{seed}"
                for exp, states, tag, model in [
                    ("003_demo_diversity", main_states, "eval", "step_002000"),
                    ("003_demo_diversity", main_states, "eval20k", "step_020000"),
                    ("004_wide_range", wide_states, "eval", "step_002000"),
                ]:
                    eid = f"{run}_{tag}" if exp.startswith("003") else f"004_{condition}_seed{seed}_{tag}"
                    yield Job(["scripts/evaluate.py", "--run-id", run, "--model", model,
                               "--states", states, "--eval-id", eid, "--exp", exp],
                              (f"exp/{exp}/results/trials_{eid}.csv", f"artifacts/{exp}/evals/{eid}/eval_info.json"))
        for exp, states, eid in [("003_demo_diversity", main_states, "003_replayB000_eval"),
                                 ("004_wide_range", wide_states, "004_replayB000_eval")]:
            yield Job(["scripts/evaluate.py", "--replay", "B000", "--states", states,
                       "--eval-id", eid, "--exp", exp],
                      (f"exp/{exp}/results/trials_{eid}.csv", f"artifacts/{exp}/evals/{eid}/eval_info.json"))
    elif stage == "analyze":
        for exp, tag in [("003_demo_diversity", "eval"), ("003_demo_diversity", "eval20k"),
                         ("004_wide_range", "eval")]:
            yield Job(["scripts/analyze.py", "--exp", exp, "--tag", tag])
        yield Job(["scripts/make_runs_csv.py"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    ap.add_argument("--gpu", default="0", help="One physical GPU ID for new jobs; jobs run sequentially")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip jobs with all expected outputs; use only with unchanged code/config/data")
    args = ap.parse_args()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    for stage in args.stages:
        print(f"\n[{stage}]", flush=True)
        for job in jobs(stage):
            cmd = ["uv", "run", "python", *job.args]
            if args.skip_existing and job.outputs and all((ROOT / p).is_file() for p in job.outputs):
                print("SKIP " + shlex.join(cmd), flush=True)
                continue
            print(shlex.join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
