"""Train ACT (lerobot 768e366) on one dataset. Based on pai_training/scripts/train.py (b4d6b54).

Usage: CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_act.py --dataset B --seed 0 --run-id 003_B_seed0
"""

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType

from crane_reach.dataset import FEATURES, FPS
from crane_reach.sim import EXP_ROOT


def dir_sha(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=EXP_ROOT, capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)        # A / B / C
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--exp", default="003_demo_diversity")
    ap.add_argument("--config", default="configs/train_act.yaml")
    ap.add_argument("--steps", type=int, default=None)  # optional training-step override
    args = ap.parse_args()

    cfg = yaml.safe_load(open(EXP_ROOT / args.config))
    steps = args.steps or cfg["training_steps"]
    save_steps = sorted({s for s in cfg["save_steps"] if s <= steps} | {steps})
    out = EXP_ROOT / "artifacts" / args.exp / "runs" / args.run_id
    if out.exists():
        raise SystemExit(f"{out} exists: use a new run id")
    out.mkdir(parents=True)
    ds_root = EXP_ROOT / "artifacts/003_demo_diversity/datasets" / f"crane_reach_{args.dataset}"
    repo_id = f"crane_reach_{args.dataset}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(cfg["device"])

    meta = LeRobotDatasetMetadata(repo_id, root=ds_root)
    features = dataset_to_policy_features(FEATURES)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}
    act_cfg = ACTConfig(input_features=input_features, output_features=output_features,
                        chunk_size=cfg["chunk_size"], n_action_steps=cfg["n_action_steps"])
    policy = ACTPolicy(act_cfg, dataset_stats=meta.stats).to(device)
    policy.train()

    delta = {"action": [k / FPS for k in range(cfg["chunk_size"])]}
    dataset = LeRobotDataset(repo_id, root=ds_root, delta_timestamps=delta)
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                                         num_workers=cfg["num_workers"], generator=g, pin_memory=True)
    optim = torch.optim.Adam(policy.parameters(), lr=cfg["learning_rate"])

    info = {
        "run_id": args.run_id, "dataset": args.dataset, "seed": args.seed, "steps": steps,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": f"uv run python scripts/train_act.py --dataset {args.dataset} --seed {args.seed} --run-id {args.run_id}"
                   + (f" --steps {args.steps}" if args.steps else ""),
        "git_commit": git("rev-parse", "HEAD").strip(), "git_dirty": bool(git("status", "--porcelain", ".").strip()),
        "config": cfg, "delta_timestamps": delta, "act_config": {k: str(v) for k, v in vars(act_cfg).items()},
        "dataset_root": str(ds_root.relative_to(EXP_ROOT)), "dataset_sha256": dir_sha(ds_root),
        "dataset_episodes": meta.total_episodes, "dataset_frames": meta.total_frames,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
        "upstream": {"pai_training": "b4d6b5459066a80433475a83567504988875248a",
                     "lerobot": "768e36660d1408c71118a2760f831c037fbfa17d"},
    }
    if info["git_dirty"]:
        (out / "uncommitted.diff").write_text(git("diff", "HEAD", "--", "."))

    log = []
    step = 0
    t0 = time.time()
    while step < steps:
        for batch in loader:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, ld = policy.forward(batch)
            loss.backward()
            optim.step()
            optim.zero_grad()
            step += 1
            if step % cfg["log_freq"] == 0:
                log.append({"step": step, "loss": loss.item(), **{k: float(v) for k, v in ld.items()}})
            if step in save_steps:
                policy.save_pretrained(out / f"step_{step:06d}")
            if step >= steps:
                break
    info["train_seconds"] = round(time.time() - t0, 1)
    info["completed_steps"] = step
    sel = cfg.get("selected_step", steps)
    info["selected_step"] = sel
    info["model_selection"] = f"model at configured step {sel}"
    info["final_model"] = str((out / f"step_{sel:06d}").relative_to(EXP_ROOT))
    json.dump(info, open(out / "run_info.json", "w"), indent=2)
    json.dump(log, open(out / "loss_log.json", "w"))
    print(f"done {args.run_id}: {step} steps in {info['train_seconds']} s, final loss {log[-1]['loss']:.4f}")


if __name__ == "__main__":
    main()
