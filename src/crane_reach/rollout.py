"""Closed-loop execution of a trained ACT policy (or a command replay) in CraneSim.

Loop per 20 Hz tick: measure joint state -> policy -> send command.
The policy's internal action queue is reset at the start of every trial.
"""

from __future__ import annotations

import numpy as np
import torch

from .sim import CraneSim, SuccessTracker


def load_policy(path, device: str = "cuda", execution: str = "chunk1"):
    """execution: how the predicted 1 s action chunk is executed (inference-side only)
      "chunk1"     : re-plan every tick and execute chunk[0] (n_action_steps=1)
      "chunk<N>"   : execute the first N actions of a chunk open-loop, then re-plan
      "ensemble"   : temporal ensembling over overlapping chunks (ACT paper, coeff 0.01)
    """
    from lerobot.common.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler

    policy = ACTPolicy.from_pretrained(str(path))
    if execution == "ensemble":
        policy.config.n_action_steps = 1
        policy.config.temporal_ensemble_coeff = 0.01
        policy.temporal_ensembler = ACTTemporalEnsembler(0.01, policy.config.chunk_size)
    else:
        policy.config.n_action_steps = int(execution.removeprefix("chunk"))
    policy.eval()
    policy.to(device)
    return policy


class ACTController:
    def __init__(self, policy, device: str = "cuda"):
        self.policy = policy
        self.device = device

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def __call__(self, qpos: np.ndarray, k: int) -> np.ndarray:
        s = torch.as_tensor(qpos, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.policy.select_action({"observation.state": s, "observation.environment_state": s})
        return a[0].cpu().numpy().astype(np.float64)


class ReplayController:
    """Non-AI reference: replays a recorded command sequence open-loop (holds the last command)."""

    def __init__(self, actions: np.ndarray):
        self.actions = np.asarray(actions, float)

    def reset(self):
        pass

    def __call__(self, qpos: np.ndarray, k: int) -> np.ndarray:
        return self.actions[min(k, len(self.actions) - 1)]


def run_trial(sim: CraneSim, controller, q0, renderer=None, camera="front", frame_every: int = 1) -> dict:
    task = sim.task
    info = sim.reset(q0)
    start_measured = info.qpos.copy()
    controller.reset()
    tracker = SuccessTracker(task)
    qs, cmds, tips, errs, ts, frames = [], [], [], [], [], []
    table = False
    self_min = np.inf
    outcome = None
    k = 0
    while outcome is None:
        if renderer is not None and k % frame_every == 0:
            renderer.update_scene(sim.data, camera=camera)
            frames.append(renderer.render().copy())
        cmd = controller(info.qpos, k)
        qs.append(info.qpos)
        cmds.append(cmd)
        tips.append(info.tip)
        errs.append(info.err)
        ts.append(info.t)
        info = sim.command(cmd)
        table |= info.table_contact
        self_min = min(self_min, info.self_min_dist)
        outcome = tracker.update(info)
        k += 1
    if renderer is not None:
        renderer.update_scene(sim.data, camera=camera)
        frames.append(renderer.render().copy())
    qs.append(info.qpos)
    tips.append(info.tip)
    errs.append(info.err)
    ts.append(info.t)
    return {
        "outcome": outcome,
        "end_time": float(info.t),
        "final_err_cm": 100 * float(info.err),
        "min_err_cm": 100 * float(np.min(errs)),
        "collision": outcome == "collision",
        "contact_geoms": ";".join(info.contact_geoms),
        "table_contact": bool(table),
        "self_min_dist_mm": 1000 * float(self_min),
        "self_penetration": bool(self_min < 0),
        "time_to_success": float(info.t) if outcome == "success" else float("nan"),
        "first_inside": tracker.first_inside if tracker.first_inside is not None else float("nan"),
        "start_measured": start_measured,
        "traj": {"q": np.array(qs, np.float32), "cmd": np.array(cmds, np.float32),
                 "tip": np.array(tips, np.float32), "err": np.array(errs, np.float32),
                 "t": np.array(ts, np.float32)},
        "frames": frames,
    }
