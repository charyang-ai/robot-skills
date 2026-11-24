"""Export the dashboard's train_log.json and meta.json from the real RLinf runs.

Run inside the rlinf-robotwin container (needs the tensorboard package, e.g. `source switch_env lingbotvla`):

    python export_trainlogs.py --out /tmp/roborl_weights

then copy <out>/<task>/{train_log.json,meta.json} over RoboRLDemo/weights/<task>/.
Steps are global steps (TensorBoard step + 1), which is how checkpoint folders are named.
"""
import argparse
import json
import os
import re

TB_ROOT = "/workspace/RLinf/logs/robotwin_logs"
CKPT_ROOT = "/opt/envs/robotwin/assets/rlinf_logs"

RUNS = {
    "place_shoe": {
        "run_id": "20261009-09:26:27-robotwin_place_shoe_grpo_lingbotvla",
        "ckpt_sub": "robotwin_grpo_lingbotvla",
        "step": 220,
        "algo": "GRPO",
        "model": "LingBot-VLA",
    },
    "place_empty_cup": {
        "run_id": "20260924-03:29:38-robotwin_place_empty_cup_grpo_openvlaoft",
        "ckpt_sub": "full_pin_64env_mb8_v10",
        "step": 140,
        "algo": "GRPO",
        "model": "OpenVLA-OFT",
    },
    "pick_dual_bottles": {
        "run_id": "20261011-07:57:11-robotwin_pick_dual_bottles_grpo_openvlaoft",
        "ckpt_sub": "robotwin_grpo_openvlaoft",
        "step": 260,
        "algo": "GRPO",
        "model": "OpenVLA-OFT",
    },
}
# The grid entry is the same place_empty_cup run with a different preview video.
ALIASES = {"place_empty_cup_grid": "place_empty_cup"}

TAGS = {
    "loss": "train/actor/policy_loss_abs",
    "accuracy": "eval/success_once",
    "success": "env/success_once",
    "reward": "env/return",
}
LOG_NOTE = (
    "Real RLinf GRPO metrics from TensorBoard; step = global step (TensorBoard step + 1). "
    "loss=policy_loss_abs, accuracy=eval/success_once (ffill), success=env/success_once, reward=env/return."
)
META_NOTE = "Real checkpoint stays in the container (too large to serve); the simulation test loads it there."
WEIGHTS = "actor/model_state_dict/full_weights.pt"


def build_log(run, tb_dir, loss, env_suc, env_ret, eval_suc, saved_steps):
    """Scalar dicts are keyed by TensorBoard step; accuracy carries the last eval forward."""
    series = []
    last_eval = None
    for s in sorted(set(loss) | set(env_suc) | set(env_ret)):
        if s in eval_suc:
            last_eval = eval_suc[s]
        acc = last_eval if last_eval is not None else env_suc.get(s, 0.0)
        prev_loss = series[-1]["loss"] if series else 0.0
        series.append({
            "step": s + 1,
            "loss": round(loss.get(s, prev_loss), 6),
            "accuracy": round(float(acc), 6),
            "success": round(float(env_suc.get(s, acc)), 6),
            "reward": round(float(env_ret.get(s, 0.0)), 6),
        })
    by_step = {p["step"]: p for p in series}
    saved = set(saved_steps)
    checkpoints = []
    for s in sorted(eval_suc):
        p = by_step[s + 1]
        checkpoints.append({
            "step": s + 1,
            "loss": p["loss"],
            "accuracy": round(float(eval_suc[s]), 6),
            "success": p["success"],
            "reward": p["reward"],
            "saved": s + 1 in saved,
        })
    return {
        "note": LOG_NOTE,
        "source": {
            "tensorboard": tb_dir,
            "run_id": run["run_id"],
            "algo": run["algo"],
            "model": run["model"],
            "tags": dict(TAGS),
        },
        "total_steps": series[-1]["step"],
        "series": series,
        "checkpoints": checkpoints,
        "global_steps": True,
    }


def build_meta(task, run, total_steps, ckpt_dir, weights_bytes, dir_bytes, saved_steps):
    step = run["step"]
    return {
        "task_id": task,
        "format": "rlinf-tensorboard",
        "algo": run["algo"],
        "model": run["model"],
        "total_steps": total_steps,
        "run_id": run["run_id"],
        "note": META_NOTE,
        "checkpoint": {
            "step": step,
            "name": f"global_step_{step}/{WEIGHTS}",
            "path": f"{ckpt_dir}/global_step_{step}",
            "weights_bytes": weights_bytes,
            "dir_bytes": dir_bytes,
            "saved_steps": saved_steps,
        },
    }


def apparent_size(path):
    """Same total as `du -sb`: file sizes plus directory entry sizes."""
    total = os.lstat(path).st_size
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            total += os.lstat(os.path.join(root, name)).st_size
    return total


def read_scalars(tb_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    get = lambda tag: {int(e.step): float(e.value) for e in ea.Scalars(tag)}
    return get(TAGS["loss"]), get(TAGS["success"]), get(TAGS["reward"]), get(TAGS["accuracy"])


def write(out, task, log, meta):
    os.makedirs(os.path.join(out, task), exist_ok=True)
    with open(os.path.join(out, task, "train_log.json"), "w") as f:
        f.write(json.dumps(log, separators=(",", ":")) + "\n")
    with open(os.path.join(out, task, "meta.json"), "w") as f:
        f.write(json.dumps(meta, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/tmp/roborl_weights")
    args = ap.parse_args()
    built = {}
    for task, run in RUNS.items():
        tb_dir = f"{TB_ROOT}/{task}/{run['run_id']}/tensorboard"
        ckpt_dir = f"{CKPT_ROOT}/{run['run_id']}/{run['ckpt_sub']}/checkpoints"
        saved_steps = sorted(
            int(m.group(1)) for m in (re.fullmatch(r"global_step_(\d+)", d) for d in os.listdir(ckpt_dir)) if m
        )
        deployed = f"{ckpt_dir}/global_step_{run['step']}"
        log = build_log(run, tb_dir, *read_scalars(tb_dir), saved_steps)
        meta = build_meta(task, run, log["total_steps"], ckpt_dir,
                          os.path.getsize(f"{deployed}/{WEIGHTS}"), apparent_size(deployed), saved_steps)
        built[task] = (log, meta)
        write(args.out, task, log, meta)
        last = log["series"][-1]
        print(f"{task}: {len(log['series'])} steps, {len(log['checkpoints'])} evals, "
              f"saved {saved_steps}, last eval {log['checkpoints'][-1]['accuracy']:.3f}, train success {last['success']:.3f}")
    for alias, task in ALIASES.items():
        log, meta = built[task]
        write(args.out, alias, log, dict(meta, task_id=alias))
        print(f"{alias}: copy of {task}")


if __name__ == "__main__":
    main()
