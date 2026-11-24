# robot-skills

Robot manipulation skills trained with reinforcement learning on AMD Radeon™ GPUs.
Each skill is a vision-language-action (VLA) policy fine-tuned with GRPO in
[RLinf](https://github.com/RLinf/RLinf) and rolled out in the dual-arm
[RoboTwin 2.0](https://robotwin-platform.github.io/) simulator.

This repo is the skill library behind the Robot RL Training Workbench demo
(PyTorch Conference 2026, AMD Demo Zoo; 8× Radeon™ PRO W7900 on Radeon Cloud).
Project page: <https://worldloop.github.io/projects/rl-workbench/>

## Skills

| Skill | What the robot does | Policy | Training steps | Train success (first → last) | Last eval success |
| --- | --- | --- | --- | --- | --- |
| [`pick_dual_bottles`](skills/pick_dual_bottles) | Both arms grasp and lift two bottles | OpenVLA-OFT | 260 | 6.4% → 91.0% | 89.1% (step 260) |
| [`place_shoe`](skills/place_shoe) | Pick up a shoe and set it on the target mat | [LingBot-VLA](https://github.com/Robbyant/lingbot-vla-v2) | 225 | 97.3% → 96.5% | 96.9% (step 220) |
| [`place_empty_cup`](skills/place_empty_cup) | Place an empty cup on the coaster | OpenVLA-OFT | 147 | 71.7% → 85.0% | 87.5% (step 140) |

Numbers come straight from the `train_log.json` files (run `python tools/summarize.py` to reproduce the table).
"Train success" is `env/success_once` from the RLinf rollouts; "eval success" is `eval/success_once` at the last saved checkpoint.
`place_shoe` starts from an already strong policy, so GRPO mostly holds and polishes it rather than learning from scratch.

## Layout

```
skills/
  skills.json              registry of all skills
  <skill>/
    train_log.json         real RLinf TensorBoard export (loss, success, reward, eval accuracy per step)
    demo.mp4               simulation rollout of the trained policy
tools/
  summarize.py             print the summary table from the logs
  export_trainlogs.py      export train_log.json from an RLinf run (run inside the rlinf-robotwin container)
```

## What is not in this repo

- Model checkpoints (8–15 GB each). They stay in the training container and are not published here.
- RLinf and RoboTwin themselves. Use the upstream projects; ROCm support is merged in RLinf
  ([PR #1762](https://github.com/RLinf/RLinf/pull/1762)).

## Reproduce on Radeon

Follow the step-by-step guide for RoboTwin + RLinf on Radeon Cloud:
<https://github.com/ZiguanWang/Robotwin-radeon-cloud>

## Adding a skill

1. Train it with RLinf on a RoboTwin 2.0 task.
2. Run `tools/export_trainlogs.py` inside the container to produce `train_log.json` (add the run to its `RUNS` table).
3. Drop `train_log.json` and a rollout `demo.mp4` in `skills/<skill>/` and add an entry to `skills/skills.json`.

## Acknowledgements

[RLinf](https://github.com/RLinf/RLinf) (Infinigence AI), [LingBot-VLA](https://github.com/Robbyant/lingbot-vla-v2) (Robbyant),
[RoboTwin 2.0](https://robotwin-platform.github.io/) (Lumina embodied-AI community).
