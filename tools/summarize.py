"""Print a summary table of every skill in skills/skills.json.

    python tools/summarize.py

Reads each skill's train_log.json (real RLinf TensorBoard export) and reports the
first and last training success rate (env/success_once) and the last
evaluation success rate (eval/success_once) at a saved checkpoint.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "skills"


def main():
    registry = json.loads((ROOT / "skills.json").read_text(encoding="utf-8"))
    rows = []
    for sk in registry["skills"]:
        log = json.loads((ROOT / sk["dir"] / "train_log.json").read_text(encoding="utf-8"))
        series = log["series"]
        ckpts = [c for c in log["checkpoints"] if c.get("accuracy") is not None]
        last_eval = ckpts[-1] if ckpts else None
        rows.append(
            (
                sk["id"],
                sk["policy"],
                str(len(series)),
                f"{series[0]['success']:.1%}",
                f"{series[-1]['success']:.1%}",
                f"{last_eval['accuracy']:.1%} @ step {last_eval['step']}" if last_eval else "n/a",
            )
        )
    head = ("skill", "policy", "steps", "train success (first)", "train success (last)", "last eval success")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths))
    print(line(head))
    print(line(tuple("-" * w for w in widths)))
    for r in rows:
        print(line(r))


if __name__ == "__main__":
    main()
