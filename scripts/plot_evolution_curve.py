from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.aaai_utils import root_path


def plot_evolution_curve(output_dir: str | Path) -> None:
    out_dir = root_path(output_dir)
    curve_path = out_dir / "evolution_curve.csv"
    if not curve_path.exists():
        print(f"[Plot] Missing {curve_path}; skipping plot.")
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[Plot] matplotlib is not installed; evolution_curve.png was not generated.")
        return

    rows = []
    with curve_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return
    rounds = [int(row["round"]) for row in rows]
    clean = [float(row["clean_acc"]) for row in rows]
    seen = [float(row["seen_adv_acc"]) for row in rows]
    asr = [float(row["asr"]) for row in rows]

    plt.figure(figsize=(7, 4))
    plt.plot(rounds, clean, marker="o", label="clean_acc")
    plt.plot(rounds, seen, marker="o", label="seen_adv_acc")
    plt.plot(rounds, asr, marker="o", label="asr")
    plt.xlabel("round")
    plt.ylabel("metric")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "evolution_curve.png", dpi=160)
    plt.close()
    print(f"[Plot] wrote {out_dir / 'evolution_curve.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/appe_aaai")
    args = parser.parse_args()
    plot_evolution_curve(args.output_dir)


if __name__ == "__main__":
    main()
