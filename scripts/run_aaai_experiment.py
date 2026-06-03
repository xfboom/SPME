from __future__ import annotations

import argparse
from pathlib import Path

from scripts.aaai_utils import dump_config, load_config, make_run_id, root_path
from scripts.aggregate_aaai_results import aggregate_aaai_results
from scripts.build_adv_sets import build_adv_sets
from scripts.build_appe_splits import build_appe_splits
from scripts.plot_evolution_curve import plot_evolution_curve
from scripts.run_aaai_ablation import run_aaai_ablation
from scripts.run_aaai_baselines import run_aaai_baselines
from scripts.run_appe import run_appe


def run_aaai_experiment(config: dict, *, mode: str, output_dir: str | Path | None = None) -> Path:
    seed = int(config.get("seed", 42))
    base_out = root_path(output_dir or config.get("outputs", {}).get("dir", "outputs/appe_aaai"))
    run_id = make_run_id(seed, mode)
    run_dir = base_out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dump_config(run_dir / "config.yaml", config)
    print(f"[AAAI] Run directory: {run_dir}")
    print("[AAAI] 1/7 build clean splits")
    build_appe_splits(config)
    print("[AAAI] 2/7 build seen/unseen adversarial sets")
    build_adv_sets(config)
    print("[AAAI] 3/7 run APPE full")
    appe_metrics = run_appe(config, variant="full", output_dir=run_dir)
    print("[AAAI] 4/7 run main baselines")
    run_aaai_baselines(config, output_dir=run_dir, appe_metrics=appe_metrics)
    print("[AAAI] 5/7 run ablations")
    run_aaai_ablation(config, output_dir=run_dir, full_metrics=appe_metrics)
    print("[AAAI] 6/7 plot evolution curve")
    plot_evolution_curve(run_dir)
    print("[AAAI] 7/7 aggregate results")
    aggregate_aaai_results(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/appe_aaai.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["full", "smoke", "real-smoke"], default="full")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config, seed=args.seed, mode=args.mode)
    run_dir = run_aaai_experiment(config, mode=args.mode, output_dir=args.output_dir)
    print(f"[AAAI] Done: {run_dir}")


if __name__ == "__main__":
    main()

