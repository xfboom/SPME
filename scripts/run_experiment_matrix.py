from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from config.experiments import EXPERIMENT_PRESETS, resolve_experiment_flags
from config.settings import OUTPUT_DIR


def _build_run_dir(root: Path, preset: str, seed: int) -> Path:
    return root / preset / f"seed_{seed}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an APO experiment matrix across presets and seeds.")
    parser.add_argument("--presets", nargs="*", default=["full", "baseline_static", "no_textual_loss", "no_meta_controller", "no_regression", "no_memory", "no_pruning", "no_abstraction", "red_only", "patch_only"], help="Experiment presets to run.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2], help="Random seeds to repeat.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    root = Path(OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)

    for preset in args.presets:
        if preset not in EXPERIMENT_PRESETS:
            print(f"[Skip] Unknown preset: {preset}")
            continue
        flags = resolve_experiment_flags(preset)
        for seed in args.seeds:
            run_dir = _build_run_dir(root, preset, seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(
                {
                    "EXPERIMENT_PRESET": preset,
                    "OUTPUT_DIR": str(run_dir),
                    "EVAL_RANDOM_SEED": str(seed),
                    "BENCHMARK_SEED": str(seed),
                }
            )
            print(f"[Run] preset={preset} seed={seed} dir={run_dir} flags={asdict(flags)}")
            if args.dry_run:
                continue
            subprocess.run([sys.executable, "main.py"], check=True, env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())