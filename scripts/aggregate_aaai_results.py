from __future__ import annotations

import argparse
from pathlib import Path

from scripts.aaai_utils import root_path


def aggregate_aaai_results(output_dir: str | Path) -> dict:
    out_dir = root_path(output_dir)
    files = {
        "main_results": out_dir / "main_results.csv",
        "ablation_results": out_dir / "ablation_results.csv",
        "evolution_curve": out_dir / "evolution_curve.csv",
        "final_prompt": out_dir / "final_prompt.txt",
        "patch_memory": out_dir / "patch_memory.jsonl",
    }
    summary = {name: path.exists() for name, path in files.items()}
    missing = [name for name, exists in summary.items() if not exists]
    if missing:
        print(f"[Aggregate] Missing outputs: {missing}")
    else:
        print(f"[Aggregate] Required AAAI outputs are present in {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/appe_aaai")
    args = parser.parse_args()
    aggregate_aaai_results(args.output_dir)


if __name__ == "__main__":
    main()
