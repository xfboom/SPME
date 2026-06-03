from __future__ import annotations

import argparse
from pathlib import Path

from scripts.aaai_utils import load_config, root_path, rounded_row, write_table_files
from scripts.run_appe import run_appe


ABLATION_COLUMNS = [
    "variant",
    "clean_acc",
    "seen_adv_acc",
    "unseen_adv_acc",
    "asr",
    "tokens",
    "active_patch_count",
]


VARIANTS = [
    ("APPE full", "full"),
    ("w/o Failure Ontology", "no_ontology"),
    ("w/o Meta-Controller", "no_meta_controller"),
    ("Single Blue Agent", "single_blue_agent"),
]


def run_aaai_ablation(
    config: dict,
    *,
    output_dir: str | Path,
    full_metrics: dict | None = None,
) -> list[dict]:
    out_dir = root_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (display_name, variant) in enumerate(VARIANTS, start=1):
        print(f"[Ablation] {idx}/{len(VARIANTS)} running {display_name}", flush=True)
        if variant == "full" and full_metrics is not None:
            metrics = dict(full_metrics)
        else:
            metrics = run_appe(config, variant=variant, output_dir=out_dir / "ablations" / variant)
        row = {
            "variant": display_name,
            "clean_acc": metrics.get("clean_acc", 0.0),
            "seen_adv_acc": metrics.get("seen_adv_acc", 0.0),
            "unseen_adv_acc": metrics.get("unseen_adv_acc", 0.0),
            "asr": metrics.get("asr", 0.0),
            "tokens": metrics.get("tokens", 0),
            "active_patch_count": metrics.get("active_patch_count", 0),
        }
        rows.append(rounded_row(row))
    write_table_files(out_dir / "ablation_results.csv", rows, ABLATION_COLUMNS)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/appe_aaai.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", choices=["full", "smoke", "real-smoke"], default="full")
    parser.add_argument("--output-dir", default="outputs/appe_aaai")
    args = parser.parse_args()
    config = load_config(args.config, seed=args.seed, mode=args.mode)
    rows = run_aaai_ablation(config, output_dir=args.output_dir)
    print(rows)


if __name__ == "__main__":
    main()

