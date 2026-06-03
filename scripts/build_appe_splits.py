from __future__ import annotations

import argparse
import random
from pathlib import Path

from scripts.aaai_utils import load_config, load_json, normalize_samples, root_path, save_json


def _take(rows: list[dict], size: int) -> tuple[list[dict], list[dict]]:
    if size <= 0:
        return [], rows
    actual = min(size, len(rows))
    return rows[:actual], rows[actual:]


def build_appe_splits(config: dict) -> dict[str, list[dict]]:
    data_cfg = config.get("data", {})
    seed = int(data_cfg.get("seed", config.get("seed", 42)))
    rng = random.Random(seed)

    train_rows = normalize_samples(load_json(data_cfg["raw_train_path"]), "train")
    test_rows = normalize_samples(load_json(data_cfg["raw_test_path"]), "test")
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    calib, train_rows = _take(train_rows, int(data_cfg.get("calib_size", 100)))
    evo_clean, train_rows = _take(train_rows, int(data_cfg.get("evo_clean_size", 1000)))
    guard, train_rows = _take(train_rows, int(data_cfg.get("guard_size", 200)))
    test_clean, _ = _take(test_rows, int(data_cfg.get("test_size", 1000)))

    split_dir = root_path(data_cfg.get("split_dir", "data/appe_splits"))
    split_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "calib": calib,
        "evo_clean": evo_clean,
        "guard": guard,
        "test_clean": test_clean,
    }
    for name, rows in splits.items():
        save_json(split_dir / f"{name}.json", rows)

    manifest = {
        "seed": seed,
        "requested": {
            "calib": int(data_cfg.get("calib_size", 100)),
            "evo_clean": int(data_cfg.get("evo_clean_size", 1000)),
            "guard": int(data_cfg.get("guard_size", 200)),
            "test_clean": int(data_cfg.get("test_size", 1000)),
        },
        "actual": {name: len(rows) for name, rows in splits.items()},
        "notes": "D_calib/D_evo_clean/D_guard are sampled from raw_train_path; D_test is sampled from raw_test_path.",
    }
    save_json(split_dir / "manifest.json", manifest)
    print(f"[APPE splits] wrote splits to {split_dir}")
    print(f"[APPE splits] actual sizes: {manifest['actual']}")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/appe_aaai.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", choices=["full", "smoke", "real-smoke"], default="full")
    args = parser.parse_args()
    config = load_config(args.config, seed=args.seed, mode=args.mode)
    build_appe_splits(config)


if __name__ == "__main__":
    main()

