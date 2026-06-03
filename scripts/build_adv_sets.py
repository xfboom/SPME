from __future__ import annotations

import argparse

from scripts.aaai_utils import (
    build_adv_samples,
    canonical_attack_type,
    load_config,
    load_json,
    root_path,
    save_json,
)


def build_adv_sets(config: dict) -> dict[str, list[dict]]:
    data_cfg = config.get("data", {})
    attack_cfg = config.get("attacks", {})
    seed = int(config.get("seed", data_cfg.get("seed", 42)))
    split_dir = root_path(data_cfg.get("split_dir", "data/appe_splits"))
    clean_test = load_json(split_dir / "test_clean.json")

    seen_attacks = [canonical_attack_type(name) for name in attack_cfg.get("seen", [])]
    unseen_attacks = [canonical_attack_type(name) for name in attack_cfg.get("unseen", [])]

    seen_adv = build_adv_samples(
        clean_test,
        seen_attacks,
        seed=seed,
        is_seen_attack=True,
        prefix="test_seen_adv",
    )
    unseen_adv = build_adv_samples(
        clean_test,
        unseen_attacks,
        seed=seed + 1,
        is_seen_attack=False,
        prefix="test_unseen_adv",
    )

    save_json(split_dir / "test_seen_adv.json", seen_adv)
    save_json(split_dir / "test_unseen_adv.json", unseen_adv)
    save_json(
        split_dir / "adv_manifest.json",
        {
            "seen_attacks": seen_attacks,
            "unseen_attacks": unseen_attacks,
            "seen_count": len(seen_adv),
            "unseen_count": len(unseen_adv),
            "note": "Unseen attacks are reserved for final evaluation and must not be used during APPE evolution.",
        },
    )
    print(f"[APPE adv] wrote seen/unseen adversarial sets to {split_dir}")
    print(f"[APPE adv] seen={len(seen_adv)}, unseen={len(unseen_adv)}")
    return {"seen_adv": seen_adv, "unseen_adv": unseen_adv}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/appe_aaai.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", choices=["full", "smoke", "real-smoke"], default="full")
    args = parser.parse_args()
    config = load_config(args.config, seed=args.seed, mode=args.mode)
    build_adv_sets(config)


if __name__ == "__main__":
    main()

