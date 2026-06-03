from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from scripts.aaai_utils import (
    accuracy,
    count_tokens,
    evaluate_prompt,
    load_config,
    load_json,
    load_prompt,
    predict_samples,
    root_path,
    rounded_row,
    write_table_files,
)


MAIN_COLUMNS = [
    "method",
    "clean_acc",
    "seen_adv_acc",
    "unseen_adv_acc",
    "asr",
    "clean_drop",
    "tokens",
    "active_patch_count",
]


@dataclass
class ExperimentMethod:
    name: str
    prompt: str
    active_patch_count: int = 0

    def fit(self, splits: dict[str, list[dict]], config: dict) -> None:
        return None

    def get_prompt(self) -> str:
        return self.prompt

    def evaluate(self, config: dict, eval_sets: dict[str, list[dict]], base_clean_acc: float | None) -> dict:
        metrics, _ = evaluate_prompt(
            prompt=self.get_prompt(),
            method=self.name,
            config=config,
            clean_samples=eval_sets["clean"],
            seen_adv_samples=eval_sets["seen_adv"],
            unseen_adv_samples=eval_sets["unseen_adv"],
            base_clean_acc=base_clean_acc,
        )
        metrics["active_patch_count"] = self.active_patch_count
        metrics["tokens"] = count_tokens(self.get_prompt())
        return rounded_row(metrics)


class FullPromptRewrite(ExperimentMethod):
    def fit(self, splits: dict[str, list[dict]], config: dict) -> None:
        # Simplified full rewrite baseline: it rewrites the whole prompt with a compact
        # defensive instruction block instead of using patch memory.
        base = load_prompt(config.get("prompt", {}).get("base_prompt_path", "config/prompts/base_math.txt"))
        self.prompt = (
            base
            + "\n\nGlobal rewrite: filter irrelevant numbers and background, ignore misleading format demands, "
            "and verify the final answer before producing it."
        )


class ReflexionMemory(ExperimentMethod):
    def fit(self, splits: dict[str, list[dict]], config: dict) -> None:
        # Natural-language memory baseline. This intentionally does not use CognitivePatch.
        self.prompt += (
            "\n\nReflections from previous mistakes:\n"
            "- Some questions contain distracting numbers or formatting requests.\n"
            "- Re-read the original task before answering.\n"
            "- Keep the final answer concise."
        )


class SimplifiedPromptOptimizer(ExperimentMethod):
    def fit(self, splits: dict[str, list[dict]], config: dict) -> None:
        candidates = [
            self.prompt,
            load_prompt(config.get("prompt", {}).get("cot_prompt_path", "config/prompts/cot_math.txt")),
            load_prompt(config.get("prompt", {}).get("long_defensive_prompt_path", "config/prompts/long_defensive_math.txt")),
        ]
        seen_adv = splits["seen_adv"][: max(1, min(24, len(splits["seen_adv"])))]
        best_prompt = candidates[0]
        best_acc = -1.0
        for prompt in candidates:
            preds = predict_samples(prompt, seen_adv, self.name, config)
            acc = accuracy(preds)
            if acc > best_acc:
                best_acc = acc
                best_prompt = prompt
        self.prompt = best_prompt


def _load_sets(config: dict) -> dict[str, list[dict]]:
    split_dir = root_path(config.get("data", {}).get("split_dir", "data/appe_splits"))
    return {
        "clean": load_json(split_dir / "test_clean.json"),
        "seen_adv": load_json(split_dir / "test_seen_adv.json"),
        "unseen_adv": load_json(split_dir / "test_unseen_adv.json"),
    }


def build_methods(config: dict) -> list[ExperimentMethod]:
    prompt_cfg = config.get("prompt", {})
    base = load_prompt(prompt_cfg.get("base_prompt_path", "config/prompts/base_math.txt"))
    cot = load_prompt(prompt_cfg.get("cot_prompt_path", "config/prompts/cot_math.txt"))
    long_defensive = load_prompt(
        prompt_cfg.get("long_defensive_prompt_path", "config/prompts/long_defensive_math.txt")
    )
    return [
        ExperimentMethod("Base Prompt", base),
        ExperimentMethod("CoT Prompt", cot),
        ExperimentMethod("Long Defensive Prompt", long_defensive),
        FullPromptRewrite("Full Prompt Rewrite", base),
        ReflexionMemory("Reflexion Memory", base),
        SimplifiedPromptOptimizer("Simplified Prompt Optimizer", base),
    ]


def run_aaai_baselines(
    config: dict,
    *,
    output_dir: str | Path,
    appe_metrics: dict | None = None,
) -> list[dict]:
    out_dir = root_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_sets = _load_sets(config)
    methods = build_methods(config)

    base_metrics, _ = evaluate_prompt(
        prompt=methods[0].prompt,
        method="Base Prompt",
        config=config,
        clean_samples=eval_sets["clean"],
        seen_adv_samples=eval_sets["seen_adv"],
        unseen_adv_samples=eval_sets["unseen_adv"],
    )
    rows = []
    for idx, method in enumerate(methods, start=1):
        print(f"[Baselines] {idx}/{len(methods)} fitting/evaluating {method.name}", flush=True)
        method.fit(eval_sets, config)
        rows.append(method.evaluate(config, eval_sets, base_metrics["clean_acc"]))

    if appe_metrics is not None:
        print("[Baselines] adding APPE row", flush=True)
        appe_row = {col: appe_metrics.get(col, "") for col in MAIN_COLUMNS}
        appe_row["method"] = "APPE"
        rows.append(appe_row)

    order = [
        "Base Prompt",
        "CoT Prompt",
        "Long Defensive Prompt",
        "Full Prompt Rewrite",
        "Reflexion Memory",
        "Simplified Prompt Optimizer",
        "APPE",
    ]
    rows = sorted(rows, key=lambda row: order.index(row["method"]) if row["method"] in order else 999)
    write_table_files(out_dir / "main_results.csv", rows, MAIN_COLUMNS)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/appe_aaai.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", choices=["full", "smoke", "real-smoke"], default="full")
    parser.add_argument("--output-dir", default="outputs/appe_aaai")
    args = parser.parse_args()
    config = load_config(args.config, seed=args.seed, mode=args.mode)
    rows = run_aaai_baselines(config, output_dir=args.output_dir)
    print(rows)


if __name__ == "__main__":
    main()

