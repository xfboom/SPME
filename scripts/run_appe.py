from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from blue_team.BlueCommittee import BlueCommittee
from core.PromptManager import PromptManager
from memory.AdversarialMemory import AdversarialMemory
from meta.PatchMetaController import PatchEvalResult, PatchMetaController
from scripts.aaai_utils import (
    accuracy,
    append_jsonl,
    build_adv_samples,
    canonical_attack_type,
    count_tokens,
    evaluate_prompt,
    load_config,
    load_json,
    load_prompt,
    predict_samples,
    root_path,
    rounded_row,
    sample_records,
    save_json,
    save_jsonl,
)


def _controller_from_config(config: dict) -> PatchMetaController:
    cfg = config.get("controller", {})
    return PatchMetaController(
        adv_gain_threshold=float(cfg.get("adv_gain_threshold", 0.02)),
        clean_drop_threshold=float(cfg.get("clean_drop_threshold", 0.01)),
        lambda_clean=float(cfg.get("lambda_clean", 1.0)),
        mu_token=float(cfg.get("mu_token", 0.001)),
        nu_redundancy=float(cfg.get("nu_redundancy", 0.5)),
        rho_conflict=float(cfg.get("rho_conflict", 0.5)),
    )


def _load_eval_sets(config: dict) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    split_dir = root_path(config.get("data", {}).get("split_dir", "data/appe_splits"))
    evo_clean = load_json(split_dir / "evo_clean.json")
    guard = load_json(split_dir / "guard.json")
    test_clean = load_json(split_dir / "test_clean.json")
    test_seen_adv = load_json(split_dir / "test_seen_adv.json")
    test_unseen_adv = load_json(split_dir / "test_unseen_adv.json")
    return evo_clean, guard, test_clean, test_seen_adv, test_unseen_adv


def evaluate_candidate_patch(
    *,
    config: dict,
    memory: AdversarialMemory,
    patch,
    clean_guard_batch: list[dict],
    adv_validation_batch: list[dict],
) -> tuple[dict, PatchEvalResult]:
    old_prompt = memory.current_system_prompt
    temp_memory = memory.copy()
    temp_memory.add_prompt_patch(patch)
    new_prompt = temp_memory.current_system_prompt

    old_clean_preds = predict_samples(old_prompt, clean_guard_batch, "APPE-old", config)
    new_clean_preds = predict_samples(new_prompt, clean_guard_batch, "APPE-new", config)
    old_adv_preds = predict_samples(old_prompt, adv_validation_batch, "APPE-old", config)
    new_adv_preds = predict_samples(new_prompt, adv_validation_batch, "APPE-new", config)

    old_clean_acc = accuracy(old_clean_preds)
    new_clean_acc = accuracy(new_clean_preds)
    old_adv_acc = accuracy(old_adv_preds)
    new_adv_acc = accuracy(new_adv_preds)
    clean_drop = max(0.0, old_clean_acc - new_clean_acc)
    adv_gain = max(0.0, new_adv_acc - old_adv_acc)

    patch.estimated_adv_gain = adv_gain
    patch.estimated_clean_drop = clean_drop
    patch.estimated_clean_risk = clean_drop
    if adv_gain > 0:
        patch.utility_score += adv_gain
        patch.usage_count += 1

    metrics = {
        "old_clean_acc": old_clean_acc,
        "new_clean_acc": new_clean_acc,
        "old_adv_acc": old_adv_acc,
        "new_adv_acc": new_adv_acc,
        "clean_drop": clean_drop,
        "adv_gain": adv_gain,
        "token_cost": patch.token_cost,
        "redundancy_score": patch.redundancy_score,
        "conflict_score": patch.conflict_score,
    }
    eval_result = PatchEvalResult(
        old_adv_correct=old_adv_acc > 0,
        new_adv_correct=new_adv_acc > old_adv_acc,
        old_clean_acc=old_clean_acc,
        new_clean_acc=new_clean_acc,
        adv_gain=adv_gain,
        clean_drop=clean_drop,
    )
    return metrics, eval_result


def _round_eval_row(
    *,
    round_idx: int,
    prompt: str,
    memory: AdversarialMemory,
    config: dict,
    test_clean: list[dict],
    test_seen_adv: list[dict],
    candidate_patch_count: int,
    accepted_patch_count: int,
    rejected_patch_count: int,
    merged_patch_count: int,
    pruned_patch_count: int,
) -> dict:
    metrics, _ = evaluate_prompt(
        prompt=prompt,
        method="APPE",
        config=config,
        clean_samples=test_clean,
        seen_adv_samples=test_seen_adv,
        unseen_adv_samples=[],
    )
    return rounded_row(
        {
            "round": round_idx,
            "clean_acc": metrics["clean_acc"],
            "seen_adv_acc": metrics["seen_adv_acc"],
            "asr": metrics["asr"],
            "tokens": count_tokens(prompt),
            "active_patch_count": len(memory.get_active_patches()),
            "candidate_patch_count": candidate_patch_count,
            "accepted_patch_count": accepted_patch_count,
            "rejected_patch_count": rejected_patch_count,
            "merged_patch_count": merged_patch_count,
            "pruned_patch_count": pruned_patch_count,
        }
    )


def run_appe(config: dict, *, variant: str = "full", output_dir: str | Path | None = None) -> dict:
    seed = int(config.get("seed", 42))
    out_dir = root_path(output_dir or config.get("outputs", {}).get("dir", "outputs/appe_aaai"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds_dir = out_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)

    evo_clean, guard, test_clean, test_seen_adv, test_unseen_adv = _load_eval_sets(config)
    prompt_cfg = config.get("prompt", {})
    base_prompt = load_prompt(prompt_cfg.get("base_prompt_path", "config/prompts/base_math.txt"))
    memory = AdversarialMemory(
        metrics_file=str(out_dir / "appe_metrics.jsonl"),
        base_instruction=base_prompt,
        max_active_patches=int(config.get("memory", {}).get("max_active_patches", 12)),
        max_prompt_tokens=int(prompt_cfg.get("max_tokens", 1500)),
    )
    controller = _controller_from_config(config)
    blue = BlueCommittee(None, PromptManager("_prompts"))

    attack_cfg = config.get("attacks", {})
    seen_attacks = [canonical_attack_type(name) for name in attack_cfg.get("seen", [])]
    appe_cfg = config.get("appe", {})
    rounds = int(appe_cfg.get("rounds", 5))
    clean_batch_size = int(appe_cfg.get("clean_batch_size", 32))
    attacks_per_sample = int(appe_cfg.get("attacks_per_sample", 3))
    clean_guard_size = int(appe_cfg.get("clean_guard_size", 64))
    adv_validation_size = int(appe_cfg.get("adv_validation_size", 64))
    max_candidates = int(appe_cfg.get("candidate_patches_per_round", 6))

    use_failure_ontology = variant != "no_ontology"
    use_meta_controller = variant != "no_meta_controller"
    single_agent = variant == "single_blue_agent"

    print(
        f"[APPE] variant={variant}, rounds={rounds}, "
        f"clean_batch_size={clean_batch_size}, model_backend={config.get('model', {}).get('backend')}",
        flush=True,
    )

    base_metrics, _ = evaluate_prompt(
        prompt=base_prompt,
        method="Base Prompt",
        config=config,
        clean_samples=test_clean,
        seen_adv_samples=test_seen_adv,
        unseen_adv_samples=test_unseen_adv,
    )
    evolution_rows = [
        _round_eval_row(
            round_idx=0,
            prompt=memory.current_system_prompt,
            memory=memory,
            config=config,
            test_clean=test_clean,
            test_seen_adv=test_seen_adv,
            candidate_patch_count=0,
            accepted_patch_count=0,
            rejected_patch_count=0,
            merged_patch_count=0,
            pruned_patch_count=0,
        )
    ]
    round0_dir = rounds_dir / "round_0"
    round0_dir.mkdir(parents=True, exist_ok=True)
    (round0_dir / "prompt.txt").write_text(memory.current_system_prompt, encoding="utf-8")
    save_json(round0_dir / "round_metrics.json", evolution_rows[0])

    for round_idx in range(1, rounds + 1):
        print(f"[APPE] Round {round_idx}/{rounds}: start", flush=True)
        round_dir = rounds_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        current_prompt = memory.current_system_prompt
        (round_dir / "prompt.txt").write_text(current_prompt, encoding="utf-8")

        clean_batch = sample_records(evo_clean, clean_batch_size, seed + round_idx)
        round_attacks = seen_attacks[: max(1, min(attacks_per_sample, len(seen_attacks)))]
        adv_batch = build_adv_samples(
            clean_batch,
            round_attacks,
            seed=seed + round_idx,
            is_seen_attack=True,
            prefix=f"round{round_idx}_seen_adv",
        )
        save_jsonl(round_dir / "adv_batch.jsonl", adv_batch)
        print(f"[APPE] Round {round_idx}/{rounds}: generated adv_batch={len(adv_batch)}", flush=True)

        adv_preds = predict_samples(
            current_prompt,
            adv_batch,
            "APPE",
            config,
            progress_label=f"APPE round {round_idx}/adv_batch",
        )
        pred_by_id = {row["sample_id"]: row for row in adv_preds}
        failures = []
        for sample in adv_batch:
            pred = pred_by_id.get(sample["sample_id"], {})
            if not pred.get("is_correct"):
                failures.append({**sample, "prediction": pred})
        save_jsonl(round_dir / "failures.jsonl", failures)
        print(f"[APPE] Round {round_idx}/{rounds}: failures={len(failures)}", flush=True)

        current_rules = [patch.repair_rule for patch in memory.get_active_patches()]
        diagnoses = blue.diagnose_failures_batch(failures, use_failure_ontology=use_failure_ontology)
        candidate_patch_jsons = blue.generate_candidate_patches_batch(
            diagnoses,
            current_rules,
            max_candidates=max_candidates,
            single_agent=single_agent,
            use_failure_ontology=use_failure_ontology,
        )
        save_jsonl(round_dir / "candidate_patches.jsonl", candidate_patch_jsons)
        print(
            f"[APPE] Round {round_idx}/{rounds}: candidate_patches={len(candidate_patch_jsons)}",
            flush=True,
        )

        clean_guard_batch = sample_records(guard, clean_guard_size, seed + 1000 + round_idx)
        adv_validation_batch = sample_records(
            failures or adv_batch,
            adv_validation_size,
            seed + 2000 + round_idx,
        )

        accepted = rejected = merged = 0
        for patch_idx, patch_json in enumerate(candidate_patch_jsons, start=1):
            print(
                f"[APPE] Round {round_idx}/{rounds}: evaluating patch "
                f"{patch_idx}/{len(candidate_patch_jsons)} "
                f"({patch_json.get('failure_type', 'Unknown')})",
                flush=True,
            )
            patch = memory.build_patch(
                failure_type=patch_json.get("failure_type", "Unknown_Vulnerability"),
                attack_type=patch_json.get("attack_type", ""),
                trigger_condition=patch_json.get("trigger_condition", ""),
                defense_rule=patch_json.get("defense_rule", ""),
                scope=patch_json.get("scope", "failure_type"),
                source_examples=[
                    row.get("question", "")[:800]
                    for row in failures
                    if row.get("attack_type") == patch_json.get("attack_type")
                ][:3],
                rationale=patch_json.get("rationale", ""),
                created_round=round_idx,
            )
            eval_metrics, eval_result = evaluate_candidate_patch(
                config=config,
                memory=memory,
                patch=patch,
                clean_guard_batch=clean_guard_batch,
                adv_validation_batch=adv_validation_batch,
            )
            if use_meta_controller:
                decision = controller.decide(patch, eval_result)
                score = controller.score(patch, eval_result)
            else:
                decision = "accept"
                score = eval_metrics["adv_gain"] - eval_metrics["clean_drop"]

            memory.apply_decision(patch, decision)
            print(
                f"[APPE] Round {round_idx}/{rounds}: patch {patch_idx}/{len(candidate_patch_jsons)} "
                f"decision={decision}, adv_gain={eval_metrics['adv_gain']:.4f}, "
                f"clean_drop={eval_metrics['clean_drop']:.4f}",
                flush=True,
            )
            if decision == "accept":
                accepted += 1
            elif decision == "merge":
                merged += 1
            else:
                rejected += 1

            append_jsonl(
                round_dir / "patch_decisions.jsonl",
                rounded_row(
                    {
                        "round": round_idx,
                        "patch_id": patch.patch_id,
                        "failure_type": patch.failure_type,
                        "attack_type": patch.attack_type,
                        "defense_rule": patch.defense_rule,
                        "adv_gain": eval_metrics["adv_gain"],
                        "clean_drop": eval_metrics["clean_drop"],
                        "token_cost": patch.token_cost,
                        "redundancy_score": patch.redundancy_score,
                        "conflict_score": patch.conflict_score,
                        "score": score,
                        "decision": decision,
                    }
                ),
            )

        pruned = 0
        if use_meta_controller:
            pruned = memory.prune_patches()

        row = _round_eval_row(
            round_idx=round_idx,
            prompt=memory.current_system_prompt,
            memory=memory,
            config=config,
            test_clean=test_clean,
            test_seen_adv=test_seen_adv,
            candidate_patch_count=len(candidate_patch_jsons),
            accepted_patch_count=accepted,
            rejected_patch_count=rejected,
            merged_patch_count=merged,
            pruned_patch_count=pruned,
        )
        evolution_rows.append(row)
        save_json(round_dir / "round_metrics.json", row)
        print(
            f"[APPE] Round {round_idx}/{rounds}: done "
            f"clean_acc={row['clean_acc']}, seen_adv_acc={row['seen_adv_acc']}, "
            f"active_patches={row['active_patch_count']}",
            flush=True,
        )

    final_prompt = memory.current_system_prompt
    final_metrics, predictions = evaluate_prompt(
        prompt=final_prompt,
        method="APPE",
        config=config,
        clean_samples=test_clean,
        seen_adv_samples=test_seen_adv,
        unseen_adv_samples=test_unseen_adv,
        base_clean_acc=base_metrics["clean_acc"],
    )
    final_metrics["active_patch_count"] = len(memory.get_active_patches())
    final_metrics["tokens"] = count_tokens(final_prompt)
    final_metrics["variant"] = variant
    final_metrics = rounded_row(final_metrics)

    (out_dir / "final_prompt.txt").write_text(final_prompt, encoding="utf-8")
    memory.save_patches_jsonl(str(out_dir / "patch_memory.jsonl"))
    save_jsonl(out_dir / "predictions_clean.jsonl", predictions["clean"])
    save_jsonl(out_dir / "predictions_seen_adv.jsonl", predictions["seen_adv"])
    save_jsonl(out_dir / "predictions_unseen_adv.jsonl", predictions["unseen_adv"])
    save_json(out_dir / "appe_final_metrics.json", final_metrics)

    import csv

    curve_path = out_dir / "evolution_curve.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as f:
        columns = [
            "round",
            "clean_acc",
            "seen_adv_acc",
            "asr",
            "tokens",
            "active_patch_count",
            "candidate_patch_count",
            "accepted_patch_count",
            "rejected_patch_count",
            "merged_patch_count",
            "pruned_patch_count",
        ]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(evolution_rows)

    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/appe_aaai.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", choices=["full", "smoke", "real-smoke"], default="full")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config, seed=args.seed, mode=args.mode)
    metrics = run_appe(config, variant=args.variant, output_dir=args.output_dir)
    print(metrics)


if __name__ == "__main__":
    main()

