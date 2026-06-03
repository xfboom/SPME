from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal
import copy
import json
import os
import re
import uuid


PatchStatus = Literal["candidate", "accepted", "rejected", "merged", "pruned"]


@dataclass
class CognitivePatch:
    patch_id: str
    failure_type: str
    attack_type: str
    trigger_condition: str
    defense_rule: str
    scope: str = "local"
    source_examples: List[str] = field(default_factory=list)
    estimated_adv_gain: float = 0.0
    estimated_clean_risk: float = 0.0
    estimated_clean_drop: float = 0.0
    redundancy_score: float = 0.0
    conflict_score: float = 0.0
    token_cost: int = 0
    usage_count: int = 0
    last_effective_round: int | None = None
    status: PatchStatus = "candidate"
    utility_score: float = 0.0
    conflict_count: int = 0
    rationale: str = ""
    created_round: int | None = None

    @property
    def failure_node(self) -> str:
        return self.failure_type

    @failure_node.setter
    def failure_node(self, value: str) -> None:
        self.failure_type = value

    @property
    def repair_rule(self) -> str:
        return self.defense_rule

    @repair_rule.setter
    def repair_rule(self, value: str) -> None:
        self.defense_rule = value
        self.token_cost = estimate_token_cost(value)

    @property
    def is_active(self) -> bool:
        return self.status == "accepted"

    @is_active.setter
    def is_active(self, value: bool) -> None:
        self.status = "accepted" if value else "pruned"

    @property
    def specificity_score(self) -> float:
        trigger = self.trigger_condition.strip()
        return 0.8 if trigger and trigger.lower() not in {"any", "always", "unknown"} else 0.25

    @property
    def generality_score(self) -> float:
        rule = self.defense_rule.lower()
        broad_markers = ["always", "never", "all tasks", "ignore all", "must not use any"]
        if any(marker in rule for marker in broad_markers):
            return 0.2
        return 0.65 if self.scope in {"failure_type", "task_family", "global"} else 0.45

    @property
    def interpretability_score(self) -> float:
        rule = self.defense_rule.strip()
        if not rule:
            return 0.0
        if len(rule.split()) <= 45 and any(word in rule.lower() for word in ["identify", "check", "verify", "ignore", "separate", "map"]):
            return 0.85
        return 0.55


def estimate_token_cost(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text or "")))


def _word_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
        if len(token) > 2
    }


def _jaccard(a: str, b: str) -> float:
    left = _word_set(a)
    right = _word_set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class AdversarialMemory:
    """
    Patch memory and prompt builder.

    The base prompt is stable. Blue-team updates enter as structured patches,
    and the final prompt is assembled from accepted patches under a budget.
    """

    def __init__(
        self,
        metrics_file: str = "metrics.jsonl",
        *,
        base_instruction: str | None = None,
        max_active_patches: int | None = None,
        max_prompt_tokens: int | None = None,
    ):
        self.base_instruction = base_instruction.strip() if base_instruction else self._load_base_instruction()
        if "<answer>" not in self.base_instruction.lower():
            self.base_instruction = (
                f"{self.base_instruction}\n\n"
                "Wrap your final answer within <answer>...</answer> tags."
            )
        self.patches: List[CognitivePatch] = []
        self.episode_history: List[dict] = []
        self.meta_policy: str = ""
        self.red_team_scores: Dict[str, int] = {}
        self.metrics_file = metrics_file
        self.max_active_patches = max_active_patches or int(os.environ.get("MAX_ACTIVE_PATCHES", "12"))
        self.max_prompt_tokens = max_prompt_tokens or int(os.environ.get("MAX_PROMPT_TOKENS", "1500"))

    def _load_base_instruction(self) -> str:
        base_dir = Path(__file__).resolve().parents[1]
        prompt_path = base_dir / "meta" / "system_prompt.txt"
        if prompt_path.exists():
            prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        else:
            prompt_text = "You are a rigorous reasoning assistant. Think step by step and answer the question."

        tag_rule = "Wrap your final answer within <answer>...</answer> tags."
        if "<answer>" not in prompt_text.lower():
            prompt_text = f"{prompt_text}\n\n{tag_rule}"
        return prompt_text

    def get_current_system_prompt(self) -> str:
        active_patches = self._select_prompt_patches()
        prompt_sections = [self.base_instruction]

        if active_patches:
            lines = []
            for idx, patch in enumerate(active_patches, start=1):
                lines.append(
                    f"{idx}. failure_type={patch.failure_type}; "
                    f"trigger={patch.trigger_condition or 'observed adversarial trigger'}; "
                    f"rule={patch.defense_rule}; scope={patch.scope}"
                )
            prompt_sections.append(
                "[Patch Memory: Adversarial Robustness Guidelines]\n"
                "Use these patches only when their trigger is relevant. The original task instruction and answer schema remain primary.\n"
                + "\n".join(lines)
            )

        if self.meta_policy:
            prompt_sections.append("[Meta Policy]\n" + self.meta_policy)

        return "\n\n".join(prompt_sections)

    def _select_prompt_patches(self) -> List[CognitivePatch]:
        active = self.get_active_patches()
        active = sorted(
            active,
            key=lambda p: (
                p.utility_score + p.estimated_adv_gain,
                -p.estimated_clean_risk,
                -p.conflict_score,
                -p.redundancy_score,
                -(p.token_cost or estimate_token_cost(p.defense_rule)),
            ),
            reverse=True,
        )

        selected: List[CognitivePatch] = []
        budget_used = estimate_token_cost(self.base_instruction)
        for patch in active[: self.max_active_patches]:
            patch_cost = patch.token_cost or estimate_token_cost(patch.defense_rule)
            if budget_used + patch_cost > self.max_prompt_tokens:
                continue
            selected.append(patch)
            budget_used += patch_cost
        return selected

    @property
    def current_system_prompt(self) -> str:
        return self.get_current_system_prompt()

    def build_patch(
        self,
        *,
        failure_type: str,
        defense_rule: str,
        attack_type: str = "",
        trigger_condition: str = "",
        scope: str = "failure_type",
        source_examples: List[str] | None = None,
        rationale: str = "",
        created_round: int | None = None,
        estimated_clean_risk: float = 0.0,
    ) -> CognitivePatch:
        patch = CognitivePatch(
            patch_id=str(uuid.uuid4())[:8],
            failure_type=failure_type or "Unknown_Vulnerability",
            attack_type=attack_type or "Unknown_Attack",
            trigger_condition=trigger_condition or "similar adversarial trigger appears",
            defense_rule=defense_rule.strip(),
            scope=scope or "failure_type",
            source_examples=source_examples or [],
            estimated_clean_risk=estimated_clean_risk,
            token_cost=estimate_token_cost(defense_rule),
            rationale=rationale,
            created_round=created_round,
        )
        patch.redundancy_score = self.estimate_redundancy(patch)
        patch.conflict_score = self.estimate_conflict(patch)
        return patch

    def add_patch(self, failure_node: str, repair_rule: str, **kwargs) -> CognitivePatch:
        patch = self.build_patch(
            failure_type=failure_node,
            defense_rule=repair_rule,
            attack_type=kwargs.get("attack_type", ""),
            trigger_condition=kwargs.get("trigger_condition", ""),
            scope=kwargs.get("scope", "failure_type"),
            source_examples=kwargs.get("source_examples"),
            rationale=kwargs.get("rationale", ""),
            created_round=kwargs.get("created_round"),
            estimated_clean_risk=kwargs.get("estimated_clean_risk", 0.0),
        )
        return self.add_prompt_patch(patch)

    def add_prompt_patch(self, patch: CognitivePatch) -> CognitivePatch:
        patch.status = "accepted"
        patch.token_cost = patch.token_cost or estimate_token_cost(patch.defense_rule)
        self.patches.append(patch)
        return patch

    def merge_patch(self, patch: CognitivePatch) -> CognitivePatch:
        active = self.get_active_patches()
        best_match = None
        best_score = 0.0
        for existing in active:
            if existing.failure_type != patch.failure_type:
                continue
            score = _jaccard(existing.defense_rule, patch.defense_rule)
            if score > best_score:
                best_match = existing
                best_score = score

        patch.status = "merged"
        self.patches.append(patch)

        if best_match is not None and best_score >= 0.35:
            best_match.utility_score = max(best_match.utility_score, patch.utility_score)
            best_match.estimated_adv_gain = max(best_match.estimated_adv_gain, patch.estimated_adv_gain)
            best_match.estimated_clean_risk = min(
                max(best_match.estimated_clean_risk, patch.estimated_clean_risk),
                1.0,
            )
            best_match.source_examples.extend(patch.source_examples[:2])
            best_match.last_effective_round = patch.last_effective_round or best_match.last_effective_round
            return best_match

        patch.status = "accepted"
        return patch

    def apply_decision(self, patch: CognitivePatch, decision: str) -> CognitivePatch | None:
        if decision == "accept":
            return self.add_prompt_patch(patch)
        if decision == "merge":
            return self.merge_patch(patch)
        if decision == "reject":
            patch.status = "rejected"
            self.patches.append(patch)
            return patch
        if decision == "prune":
            patch.status = "pruned"
            self.patches.append(patch)
            return patch
        return None

    def get_active_patches(self) -> List[CognitivePatch]:
        return [patch for patch in self.patches if patch.status == "accepted"]

    def estimate_redundancy(self, patch: CognitivePatch) -> float:
        scores = [_jaccard(patch.defense_rule, active.defense_rule) for active in self.get_active_patches()]
        return max(scores, default=0.0)

    def estimate_conflict(self, patch: CognitivePatch) -> float:
        rule = patch.defense_rule.lower()
        conflict_score = 0.0
        broad_or_risky = [
            "ignore all",
            "never use",
            "always ignore",
            "do not answer",
            "refuse",
            "change the answer",
            "skip calculation",
        ]
        if any(marker in rule for marker in broad_or_risky):
            conflict_score += 0.45

        for active in self.get_active_patches():
            active_rule = active.defense_rule.lower()
            if "follow the requested format" in active_rule and "ignore format" in rule:
                conflict_score += 0.25
            if "ignore" in active_rule and "use" in rule and _jaccard(active_rule, rule) > 0.4:
                conflict_score += 0.2

        return min(1.0, conflict_score)

    def prune_patches(self, min_utility: float = 0.0, max_conflicts: int = 3) -> int:
        pruned_count = 0
        for patch in self.get_active_patches():
            stale = patch.usage_count == 0 and patch.created_round is not None
            risky = patch.conflict_count >= max_conflicts or patch.conflict_score >= 0.8
            low_value = patch.utility_score <= min_utility and patch.estimated_adv_gain <= 0.0
            if risky or (stale and low_value):
                patch.status = "pruned"
                pruned_count += 1

        active = self.get_active_patches()
        if len(active) > self.max_active_patches:
            ranked = sorted(
                active,
                key=lambda p: (
                    p.utility_score + p.estimated_adv_gain,
                    -p.estimated_clean_risk,
                    -p.conflict_score,
                    -p.redundancy_score,
                    -p.token_cost,
                ),
                reverse=True,
            )
            keep = {patch.patch_id for patch in ranked[: self.max_active_patches]}
            for patch in active:
                if patch.patch_id not in keep:
                    patch.status = "pruned"
                    pruned_count += 1

        return pruned_count

    def reward_red_operator(self, red_operator, score: int):
        operator_name = red_operator.__class__.__name__
        if operator_name not in self.red_team_scores:
            self.red_team_scores[operator_name] = 0
        self.red_team_scores[operator_name] += score

    def record_episode(
        self,
        *,
        epoch: int,
        failure_node: str,
        vulnerable_segment: str,
        repair_direction: str,
        textual_loss: str,
        decision: str,
        is_correct: bool,
        regression_pass: bool | None,
        selected_patch: str = "",
        red_operator_name: str = "",
        patch_id: str = "",
        adv_gain: float | None = None,
        clean_drop: float | None = None,
    ) -> dict:
        episode = {
            "epoch": epoch,
            "failure_node": failure_node,
            "vulnerable_segment": vulnerable_segment,
            "repair_direction": repair_direction,
            "textual_loss": textual_loss,
            "decision": decision,
            "is_correct": is_correct,
            "regression_pass": regression_pass,
            "selected_patch": selected_patch,
            "red_operator_name": red_operator_name,
            "patch_id": patch_id,
            "adv_gain": adv_gain,
            "clean_drop": clean_drop,
        }
        self.episode_history.append(episode)
        return episode

    def update_meta_policy(self, episode: dict) -> str:
        failure_node = episode.get("failure_node", "Unknown_Vulnerability")
        repair_direction = episode.get("repair_direction", "")
        decision = episode.get("decision", "reject")
        regression_pass = episode.get("regression_pass")
        selected_patch = episode.get("selected_patch", "").strip()

        if regression_pass is False:
            policy = (
                f"Prioritize localized repairs for {failure_node}. "
                "Avoid broad constraints that reduce clean reasoning. "
                f"Prefer patches aligned with: {repair_direction}."
            )
        elif decision in {"accept", "merge"} and selected_patch:
            policy = (
                f"Promote compact, typed patches for {failure_node}. "
                "Keep adversarial gains only when clean accuracy and prompt budget remain stable."
            )
        elif episode.get("is_correct"):
            policy = (
                "Maintain the current defense set. "
                "Reward rules that remain effective under the present adversarial distribution."
            )
        else:
            policy = (
                f"Request tighter causal linkage for {failure_node}. "
                "Reject candidate patches that restate the problem instead of constraining the failure mode."
            )

        self.meta_policy = policy
        return policy

    def update_system_prompt(self, new_prompt: str):
        self.base_instruction = new_prompt
        self.patches.clear()
        self.episode_history.clear()
        self.meta_policy = ""

    def log_epoch(
        self,
        epoch: int,
        val_acc: float,
        train_acc: float,
        clean_acc: float | None = None,
        adv_acc: float | None = None,
        blue_trigger_count: int | None = None,
        blue_merge_count: int | None = None,
        blue_reject_count: int | None = None,
        blue_error_count: int | None = None,
    ):
        current_prompt = self.get_current_system_prompt()
        record = {
            "epoch": epoch,
            "prompt_length_tokens": estimate_token_cost(current_prompt),
            "active_patch_count": len(self.get_active_patches()),
            "val_acc": val_acc,
            "train_acc": train_acc,
            "clean_acc": clean_acc,
            "adv_acc": adv_acc,
            "blue_trigger_count": blue_trigger_count,
            "blue_merge_count": blue_merge_count,
            "blue_reject_count": blue_reject_count,
            "blue_error_count": blue_error_count,
            "current_prompt": current_prompt,
            "active_patches": [asdict(patch) for patch in self.get_active_patches()],
        }
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def copy(self) -> "AdversarialMemory":
        return copy.deepcopy(self)

    def save_patches_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for patch in self.patches:
                f.write(json.dumps(asdict(patch), ensure_ascii=False) + "\n")

    def load_patches_jsonl(self, path: str) -> None:
        self.patches.clear()
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                self.patches.append(CognitivePatch(**data))
