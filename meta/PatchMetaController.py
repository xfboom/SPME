from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PatchDecision = Literal["accept", "reject", "merge", "prune"]


@dataclass
class PatchEvalResult:
    old_adv_correct: bool = False
    new_adv_correct: bool = False
    old_clean_acc: float | None = None
    new_clean_acc: float | None = None
    adv_gain: float = 0.0
    clean_drop: float = 0.0


class PatchMetaController:
    """
    Deterministic gate for patch-level evolution.

    The LLM blue team can propose and rank patches, but this controller is the
    final guard that decides whether a patch is worth entering memory.
    """

    def __init__(
        self,
        *,
        adv_gain_threshold: float = 0.01,
        clean_drop_threshold: float = 0.03,
        max_conflict_score: float = 0.65,
        max_redundancy_score: float = 0.82,
        lambda_clean: float = 1.25,
        mu_token: float = 0.08,
        nu_redundancy: float = 0.35,
        rho_conflict: float = 0.55,
        alpha_specificity: float = 0.08,
        beta_generality: float = 0.05,
        gamma_interpretability: float = 0.05,
        accept_score_threshold: float = 0.0,
        token_budget_unit: int = 80,
    ):
        self.adv_gain_threshold = adv_gain_threshold
        self.clean_drop_threshold = clean_drop_threshold
        self.max_conflict_score = max_conflict_score
        self.max_redundancy_score = max_redundancy_score
        self.lambda_clean = lambda_clean
        self.mu_token = mu_token
        self.nu_redundancy = nu_redundancy
        self.rho_conflict = rho_conflict
        self.alpha_specificity = alpha_specificity
        self.beta_generality = beta_generality
        self.gamma_interpretability = gamma_interpretability
        self.accept_score_threshold = accept_score_threshold
        self.token_budget_unit = max(1, token_budget_unit)

    def score(self, patch, eval_result: PatchEvalResult) -> float:
        token_cost = getattr(patch, "token_cost", 0) or 0
        token_penalty = min(1.0, token_cost / self.token_budget_unit)

        return (
            eval_result.adv_gain
            - self.lambda_clean * max(0.0, eval_result.clean_drop)
            - self.mu_token * token_penalty
            - self.nu_redundancy * getattr(patch, "redundancy_score", 0.0)
            - self.rho_conflict * getattr(patch, "conflict_score", 0.0)
            + self.alpha_specificity * getattr(patch, "specificity_score", 0.0)
            + self.beta_generality * getattr(patch, "generality_score", 0.0)
            + self.gamma_interpretability * getattr(patch, "interpretability_score", 0.0)
        )

    def decide(self, patch, eval_result: PatchEvalResult) -> PatchDecision:
        if eval_result.clean_drop > self.clean_drop_threshold:
            return "reject"

        if getattr(patch, "conflict_score", 0.0) > self.max_conflict_score:
            return "reject"

        patch_score = self.score(patch, eval_result)

        if getattr(patch, "redundancy_score", 0.0) > self.max_redundancy_score:
            if eval_result.adv_gain >= self.adv_gain_threshold and patch_score >= self.accept_score_threshold:
                return "merge"
            return "reject"

        if eval_result.adv_gain >= self.adv_gain_threshold and patch_score >= self.accept_score_threshold:
            return "accept"

        return "reject"
