from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentFlags:
    enable_red_team: bool = True
    enable_blue_diagnosis: bool = True
    enable_textual_loss: bool = True
    enable_meta_controller: bool = True
    enable_regression_test: bool = True
    enable_memory_updates: bool = True
    enable_pruning: bool = True
    enable_abstraction: bool = True
    evaluate_clean_set: bool = True
    evaluate_adv_set: bool = True


EXPERIMENT_PRESETS: dict[str, ExperimentFlags] = {
    "full": ExperimentFlags(),
    "baseline_static": ExperimentFlags(
        enable_red_team=False,
        enable_blue_diagnosis=False,
        enable_textual_loss=False,
        enable_meta_controller=False,
        enable_regression_test=False,
        enable_memory_updates=False,
        enable_pruning=False,
        enable_abstraction=False,
    ),
    "red_only": ExperimentFlags(
        enable_red_team=True,
        enable_blue_diagnosis=False,
        enable_textual_loss=False,
        enable_meta_controller=False,
        enable_regression_test=False,
        enable_memory_updates=False,
        enable_pruning=False,
        enable_abstraction=False,
    ),
    "no_red_team": ExperimentFlags(
        enable_red_team=False,
    ),
    "patch_only": ExperimentFlags(
        enable_red_team=True,
        enable_blue_diagnosis=True,
        enable_textual_loss=True,
        enable_meta_controller=False,
        enable_regression_test=False,
        enable_memory_updates=True,
        enable_pruning=False,
        enable_abstraction=False,
    ),
    "no_textual_loss": ExperimentFlags(
        enable_red_team=True,
        enable_blue_diagnosis=True,
        enable_textual_loss=False,
    ),
    "no_meta_controller": ExperimentFlags(
        enable_red_team=True,
        enable_blue_diagnosis=True,
        enable_meta_controller=False,
    ),
    "no_regression": ExperimentFlags(
        enable_red_team=True,
        enable_blue_diagnosis=True,
        enable_regression_test=False,
    ),
    "no_clean_guard": ExperimentFlags(
        enable_red_team=True,
        enable_blue_diagnosis=True,
        enable_regression_test=False,
    ),
    "no_memory": ExperimentFlags(
        enable_red_team=True,
        enable_blue_diagnosis=True,
        enable_memory_updates=False,
        enable_pruning=False,
        enable_abstraction=False,
    ),
    "no_patch_memory": ExperimentFlags(
        enable_red_team=True,
        enable_blue_diagnosis=True,
        enable_memory_updates=False,
        enable_pruning=False,
        enable_abstraction=False,
    ),
    "no_pruning": ExperimentFlags(
        enable_red_team=True,
        enable_blue_diagnosis=True,
        enable_pruning=False,
        enable_abstraction=False,
    ),
    "no_abstraction": ExperimentFlags(
        enable_red_team=True,
        enable_blue_diagnosis=True,
        enable_abstraction=False,
    ),
}


def resolve_experiment_flags(name: str | None) -> ExperimentFlags:
    if not name:
        return EXPERIMENT_PRESETS["full"]
    return EXPERIMENT_PRESETS.get(name, EXPERIMENT_PRESETS["full"])


def experiment_flags_to_dict(flags: ExperimentFlags) -> dict[str, bool]:
    return asdict(flags)
