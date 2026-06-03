# core/_configs.py (Blue team configuration)
DIAGNOSTICIAN_REQUIRED_KEYS = [
    "failure_node",
    "vulnerable_segment",
    "repair_direction"
]

PATCHER_REQUIRED_KEYS = [
    "failure_type",
    "attack_type",
    "trigger_condition",
    "defense_rule",
    "scope",
    "rationale",
    "proposed_rule"
]

PRUNER_REQUIRED_KEYS = [
    "compressed_prompt"
]

META_REQUIRED_KEYS = [
    "decision",
    "reasoning",
    "selected_patch"
]
