from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    """Tiny .env loader so local experiments do not require python-dotenv."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if " #" in value:
            value = value.split(" #", 1)[0].strip()
        if value and value[0] in {"'", "\""} and value[-1:] == value[0]:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_env_file(BASE_DIR / ".env")

# Dataset paths
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    str(BASE_DIR / "Dataset_format" / "MMLU" / "professional_law_test.csv"),
)
DATASET_QUESTION_COL = os.environ.get("DATASET_QUESTION_COL", "question")
DATASET_ANSWER_COL = os.environ.get("DATASET_ANSWER_COL", "answer")
DATA_DIR = BASE_DIR / "data"

# Split datasets for strict isolation
TRAIN_SET_PATH = str(DATA_DIR / "train.json")
VAL_SET_PATH = str(DATA_DIR / "val.json")
TEST_SET_PATH = str(DATA_DIR / "test.json")
CLEAN_TEST_SET_PATH = os.environ.get("CLEAN_TEST_SET_PATH", str(DATA_DIR / "test_clean.json"))
ADV_TEST_SET_PATH = os.environ.get("ADV_TEST_SET_PATH", str(DATA_DIR / "test_adv.json"))

# Evaluation controls
EVAL_MAX_SAMPLES = int(os.environ.get("EVAL_MAX_SAMPLES", "300"))
EVAL_RANDOM_SEED = int(os.environ.get("EVAL_RANDOM_SEED", "42"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(BASE_DIR / "out"))

# Benchmark generation
BENCHMARK_DATASET = os.environ.get("BENCHMARK_DATASET", "gsm8k")
BENCHMARK_CONFIG = os.environ.get("BENCHMARK_CONFIG", "main")
BENCHMARK_SIZE = int(os.environ.get("BENCHMARK_SIZE", "50"))
BENCHMARK_SEED = int(os.environ.get("BENCHMARK_SEED", "42"))
AUTO_GENERATE_BENCHMARK = os.environ.get("AUTO_GENERATE_BENCHMARK", "false").lower() == "true"
BENCHMARK_SOURCE = os.environ.get("BENCHMARK_SOURCE", "hf")
MULTITASK_SOURCE_ROOTS = [
    root.strip()
    for root in os.environ.get(
        "MULTITASK_SOURCE_ROOTS",
        str(BASE_DIR / "Dataset_format"),
    ).split(os.pathsep)
    if root.strip()
]
MULTITASK_MAX_PER_TASK = int(os.environ.get("MULTITASK_MAX_PER_TASK", "200"))
MULTITASK_TRAIN_RATIO = float(os.environ.get("MULTITASK_TRAIN_RATIO", "0.8"))
MULTITASK_VAL_RATIO = float(os.environ.get("MULTITASK_VAL_RATIO", "0.1"))
MULTITASK_TEST_RATIO = float(os.environ.get("MULTITASK_TEST_RATIO", "0.1"))
EXPERIMENT_PRESET = os.environ.get("EXPERIMENT_PRESET", "full")

# Feature flags (reserved for future use)
ENABLE_SMA = os.environ.get("ENABLE_SMA", "true").lower() == "true"
ENABLE_INPUT_VALIDATION = os.environ.get("ENABLE_INPUT_VALIDATION", "false").lower() == "true"
ENABLE_STRATEGY_SWITCHING = os.environ.get("ENABLE_STRATEGY_SWITCHING", "true").lower() == "true"
ENABLE_ROI_CHECK = os.environ.get("ENABLE_ROI_CHECK", "true").lower() == "true"
ENABLE_LOOP_DETECTION = os.environ.get("ENABLE_LOOP_DETECTION", "true").lower() == "true"

PRUNING_MODE = os.environ.get("PRUNING_MODE", "None")


@dataclass(frozen=True)
class LLMApiConfig:
    provider: str
    api_key: str | None
    base_url: str
    model: str


@dataclass(frozen=True)
class LLMRoleConfig:
    provider: str | None
    model: str | None
    temperature: float | None


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_project_model_config() -> dict[str, Any]:
    try:
        from config.model_config import MODEL_CONFIG
    except ModuleNotFoundError:
        MODEL_CONFIG = {}

    config = dict(MODEL_CONFIG)
    try:
        from config.model_config_private import MODEL_CONFIG_OVERRIDES
    except ModuleNotFoundError:
        MODEL_CONFIG_OVERRIDES = {}

    if MODEL_CONFIG_OVERRIDES:
        config = _deep_merge(config, MODEL_CONFIG_OVERRIDES)
    return config


def _provider_env_fallback(provider: str) -> dict[str, str | None]:
    if provider == "volc":
        return {
            "api_key": os.environ.get("VOLC_API_KEY"),
            "base_url": os.environ.get("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            "model": os.environ.get("VOLC_MODEL"),
        }
    if provider in {"local", "openai_compatible", "vllm", "lmstudio"}:
        return {
            "api_key": os.environ.get("LOCAL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
            "base_url": os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"),
            "model": os.environ.get("LOCAL_MODEL", os.environ.get("OPENAI_MODEL", "local-model")),
        }
    if provider == "ollama":
        return {
            "api_key": None,
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            "model": os.environ.get("OLLAMA_MODEL", os.environ.get("LOCAL_MODEL", "llama3.1:8b")),
        }
    return {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.com.cn/v1"),
        "model": os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
    }


def get_llm_role_config(role: str = "target") -> LLMRoleConfig:
    config = _load_project_model_config()
    role_cfg = config.get("roles", {}).get(role, {})
    return LLMRoleConfig(
        provider=role_cfg.get("provider"),
        model=role_cfg.get("model"),
        temperature=role_cfg.get("temperature"),
    )


def get_llm_api_config(
    provider_override: str | None = None,
    *,
    role: str = "target",
    model_override: str | None = None,
) -> LLMApiConfig:
    project_config = _load_project_model_config()
    role_cfg = project_config.get("roles", {}).get(role, {})
    provider = (
        provider_override
        or role_cfg.get("provider")
        or project_config.get("default_provider")
        or os.environ.get("LLM_PROVIDER", "openai")
    ).lower()

    provider_cfg = project_config.get("providers", {}).get(provider, {})
    env_fallback = _provider_env_fallback(provider)

    api_key = provider_cfg.get("api_key")
    if api_key == "":
        api_key = None
    api_key = api_key if api_key is not None else env_fallback.get("api_key")

    base_url = provider_cfg.get("base_url") or env_fallback.get("base_url") or ""
    model = (
        model_override
        or role_cfg.get("model")
        or provider_cfg.get("model")
        or env_fallback.get("model")
        or "gpt-3.5-turbo"
    )

    return LLMApiConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )


def get_model_name(role: str = "target") -> str:
    return get_llm_api_config(role=role).model
