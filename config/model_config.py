"""
Project-level model configuration for S-MAPE / APPE.

Edit this file to choose providers and model roles. Environment variables are
no longer the primary integration path; they are only used as a fallback when a
field such as api_key is left blank here.

For private keys, prefer creating `config/model_config_private.py` with:

MODEL_CONFIG_OVERRIDES = {
    "providers": {
        "volc": {"api_key": "your-key-here"}
    }
}
"""

MODEL_CONFIG = {
    "default_provider": "volc",
    "roles": {
        "target": {
            "provider": "volc",
            "model": "doubao-seed-2.0-pro",
            "temperature": 0.0,
        },
        "red": {
            "provider": "volc",
            "model": "deepseek-v3.2",
            "temperature": 0.9,
        },
        "patch": {
            "provider": "volc",
            "model": "deepseek-v3.2",
            "temperature": 0.7,
        },
        "eval": {
            "provider": "volc",
            "model": "doubao-seed-2.0-pro",
            "temperature": 0.0,
        },
    },
    "providers": {
        "volc": {
            "api_key": "",
            "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
        },
        "openai": {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
        },
        "local": {
            "api_key": "EMPTY",
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "local-model",
        },
        "ollama": {
            "api_key": None,
            "base_url": "http://127.0.0.1:11434",
            "model": "qwen2.5:7b-instruct",
        },
    },
}
