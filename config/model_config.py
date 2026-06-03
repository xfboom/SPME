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
    "default_provider": "ollama",
    "roles": {
        "target": {
            "provider": "ollama",
            "model": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M:latest",
            "temperature": 0.0,
            "max_output_tokens": 64,
        },
        "red": {
            "provider": "ollama",
            "model": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M:latest",
            "temperature": 0.9,
            "max_output_tokens": 256,
        },
        "patch": {
            "provider": "ollama",
            "model": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M:latest",
            "temperature": 0.7,
            "max_output_tokens": 256,
        },
        "eval": {
            "provider": "ollama",
            "model": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M:latest",
            "temperature": 0.0,
            "max_output_tokens": 64,
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
            "model": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M:latest",
        },
    },
}
