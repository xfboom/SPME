# Running S-MAPE with a Local Model

## Current Project Stage

S-MAPE is currently a research prototype / MVP, not yet a paper-grade full benchmark suite.

Implemented:

- Red-team adversarial mutation.
- Blue-team failure diagnosis.
- Structured prompt patch generation.
- Patch memory instead of full prompt rewrite.
- Adversarial replay for candidate patches.
- Clean-accuracy guard before patch admission.
- Deterministic meta-controller for accept / merge / reject.
- Patch pruning and active-patch prompt assembly.

Still needed for a publication-grade experiment:

- Batch-level evolution instead of one seed example per round.
- Larger fixed clean/adversarial train, validation, and test splits.
- Strong semantic verifier for red-team attacks.
- Full baseline matrix: Base, CoT, Robust Prompt, Full Rewrite, Reflexion Memory, ProTeGi/OPRO-style baselines.
- Repeated runs across seeds and models.
- Transfer evaluation on unseen attack types and unseen datasets.

## Model Configuration

Model selection is now project-config driven. Edit:

```text
config/model_config.py
```

The important fields are:

```python
MODEL_CONFIG = {
    "default_provider": "volc",
    "roles": {
        "target": {"provider": "volc", "model": "doubao-seed-2.0-pro", "temperature": 0.0},
        "red": {"provider": "volc", "model": "deepseek-v3.2", "temperature": 0.9},
        "patch": {"provider": "volc", "model": "deepseek-v3.2", "temperature": 0.7},
        "eval": {"provider": "volc", "model": "doubao-seed-2.0-pro", "temperature": 0.0},
    },
    "providers": {
        "volc": {"api_key": "", "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3"}
    }
}
```

For private keys, create:

```text
config/model_config_private.py
```

using `config/model_config_private.py.example` as the template. This file is ignored by git. `.env` is now only a fallback when `api_key` is left blank.

## Recommended Local Model Setup

Use an instruction-tuned model with reliable JSON output. Good practical choices:

- Qwen2.5 / Qwen3 Instruct, 7B or larger.
- DeepSeek-R1-Distill-Qwen, if you want stronger reasoning but can tolerate slower runs.
- Llama 3.1 / 3.2 Instruct, 8B or larger.
- For serious experiments, use a stronger model for the blue team and a separate target model if possible.

For a first local smoke test, use one model for all roles. For paper experiments, prefer:

```text
target model: the model being defended
red model: same or stronger local model
blue model: strongest available local instruction model
```

The current code uses the role entries in `config/model_config.py`; `target`, `red`, `patch`, and `eval` can point to the same model or to separate models.

## Option A: OpenAI-Compatible Local Server

This works with vLLM, LM Studio, llama.cpp server, and other servers that expose `/v1/chat/completions`.

Example `.env`:

```env
LLM_PROVIDER="local"
LOCAL_BASE_URL="http://127.0.0.1:8000/v1"
LOCAL_API_KEY="EMPTY"
LOCAL_MODEL="Qwen2.5-7B-Instruct"

TARGET_MODEL="Qwen2.5-7B-Instruct"
RED_MODEL="Qwen2.5-7B-Instruct"

EVOLUTION_ROUNDS="5"
CANDIDATE_PATCHES_PER_ROUND="4"
MAX_ACTIVE_PATCHES="12"
MAX_PROMPT_TOKENS="1500"
EVAL_MAX_SAMPLES="100"
EXPERIMENT_PRESET="full"
```

If your server base URL already ends with `/chat/completions`, that is also accepted.

## Option B: Ollama Native API

Example `.env`:

```env
LLM_PROVIDER="ollama"
OLLAMA_BASE_URL="http://127.0.0.1:11434"
OLLAMA_MODEL="qwen2.5:7b-instruct"

TARGET_MODEL="qwen2.5:7b-instruct"
RED_MODEL="qwen2.5:7b-instruct"

EVOLUTION_ROUNDS="5"
CANDIDATE_PATCHES_PER_ROUND="4"
MAX_ACTIVE_PATCHES="12"
MAX_PROMPT_TOKENS="1500"
EVAL_MAX_SAMPLES="100"
EXPERIMENT_PRESET="full"
```

## Minimal Local Smoke Tests

There are now two small-run modes:

```text
smoke       small data, mock model backend, checks pipeline only
real-smoke  small data, real configured model, checks API/local-model behavior
```

Mock smoke:

```powershell
& "C:\Users\30301\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m scripts.run_aaai_experiment --config config/appe_aaai.yaml --seed 42 --mode smoke
```

Real-model smoke:

```powershell
& "C:\Users\30301\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m scripts.run_aaai_experiment --config config/appe_aaai.yaml --seed 42 --mode real-smoke
```

Use `real-smoke` before any full paper run. It keeps the smoke dataset sizes but uses the model roles from `config/model_config.py`.
For reasoning models that emit hidden/visible thinking text first, keep `max_output_tokens` large enough for the final `<answer>...</answer>` block to appear. The default local config uses 512 output tokens for target/eval for this reason.

## Legacy Minimal Local Smoke Test

Use small settings first:

```env
EVOLUTION_ROUNDS="2"
CANDIDATE_PATCHES_PER_ROUND="2"
EVAL_MAX_SAMPLES="20"
AUTO_GENERATE_BENCHMARK="false"
EXPERIMENT_PRESET="full"
```

Then run:

```powershell
& "C:\Users\30301\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" main.py
```

Outputs are written to `out/`, especially:

- `out/final_prompt.txt`
- `out/final_metrics.json`
- `out/metrics.jsonl`

## Practical Warnings

Local models are more likely to fail strict JSON output. If diagnosis or patch generation fails often, use a stronger local model for `TARGET_MODEL`, lower temperature, or reduce candidate patches per round while debugging.

The current prototype performs patch admission on one replayed adversarial sample plus a clean guard. For final experiments, upgrade this to batch-level replay: evaluate each candidate patch on a small adversarial validation batch and a clean guard batch.
