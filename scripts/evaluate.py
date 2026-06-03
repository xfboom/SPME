import json
import sys
import os
import random
import csv
import re
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.LLMClient import LLMClient
from scripts.aaai_utils import extract_answer, is_correct_output
from config.settings import DATASET_ANSWER_COL, DATASET_QUESTION_COL, EVAL_MAX_SAMPLES, EVAL_RANDOM_SEED
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

# Load environment variables from .env (e.g., OPENAI_API_KEY)
load_dotenv()

def _load_test_data(test_file_path: str):
    ext = os.path.splitext(test_file_path)[1].lower()
    if ext == ".csv":
        with open(test_file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [
                {
                    "question": row.get(DATASET_QUESTION_COL, "").strip(),
                    "answer": row.get(DATASET_ANSWER_COL, "").strip(),
                }
                for row in reader
                if row.get(DATASET_QUESTION_COL)
            ]
    with open(test_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_number(text: str) -> float | None:
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _is_numeric_answer(answer: str) -> bool:
    return _extract_number(answer) is not None and answer.strip() != ""


def _extract_answer(text: str) -> str:
    return extract_answer(text)


def _is_correct(response: str, answer: str) -> bool:
    return is_correct_output(response, answer)


def is_correct_response(response: str, expected_answer: str) -> bool:
    return _is_correct(response, expected_answer)

async def _arun_evaluation(system_prompt: str, test_data: list, model_name: str | None, log: bool) -> int:
    target_llm = LLMClient(model_name=model_name or "gpt-3.5-turbo", temperature=0.0)
    
    # 限制并发量，防止过高导致 API Rate Limit
    sem = asyncio.Semaphore(15)
    
    async def process_item(i, item, total):
        async with sem:
            full_prompt = f"{system_prompt}\n\nQuestion: {item['question']}"
            try:
                response = await target_llm.ainvoke(full_prompt)
                is_corr = _is_correct(response, str(item["answer"]))
            except Exception as e:
                if log:
                    print(f"    [{i+1}/{total}] Error: {e}")
                return 0
                
            if log:
                q_brief = item['question'].replace('\n', ' ')[:40]
                result_label = "Correct" if is_corr else "Wrong"
                print(f"    [{i+1}/{total}] Result: {result_label} | Q: {q_brief}...")
            return 1 if is_corr else 0

    tasks = [process_item(i, item, len(test_data)) for i, item in enumerate(test_data)]
    results = await asyncio.gather(*tasks)
    return sum(results)

def run_evaluation(system_prompt: str, test_file_path: str, log: bool = False, max_samples: int = None, model_name: str | None = None):
    """
    Generic evaluator: takes a prompt and a test file, returns accuracy. Runs via asyncio for speed.
    """
    test_data = _load_test_data(test_file_path)

    eval_limit = max_samples if max_samples is not None else EVAL_MAX_SAMPLES
    if eval_limit > 0 and len(test_data) > eval_limit:
        rng = random.Random(EVAL_RANDOM_SEED)
        test_data = rng.sample(test_data, eval_limit)

    if log:
        print(
            "  -> [Prompt Eval] "
            f"dataset={os.path.basename(test_file_path)}, "
            f"samples={len(test_data)}, "
            f"seed={EVAL_RANDOM_SEED}"
        )
        print("  -> [Evaluation Prompt]\n" + system_prompt)
        print("  -> [Scoring] Must use <answer> tag; Numeric: last number exact match; Non-numeric: exact match")
        
    correct_count = asyncio.run(_arun_evaluation(system_prompt, test_data, model_name, log))
            
    total = len(test_data)
    accuracy = correct_count / total if total > 0 else 0
    return accuracy

if __name__ == "__main__":
    # Quick baseline test
    baseline_prompt = "Please think step by step and answer."
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    clean_path = os.path.join(data_dir, "test_clean.json")
    adv_path = os.path.join(data_dir, "test_adv.json")
    
    if not os.path.exists(clean_path):
        print("Please run python scripts/generate_benchmark.py to generate the test set first.")
        sys.exit(1)
        
    print("Evaluating clean and adversarial sets...")
    acc_clean = run_evaluation(baseline_prompt, clean_path)
    acc_adv = run_evaluation(baseline_prompt, adv_path)
    
    print("\n=== Baseline Results ===")
    print(f"Clean accuracy: {acc_clean * 100:.1f}%")
    print(f"Adversarial accuracy: {acc_adv * 100:.1f}%")
    
    if acc_clean > 0:
        print(f"PDR (performance drop rate): {((acc_clean - acc_adv) / acc_clean) * 100:.2f}%")
