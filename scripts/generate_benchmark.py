import csv
import json
import os
import sys
import random
# Ensure parent imports resolve
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    BENCHMARK_CONFIG,
    BENCHMARK_DATASET,
    BENCHMARK_SEED,
    BENCHMARK_SIZE,
    DATA_DIR,
    DATASET_ANSWER_COL,
    DATASET_PATH,
    DATASET_QUESTION_COL,
    TRAIN_SET_PATH,
    VAL_SET_PATH,
    TEST_SET_PATH,
)
try:
    from datasets import load_dataset
except ModuleNotFoundError:
    load_dataset = None


def _load_from_csv():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get(DATASET_QUESTION_COL)]

    rng = random.Random(BENCHMARK_SEED)
    if BENCHMARK_SIZE > 0 and len(rows) > BENCHMARK_SIZE:
        rows = rng.sample(rows, BENCHMARK_SIZE)

    clean_data = []
    for i, row in enumerate(rows):
        clean_data.append({
            "id": i + 1,
            "question": row.get(DATASET_QUESTION_COL, "").strip(),
            "answer": row.get(DATASET_ANSWER_COL, "").strip(),
        })
    return clean_data


def _load_from_hf():
    if load_dataset is None:
        raise ModuleNotFoundError(
            "The 'datasets' package is required for HuggingFace benchmark generation. "
            "Set BENCHMARK_DATASET=csv to use a local CSV without this dependency."
        )
    print(f"Fetching {BENCHMARK_DATASET}/{BENCHMARK_CONFIG} from HuggingFace...")
    dataset = load_dataset(BENCHMARK_DATASET, BENCHMARK_CONFIG)
    val_set = dataset["test"].shuffle(seed=BENCHMARK_SEED).select(range(BENCHMARK_SIZE))

    clean_data = []
    for i, item in enumerate(val_set):
        clean_data.append({
            "id": i + 1,
            "question": item["question"],
            "answer": item["answer"].split("#### ")[-1]
        })
    return clean_data


def _split_dataset(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(BENCHMARK_SEED)
    rng.shuffle(rows)

    total = len(rows)
    if total == 0:
        return [], [], []

    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    test_size = total - train_size - val_size

    if total >= 200:
        val_size = max(80, min(val_size, 150))
    if total >= 400:
        test_size = max(200, min(test_size, 300))

    # Rebalance to ensure non-negative sizes
    if train_size + val_size + test_size > total:
        overflow = train_size + val_size + test_size - total
        train_size = max(1, train_size - overflow)
    if train_size <= 0:
        train_size = max(1, total - val_size - test_size)
    if train_size + val_size + test_size != total:
        test_size = max(1, total - train_size - val_size)

    train_set = rows[:train_size]
    val_set = rows[train_size:train_size + val_size]
    test_set = rows[train_size + val_size:train_size + val_size + test_size]
    return train_set, val_set, test_set


def generate_benchmark():
    if BENCHMARK_DATASET.lower() == "csv":
        print(f"Loading evaluation set from local CSV: {DATASET_PATH}")
        clean_data = _load_from_csv()
    else:
        clean_data = _load_from_hf()

    train_set, val_set, test_set = _split_dataset(clean_data)

    # Save datasets to disk
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(TRAIN_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open(VAL_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(val_set, f, ensure_ascii=False, indent=2)
    with open(TEST_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    print("Done. Saved to data/ (train.json, val.json, test.json).")

if __name__ == "__main__":
    generate_benchmark()
