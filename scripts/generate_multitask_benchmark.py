from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

from config.settings import (
    DATA_DIR,
    MULTITASK_MAX_PER_TASK,
    MULTITASK_SOURCE_ROOTS,
    MULTITASK_TEST_RATIO,
    MULTITASK_TRAIN_RATIO,
    MULTITASK_VAL_RATIO,
    TRAIN_SET_PATH,
    VAL_SET_PATH,
    TEST_SET_PATH,
)


def _normalize_record(question: str, answer: str, source: str) -> dict:
    return {
        "question": question.strip(),
        "answer": str(answer).strip(),
        "source": source,
    }


def _load_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        question_key = "question" if "question" in headers else headers[0] if headers else "question"
        answer_key = "answer" if "answer" in headers else headers[-1] if headers else "answer"
        for row in reader:
            question = row.get(question_key, "").strip()
            answer = row.get(answer_key, "").strip()
            if question:
                rows.append(_normalize_record(question, answer, str(path)))
    return rows


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[dict] = []
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            question = item.get("question") or item.get("input") or item.get("prompt")
            answer = item.get("answer") or item.get("target") or item.get("output")
            if question and answer is not None:
                rows.append(_normalize_record(str(question), str(answer), str(path)))
        return rows

    if isinstance(payload, dict):
        if "examples" in payload and isinstance(payload["examples"], list):
            for item in payload["examples"]:
                if not isinstance(item, dict):
                    continue
                question = item.get("input") or item.get("question")
                answer = item.get("target") or item.get("answer")
                if question and answer is not None:
                    rows.append(_normalize_record(str(question), str(answer), str(path)))
        elif "data" in payload and isinstance(payload["data"], list):
            for item in payload["data"]:
                if not isinstance(item, dict):
                    continue
                question = item.get("question") or item.get("input")
                answer = item.get("answer") or item.get("target")
                if question and answer is not None:
                    rows.append(_normalize_record(str(question), str(answer), str(path)))

    return rows


def _collect_sources() -> list[Path]:
    candidates: list[Path] = []
    for root in MULTITASK_SOURCE_ROOTS:
        root_path = Path(root)
        if not root_path.exists():
            continue
        if root_path.is_file() and root_path.suffix.lower() in {".json", ".csv"}:
            candidates.append(root_path)
        else:
            candidates.extend(sorted(root_path.rglob("*.json")))
            candidates.extend(sorted(root_path.rglob("*.csv")))
    return candidates


def _load_examples(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    return _load_json(path)


def _split_dataset(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(42)
    rng.shuffle(rows)

    total = len(rows)
    if total == 0:
        return [], [], []

    train_size = max(1, int(total * MULTITASK_TRAIN_RATIO))
    val_size = max(1, int(total * MULTITASK_VAL_RATIO))
    test_size = max(1, total - train_size - val_size)

    if train_size + val_size + test_size > total:
        overflow = train_size + val_size + test_size - total
        train_size = max(1, train_size - overflow)

    train_set = rows[:train_size]
    val_set = rows[train_size:train_size + val_size]
    test_set = rows[train_size + val_size:train_size + val_size + test_size]
    return train_set, val_set, test_set


def generate_multitask_benchmark() -> None:
    sources = _collect_sources()
    all_rows: list[dict] = []
    source_counts: dict[str, int] = {}

    for path in sources:
        examples = _load_examples(path)
        if not examples:
            continue
        if MULTITASK_MAX_PER_TASK > 0 and len(examples) > MULTITASK_MAX_PER_TASK:
            examples = random.Random(42).sample(examples, MULTITASK_MAX_PER_TASK)
        source_counts[str(path)] = len(examples)
        all_rows.extend(examples)

    train_set, val_set, test_set = _split_dataset(all_rows)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TRAIN_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open(VAL_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(val_set, f, ensure_ascii=False, indent=2)
    with open(TEST_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    manifest_path = Path(DATA_DIR) / "multitask_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_counts": source_counts,
                "total_examples": len(all_rows),
                "train_size": len(train_set),
                "val_size": len(val_set),
                "test_size": len(test_set),
                "train_ratio": MULTITASK_TRAIN_RATIO,
                "val_ratio": MULTITASK_VAL_RATIO,
                "test_ratio": MULTITASK_TEST_RATIO,
                "max_per_task": MULTITASK_MAX_PER_TASK,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Done. Saved multitask splits to {DATA_DIR} and manifest to {manifest_path}.")


if __name__ == "__main__":
    generate_multitask_benchmark()