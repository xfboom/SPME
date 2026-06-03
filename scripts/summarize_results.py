from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


NUMERIC_KEYS = [
    "val_acc",
    "train_acc",
    "clean_acc",
    "adv_acc",
    "baseline_val_acc",
    "baseline_clean_acc",
    "baseline_adv_acc",
    "baseline_test_acc",
    "final_full_dataset_acc",
    "blue_trigger_count",
    "blue_merge_count",
    "blue_reject_count",
    "blue_error_count",
]


def _load_final_metrics(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in root.rglob("final_metrics.json"):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["run_dir"] = str(path.parent)
        rows.append(payload)
    return rows


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        preset = row.get("experiment_preset", "unknown")
        groups[preset].append(row)
    return groups


def _aggregate_rows(rows: list[dict]) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {
        "n_runs": len(rows),
    }
    for key in NUMERIC_KEYS:
        values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return summary


def _write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(rows: list[dict], output_path: Path) -> None:
    if not rows:
        output_path.write_text("No results found.\n", encoding="utf-8")
        return
    keys = ["experiment_preset", "n_runs", "val_acc_mean", "clean_acc_mean", "adv_acc_mean", "blue_merge_count_mean", "blue_reject_count_mean"]
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize APO experiment outputs into tables.")
    parser.add_argument("--root", default="out", help="Root directory that contains run folders.")
    parser.add_argument("--csv", default="", help="Optional CSV output path.")
    parser.add_argument("--md", default="", help="Optional Markdown output path.")
    args = parser.parse_args()

    root = Path(args.root)
    rows = _load_final_metrics(root)
    if not rows:
        print(f"No final_metrics.json files found under {root}")
        return 1

    grouped = _group_rows(rows)
    summary_rows: list[dict] = []
    for preset, preset_rows in sorted(grouped.items()):
        summary = _aggregate_rows(preset_rows)
        summary["experiment_preset"] = preset
        summary_rows.append(summary)

    summary_rows.sort(key=lambda row: row.get("experiment_preset", ""))

    csv_path = Path(args.csv) if args.csv else root / "summary.csv"
    md_path = Path(args.md) if args.md else root / "summary.md"
    _write_csv(summary_rows, csv_path)
    _write_markdown(summary_rows, md_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())