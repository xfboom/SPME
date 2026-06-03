from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_metrics(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _series(records: list[dict], key: str) -> list[tuple[int, float]]:
    series: list[tuple[int, float]] = []
    for item in records:
        value = item.get(key)
        epoch = item.get("epoch")
        if isinstance(epoch, int) and isinstance(value, (int, float)):
            series.append((epoch, float(value)))
    return series


def _svg_polyline(series: list[tuple[int, float]], x0: int, y0: int, width: int, height: int, color: str) -> str:
    if not series:
        return ""
    xs = [epoch for epoch, _ in series]
    ys = [value for _, value in series]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1e-6

    points = []
    for epoch, value in series:
        x = x0 + (epoch - min_x) / (max_x - min_x) * width
        y = y0 + height - (value - min_y) / (max_y - min_y) * height
        points.append(f"{x:.1f},{y:.1f}")
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}" />'


def _write_svg(records: list[dict], output_path: Path, title: str) -> None:
    clean_series = _series(records, "clean_acc") or _series(records, "val_acc")
    adv_series = _series(records, "adv_acc")
    merge_series = _series(records, "blue_merge_count")
    reject_series = _series(records, "blue_reject_count")

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720">',
        f'<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="40" y="40" font-family="Arial" font-size="24" fill="#111">{title}</text>',
        '<text x="40" y="72" font-family="Arial" font-size="14" fill="#444">clean/adv accuracy and cumulative patch outcomes</text>',
        '<rect x="40" y="100" width="520" height="240" fill="#fafafa" stroke="#ddd" />',
        '<rect x="640" y="100" width="520" height="240" fill="#fafafa" stroke="#ddd" />',
        '<rect x="40" y="400" width="1120" height="240" fill="#fafafa" stroke="#ddd" />',
        '<text x="50" y="130" font-family="Arial" font-size="16" fill="#111">Accuracy</text>',
        '<text x="650" y="130" font-family="Arial" font-size="16" fill="#111">Patch Decisions</text>',
        '<text x="50" y="430" font-family="Arial" font-size="16" fill="#111">Cumulative Merge / Reject Counts</text>',
        _svg_polyline(clean_series, 70, 160, 460, 150, "#1f77b4"),
        _svg_polyline(adv_series, 70, 160, 460, 150, "#d62728"),
        _svg_polyline(merge_series, 670, 160, 460, 150, "#2ca02c"),
        _svg_polyline(reject_series, 670, 160, 460, 150, "#ff7f0e"),
        _svg_polyline(merge_series, 70, 460, 1060, 150, "#2ca02c"),
        _svg_polyline(reject_series, 70, 460, 1060, 150, "#ff7f0e"),
        '<text x="520" y="346" font-family="Arial" font-size="12" fill="#1f77b4">clean</text>',
        '<text x="560" y="346" font-family="Arial" font-size="12" fill="#d62728">adv</text>',
        '<text x="1070" y="346" font-family="Arial" font-size="12" fill="#2ca02c">merge</text>',
        '<text x="1120" y="346" font-family="Arial" font-size="12" fill="#ff7f0e">reject</text>',
        '<text x="1080" y="646" font-family="Arial" font-size="12" fill="#2ca02c">merge</text>',
        '<text x="1130" y="646" font-family="Arial" font-size="12" fill="#ff7f0e">reject</text>',
        '</svg>',
    ]
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot APO metrics as an SVG report.")
    parser.add_argument("--metrics", required=True, help="Path to metrics.jsonl")
    parser.add_argument("--output", default="metrics.svg", help="Output SVG path")
    parser.add_argument("--title", default="S-MAPE Metrics", help="Chart title")
    args = parser.parse_args()

    records = _load_metrics(Path(args.metrics))
    if not records:
        print(f"No records found in {args.metrics}")
        return 1

    _write_svg(records, Path(args.output), args.title)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())