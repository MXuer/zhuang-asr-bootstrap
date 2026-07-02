#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * p / 100)))
    return values[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/v1_clean_segments.jsonl")
    parser.add_argument("--out", default="reports/v1_clean_audit.md")
    args = parser.parse_args()

    rows = [json.loads(x) for x in Path(args.manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    durations = [float(r["duration_sec"]) for r in rows]
    books = sorted({r["book_id"] for r in rows})
    total_hours = sum(durations) / 3600
    lines = [
        "# V1 Clean Audit",
        "",
        f"- segments: {len(rows)}",
        f"- books: {', '.join(books)}",
        f"- hours: {total_hours:.3f}",
        f"- duration p50/p90/p99: {percentile(durations, 50):.3f} / {percentile(durations, 90):.3f} / {percentile(durations, 99):.3f}",
        "",
        "## First Samples",
        "",
    ]
    for r in rows[:20]:
        lines.append(f"- `{r['utt_id']}` {r['duration_sec']}s {r['text'][:80]}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

