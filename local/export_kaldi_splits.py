#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

from export_punct_sentences import split_sentences, verse_texts


BOOKS = [
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH",
    "PHP", "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS",
    "1PE", "2PE", "1JN", "2JN", "3JN", "JUD", "REV",
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def source_units(row: dict) -> set[str]:
    if row.get("source_utt_ids"):
        return set(row["source_utt_ids"])
    if row.get("source_utt_id"):
        return {row["source_utt_id"]}
    return {row["utt_id"]}


def chapter_by_clean_utt(chapter_root: Path) -> dict[str, str]:
    out = {}
    for book in BOOKS:
        index = 1
        for chapter_no, path in enumerate(sorted((chapter_root / book).glob("*.html")), start=1):
            chapter_id = f"{book}_{chapter_no:03d}"
            for verse in verse_texts(path):
                for _ in split_sentences(verse):
                    out[f"{book}_{index:06d}"] = chapter_id
                    index += 1
    return out


def heldout_units_by_chapter(chapter_by_unit: dict[str, str], val_per_chapter: int, test_per_chapter: int, seed: int):
    rng = random.Random(seed)
    by_chapter = {}
    for unit, chapter in chapter_by_unit.items():
        by_chapter.setdefault(chapter, []).append(unit)
    val_units, test_units = set(), set()
    for chapter, units in sorted(by_chapter.items()):
        units = sorted(units)
        rng.shuffle(units)
        test_take = min(test_per_chapter, len(units))
        val_take = min(val_per_chapter, max(0, len(units) - test_take))
        test_units.update(units[:test_take])
        val_units.update(units[test_take:test_take + val_take])
    return val_units, test_units


def assign_splits(rows: list[dict], chapter_by_unit: dict[str, str], val_per_chapter: int, test_per_chapter: int, seed: int):
    val_units, test_units = heldout_units_by_chapter(chapter_by_unit, val_per_chapter, test_per_chapter, seed)
    changed = True
    while changed:
        changed = False
        for row in rows:
            units = source_units(row)
            if units & test_units:
                before = len(test_units)
                test_units.update(units)
                changed = changed or len(test_units) != before
            elif units & val_units:
                before = len(val_units)
                val_units.update(units)
                changed = changed or len(val_units) != before
    val_units -= test_units
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        units = source_units(row)
        if units & test_units:
            splits["test"].append(row)
        elif units & val_units:
            splits["val"].append(row)
        else:
            splits["train"].append(row)
    return splits


def write_kaldi(rows: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["utt_id"])
    (out_dir / "wav.scp").write_text(
        "".join(f"{r['utt_id']} {Path(r['wav_path']).resolve()}\n" for r in rows),
        encoding="utf-8",
    )
    (out_dir / "text").write_text(
        "".join(f"{r['utt_id']} {r['text']}\n" for r in rows),
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", nargs="+", default=[
        "data/manifests/v1_clean_segments.jsonl",
        "data/manifests/v1_pair_concat_segments.jsonl",
        "data/manifests/v1_partial_segments.jsonl",
    ])
    p.add_argument("--chapter-root", default="data/raw/mms_source_zyb/chapter_text")
    p.add_argument("--out-dir", default="data/kaldi/v1")
    p.add_argument("--val-per-chapter", type=int, default=1)
    p.add_argument("--test-per-chapter", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260703)
    args = p.parse_args()

    rows = []
    for path in args.manifests:
        rows.extend(load_jsonl(Path(path)))
    chapter_map = chapter_by_clean_utt(Path(args.chapter_root))
    splits = assign_splits(rows, chapter_map, args.val_per_chapter, args.test_per_chapter, args.seed)
    out = Path(args.out_dir)
    for name, split_rows in splits.items():
        write_kaldi(split_rows, out / name)
    summary = {
        name: {
            "utts": len(split_rows),
            "hours": round(sum(float(r["duration_sec"]) for r in split_rows) / 3600, 3),
        }
        for name, split_rows in splits.items()
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
