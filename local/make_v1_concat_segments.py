#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import subprocess
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def concat_candidates(rows: list[dict], max_gap: float):
    rows = sorted(rows, key=lambda r: (r["book_id"], float(r["vad_start"])))
    for left, right in zip(rows, rows[1:]):
        if left["book_id"] != right["book_id"]:
            continue
        gap = float(right["vad_start"]) - float(left["vad_end"])
        if gap < 0 or gap > max_gap:
            continue
        start = float(left["vad_start"])
        end = float(right["vad_end"])
        item = dict(left)
        item.update({
            "utt_id": f'{left["utt_id"]}_concat2',
            "vad_start": round(start, 3),
            "vad_end": round(end, 3),
            "duration_sec": round(end - start, 3),
            "text": f'{left["text"]}{right["text"]}',
            "raw_text": f'{left.get("raw_text", left["text"])}{right.get("raw_text", right["text"])}',
            "source_utt_ids": [left["utt_id"], right["utt_id"]],
            "augmentation": "adjacent_concat2",
            "dataset_version": "v1_pair_concat_real_mms_star_tenvad",
        })
        yield item


def keep(row, min_sec, max_sec):
    dur = float(row["duration_sec"])
    return min_sec <= dur <= max_sec and bool(str(row.get("text", "")).strip())


def cut_wav(row):
    dst = Path(row["wav_path"])
    if dst.exists() and dst.stat().st_size > 44:
        return row["utt_id"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f'{float(row["vad_start"]):.3f}',
        "-t", f'{float(row["duration_sec"]):.3f}',
        "-i", row["source_wav"],
        "-ac", "1", "-ar", "16000",
        str(tmp),
    ], check=True)
    tmp.replace(dst)
    return row["utt_id"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-manifest", default="data/manifests/v1_clean_segments.jsonl")
    p.add_argument("--out-manifest", default="data/manifests/v1_pair_concat_segments.jsonl")
    p.add_argument("--wav-root", default="data/wav/v1_pair_concat")
    p.add_argument("--max-gap", type=float, default=2.0)
    p.add_argument("--min-sec", type=float, default=0.5)
    p.add_argument("--max-sec", type=float, default=30.0)
    p.add_argument("--limit", type=int)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--no-cut", action="store_true")
    args = p.parse_args()

    out_rows = []
    drops = {"too_short_or_long": 0}
    for row in concat_candidates(load_jsonl(Path(args.in_manifest)), args.max_gap):
        if not keep(row, args.min_sec, args.max_sec):
            drops["too_short_or_long"] += 1
            continue
        row["wav_path"] = str(Path(args.wav_root) / row["book_id"] / f'{row["utt_id"]}.wav')
        out_rows.append(row)
        if args.limit and len(out_rows) >= args.limit:
            break

    if not args.no_cut:
        if args.jobs <= 1:
            for row in out_rows:
                cut_wav(row)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                for _ in pool.map(cut_wav, out_rows):
                    pass

    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    print(json.dumps({"kept": len(out_rows), "drops": drops, "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
