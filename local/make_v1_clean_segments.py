#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from pathlib import Path

QUOTE_RE = re.compile(r"[\"“”‘’]")
PUNCT_MAP = str.maketrans({",": "，", ".": "。", ";": "。", "?": "？", "!": "！", ":": ""})


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def train_text(text: str) -> str:
    text = QUOTE_RE.sub("", str(text))
    return re.sub(r"\s+", " ", text.translate(PUNCT_MAP)).strip()


def keep(row, min_sec, max_sec):
    dur = float(row["duration_sec"])
    if dur < min_sec:
        return False, "too_short"
    if dur > max_sec:
        return False, "too_long"
    if not str(row.get("text", "")).strip():
        return False, "empty_text"
    return True, "kept"


def cut_wav(src, dst, start, duration):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", src,
        "-ac", "1",
        "-ar", "16000",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-manifest", required=True)
    p.add_argument("--out-manifest", default="data/manifests/v1_clean_segments.jsonl")
    p.add_argument("--wav-root", default="data/wav/v1_clean")
    p.add_argument("--min-sec", type=float, default=0.5)
    p.add_argument("--max-sec", type=float, default=30.0)
    p.add_argument("--no-cut", action="store_true")
    args = p.parse_args()

    out_rows = []
    drops = {}
    for row in load_jsonl(Path(args.in_manifest)):
        ok, reason = keep(row, args.min_sec, args.max_sec)
        if not ok:
            drops[reason] = drops.get(reason, 0) + 1
            continue
        wav_path = Path(args.wav_root) / row["book_id"] / f'{row["utt_id"]}.wav'
        if not args.no_cut:
            cut_wav(row["source_wav"], wav_path, float(row["vad_start"]), float(row["duration_sec"]))
        item = dict(row)
        item["raw_text"] = row["text"]
        item["text"] = train_text(row["text"])
        item["wav_path"] = str(wav_path)
        item["dataset_version"] = "v1_clean_real_mms_star_tenvad"
        out_rows.append(item)

    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    print(json.dumps({"kept": len(out_rows), "drops": drops, "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
