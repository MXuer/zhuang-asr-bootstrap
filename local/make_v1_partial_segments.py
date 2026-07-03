#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import soundfile as sf

from mms_runtime.aligner import MmsAligner


WORD_RE = re.compile(r"\S+")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def words(text: str) -> list[str]:
    return WORD_RE.findall(str(text))


def word_windows(text: str, min_words: int = 4):
    ws = words(text)
    n = len(ws)
    if n < min_words:
        return []
    half = max(min_words, n // 2)
    windows = [
        ("front", ws[:half]),
        ("back", ws[n - half:]),
    ]
    if n >= min_words + 2:
        mid_start = max(1, (n - half) // 2)
        windows.append(("middle", ws[mid_start:mid_start + half]))
    return windows


def cut_wav(src: str, dst: Path, start: float, duration: float):
    if dst.exists() and dst.stat().st_size > 44:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", src,
        "-ac", "1", "-ar", "16000",
        str(tmp),
    ], check=True)
    tmp.replace(dst)


def align_words(aligner: MmsAligner, row: dict, language: str):
    ws = words(row["text"])
    if len(ws) < 2:
        return []
    audio, sr = sf.read(row["wav_path"], dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return aligner.align(
        ws,
        audio,
        sr,
        [f"{i:06d}" for i in range(len(ws))],
        use_star=False,
        language=language,
        raw_transcripts=ws,
    )


def make_partial_rows(row: dict, word_alignments: list[dict], wav_root: Path, min_sec: float, max_sec: float, min_words: int):
    aligned_words = [x["text"] for x in word_alignments]
    out = []
    for kind, chunk in word_windows(row["text"], min_words=min_words):
        chunk_len = len(chunk)
        start_idx = next((i for i in range(len(aligned_words) - chunk_len + 1) if aligned_words[i:i + chunk_len] == chunk), None)
        if start_idx is None:
            continue
        span = word_alignments[start_idx:start_idx + chunk_len]
        start = float(span[0]["start"])
        end = float(span[-1]["end"])
        dur = round(end - start, 3)
        if dur < min_sec or dur > max_sec:
            continue
        item = dict(row)
        item.update({
            "utt_id": f'{row["utt_id"]}_part_{kind}',
            "source_utt_id": row["utt_id"],
            "source_wav": row["wav_path"],
            "vad_start": round(start, 3),
            "vad_end": round(end, 3),
            "duration_sec": dur,
            "text": " ".join(chunk),
            "raw_text": " ".join(chunk),
            "augmentation": f"partial_{kind}",
            "dataset_version": "v1_partial_real_mms_word",
        })
        item["wav_path"] = str(wav_root / row["book_id"] / f'{item["utt_id"]}.wav')
        out.append(item)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-manifest", default="data/manifests/v1_clean_segments.jsonl")
    p.add_argument("--out-manifest", default="data/manifests/v1_partial_segments.jsonl")
    p.add_argument("--wav-root", default="data/wav/v1_partial")
    p.add_argument("--model-path", default="/data/duhu/semantic-asr/pretrained_models/mmsalign/model.pt")
    p.add_argument("--uroman-path", default="/data/duhu/semantic-asr/uroman/bin")
    p.add_argument("--device", default="cuda:7")
    p.add_argument("--language", default="zyb")
    p.add_argument("--limit", type=int)
    p.add_argument("--min-words", type=int, default=4)
    p.add_argument("--min-sec", type=float, default=0.5)
    p.add_argument("--max-sec", type=float, default=20.0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--no-cut", action="store_true")
    args = p.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    aligner = MmsAligner(args.model_path, args.device, args.uroman_path)
    out_rows = []
    drops = {"align_failed": 0, "no_partial": 0}
    for seen, row in enumerate(load_jsonl(Path(args.in_manifest))):
        if seen % args.num_shards != args.shard_index:
            continue
        if args.limit and len(out_rows) >= args.limit:
            break
        try:
            word_alignments = align_words(aligner, row, args.language)
            partials = make_partial_rows(row, word_alignments, Path(args.wav_root), args.min_sec, args.max_sec, args.min_words)
        except Exception:
            drops["align_failed"] += 1
            continue
        if not partials:
            drops["no_partial"] += 1
            continue
        for item in partials:
            if args.limit and len(out_rows) >= args.limit:
                break
            if not args.no_cut:
                cut_wav(item["source_wav"], Path(item["wav_path"]), float(item["vad_start"]), float(item["duration_sec"]))
            out_rows.append(item)
        if args.progress_every and (seen + 1) % args.progress_every == 0:
            print(json.dumps({
                "shard": args.shard_index,
                "seen": seen + 1,
                "kept": len(out_rows),
                "drops": drops,
            }, ensure_ascii=False), file=sys.stderr, flush=True)

    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    print(json.dumps({"kept": len(out_rows), "drops": drops, "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
