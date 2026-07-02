#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


BOOKS = [
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH",
    "PHP", "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS",
    "1PE", "2PE", "1JN", "2JN", "3JN", "JUD", "REV",
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def valid_alignment(path: Path) -> bool:
    try:
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    if not rows:
        return False
    if any(float(r.get("duration", 0)) <= 0 for r in rows):
        return False
    return all(float(b["start"]) >= float(a["end"]) - 1e-3 for a, b in zip(rows, rows[1:]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--books", nargs="*", default=BOOKS)
    p.add_argument("--wav-root", default="data/raw/mms_source_zyb/book_audio_wav")
    p.add_argument("--out-root", default="data/manifests")
    p.add_argument("--device", default="cuda:7")
    p.add_argument("--python", default="/home/duhu/anaconda3/envs/qwen3-asr/bin/python")
    p.add_argument("--stars", choices=["false", "true", "both"], default="both")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--log-root", default="data/manifests/logs")
    args = p.parse_args()

    py = Path(sys.executable)
    stars = ["false", "true"] if args.stars == "both" else [args.stars]
    for book in args.books:
        wav = Path(args.wav_root) / f"{book}.wav"
        if not wav.exists():
            print(f"skip missing wav: {wav}", flush=True)
            continue

        tsv = Path(args.out_root) / f"{book}_punct_sentences.tsv"
        if args.overwrite or not tsv.exists():
            run([
                str(py), "local/export_punct_sentences.py",
                "--book", book,
                "--out", str(tsv),
            ])

        for star in stars:
            out = Path(args.out_root) / f"{book}_punct_sentences.use_star_{star}.json"
            if out.exists() and valid_alignment(out) and not args.overwrite:
                print(f"skip existing: {out}", flush=True)
                continue
            tmp_out = out.with_suffix(out.suffix + ".tmp")
            tmp_out.unlink(missing_ok=True)
            log = Path(args.log_root) / f"{book}.use_star_{star}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                args.python, "local/run_mms_sentence_alignment.py",
                "--audio", str(wav),
                "--transcript", str(tsv),
                "--out", str(tmp_out),
                "--device", args.device,
                "--language", "zyb",
            ]
            if star == "true":
                cmd.append("--use-star")
            start = time.time()
            try:
                print("+", " ".join(cmd), flush=True)
                with log.open("w", encoding="utf-8") as f:
                    subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as exc:
                print(f"failed {book} star={star} exit={exc.returncode} log={log}", flush=True)
                if not args.keep_going:
                    raise
                continue
            tmp_out.replace(out)

            rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
            if not valid_alignment(out):
                print(f"invalid {book} star={star}: {out}", flush=True)
                if not args.keep_going:
                    raise SystemExit(1)
                continue
            print(
                f"ok {book} star={star} rows={len(rows)} "
                f"first={rows[0]['start']:.3f}-{rows[0]['end']:.3f} "
                f"last={rows[-1]['start']:.3f}-{rows[-1]['end']:.3f} "
                f"elapsed_s={time.time() - start:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
