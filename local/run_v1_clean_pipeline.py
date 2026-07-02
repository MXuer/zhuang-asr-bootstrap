#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


QWEN_PY = "/home/duhu/anaconda3/envs/qwen3-asr/bin/python"


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def alignment_files(root: Path):
    return sorted(root.glob("*_punct_sentences.use_star_true.json"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--books", nargs="*")
    p.add_argument("--manifest-root", default="data/manifests")
    p.add_argument("--wav-root", default="data/raw/mms_source_zyb/book_audio_wav")
    p.add_argument("--work-root", default="data/work/v1_clean")
    p.add_argument("--out-manifest", default="data/manifests/v1_clean_segments.jsonl")
    p.add_argument("--no-cut", action="store_true")
    args = p.parse_args()

    manifest_root = Path(args.manifest_root)
    books = set(args.books or [])
    corrected = []
    for alignment in alignment_files(manifest_root):
        book = alignment.name.split("_", 1)[0]
        if books and book not in books:
            continue
        vad_npz = Path("data/work/vad") / f"{book}.tenvad.npz"
        if not vad_npz.exists():
            run([QWEN_PY, "local/cache_ten_vad.py", "--book", book, "--wav-root", args.wav_root])
        out = Path(args.work_root) / f"{book}.vad_corrected.jsonl"
        run([
            "python", "local/vad_correct_segments.py",
            "--book", book,
            "--alignment", str(alignment),
            "--vad-npz", str(vad_npz),
            "--wav-root", args.wav_root,
            "--out", str(out),
        ])
        corrected.append(out)

    merged = Path(args.work_root) / "v1_segments_vad_corrected.jsonl"
    merged.parent.mkdir(parents=True, exist_ok=True)
    with merged.open("w", encoding="utf-8") as f:
        for path in corrected:
            f.write(path.read_text(encoding="utf-8"))

    cmd = [
        "python", "local/make_v1_clean_segments.py",
        "--in-manifest", str(merged),
        "--out-manifest", args.out_manifest,
    ]
    if args.no_cut:
        cmd.append("--no-cut")
    run(cmd)
    run(["python", "local/report_v1_clean.py", "--manifest", args.out_manifest])
    rows = [json.loads(x) for x in Path(args.out_manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    print(json.dumps({"books": sorted({r["book_id"] for r in rows}), "segments": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

