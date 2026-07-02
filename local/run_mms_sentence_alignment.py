#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import soundfile as sf

from mms_runtime.aligner import MmsAligner


def read_tsv(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(line.split("\t", 1))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--transcript", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-path", default="/data/duhu/semantic-asr/pretrained_models/mmsalign/model.pt")
    p.add_argument("--uroman-path", default="/data/duhu/semantic-asr/uroman/bin")
    p.add_argument("--device", default="cuda:7")
    p.add_argument("--language", default="zyb")
    p.add_argument("--use-star", action="store_true")
    args = p.parse_args()

    id_texts = read_tsv(Path(args.transcript))
    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    aligner = MmsAligner(args.model_path, args.device, args.uroman_path)
    aligned = aligner.align(
        [x[1] for x in id_texts],
        audio,
        sr,
        [x[0] for x in id_texts],
        use_star=args.use_star,
        language=args.language,
        raw_transcripts=[x[1] for x in id_texts],
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in aligned) + "\n", encoding="utf-8")
    print(f"wrote {len(aligned)} rows to {out}")


if __name__ == "__main__":
    main()
