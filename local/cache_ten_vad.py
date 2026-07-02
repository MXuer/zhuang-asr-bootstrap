#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from vad.ten_vad_runtime import TenVadAdapter, TenVadConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--book", required=True)
    p.add_argument("--wav-root", default="data/raw/mms_source_zyb/book_audio_wav")
    p.add_argument("--out-root", default="data/work/vad")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    wav = Path(args.wav_root) / f"{args.book}.wav"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    npz_path = out_root / f"{args.book}.tenvad.npz"
    meta_path = out_root / f"{args.book}.tenvad.json"

    vad = TenVadAdapter(TenVadConfig(threshold=args.threshold))
    result = vad.detect(str(wav))
    probs = np.asarray(result["frame_speech_probs"]["probs"], dtype=np.float32)
    np.savez_compressed(npz_path, probs=probs)
    meta = {
        "book": args.book,
        "wav_path": str(wav),
        "npz_path": str(npz_path),
        "threshold": args.threshold,
        "frame_shift_ms": result["frame_speech_probs"]["frame_shift_ms"],
        "frame_length_ms": result["frame_speech_probs"]["frame_length_ms"],
        "timestamps": result["timestamps"],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {npz_path} and {meta_path}")


if __name__ == "__main__":
    main()

