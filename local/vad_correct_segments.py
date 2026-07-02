#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def vad_bounds(probs, frame_shift_s, start, end, margin, threshold, pad_before, pad_after, audio_duration):
    left = max(0.0, start - margin)
    right = min(audio_duration, end + margin)
    i0 = max(0, int(left / frame_shift_s))
    i1 = min(len(probs), int(right / frame_shift_s) + 1)
    active = np.flatnonzero(probs[i0:i1] >= threshold)
    if active.size == 0:
        return start, end, "no_vad_speech_in_window"
    speech_start = (i0 + int(active[0])) * frame_shift_s
    speech_end = (i0 + int(active[-1]) + 1) * frame_shift_s
    new_start = max(0.0, speech_start - pad_before)
    new_end = min(audio_duration, speech_end + pad_after)
    if new_end <= new_start:
        return start, end, "invalid_vad_bounds"
    return round(new_start, 3), round(new_end, 3), "vad_corrected"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--book", required=True)
    p.add_argument("--alignment", required=True)
    p.add_argument("--vad-npz", required=True)
    p.add_argument("--wav-root", default="data/raw/mms_source_zyb/book_audio_wav")
    p.add_argument("--out", required=True)
    p.add_argument("--margin-sec", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--pad-before-sec", type=float, default=0.08)
    p.add_argument("--pad-after-sec", type=float, default=0.12)
    args = p.parse_args()

    wav = Path(args.wav_root) / f"{args.book}.wav"
    info = sf.info(str(wav))
    audio_duration = info.frames / info.samplerate
    probs = np.load(args.vad_npz)["probs"]
    frame_shift_s = 256 / 16000
    rows = []
    for item in load_jsonl(Path(args.alignment)):
        start = float(item["start"])
        end = float(item["end"])
        vad_start, vad_end, status = vad_bounds(
            probs,
            frame_shift_s,
            start,
            end,
            args.margin_sec,
            args.threshold,
            args.pad_before_sec,
            args.pad_after_sec,
            audio_duration,
        )
        rows.append({
            "utt_id": f'{args.book}_{item["name"]}',
            "book_id": args.book,
            "source_wav": str(wav),
            "align_start": start,
            "align_end": end,
            "vad_start": vad_start,
            "vad_end": vad_end,
            "duration_sec": round(vad_end - vad_start, 3),
            "text": item["text"],
            "alignment_name": item["name"],
            "boundary_status": status,
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()

