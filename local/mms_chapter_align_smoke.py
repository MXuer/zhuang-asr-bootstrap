#!/usr/bin/env python3
import argparse
import html
import json
import re
import unicodedata
from pathlib import Path

import soundfile as sf

from mms_runtime.aligner import MmsAligner


NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>')
SPACE_RE = re.compile(r"\s+")


def norm_text(text: str) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^A-Za-z'\-\s]", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def chapter_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    match = NEXT_DATA_RE.search(raw)
    if not match:
        raise ValueError(f"missing __NEXT_DATA__: {path}")
    data = json.loads(html.unescape(match.group(1)))
    verses = data["props"]["pageProps"]["chapterText"]
    parts = [norm_text(v.get("verse_text") or "") for v in verses]
    return SPACE_RE.sub(" ", " ".join(p for p in parts if p)).strip()


def chapter_verses(path: Path, book_id: str = "MAT") -> list[dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    match = NEXT_DATA_RE.search(raw)
    if not match:
        raise ValueError(f"missing __NEXT_DATA__: {path}")
    data = json.loads(html.unescape(match.group(1)))
    verses = data["props"]["pageProps"]["chapterText"]
    rows = []
    for index, verse in enumerate(verses, 1):
        text = norm_text(verse.get("verse_text") or "")
        if text:
            chapter = int(verse.get("chapter") or path.stem)
            verse_id = int(verse.get("verse_start") or index)
            rows.append({
                "name": f"{book_id}_{chapter:03d}_{verse_id:03d}",
                "chapter": chapter,
                "verse": verse_id,
                "text": text,
            })
    return rows


def load_chapter_texts(chapter_dir: Path, book_id: str = "MAT") -> list[dict]:
    chapters = []
    for path in sorted(chapter_dir.glob("*.html")):
        chapter = int(path.stem)
        text = chapter_text(path)
        if text:
            chapters.append({"name": f"{book_id}_{chapter:03d}", "chapter": chapter, "text": text})
    return chapters


def load_verse_texts(chapter_dir: Path, book_id: str = "MAT") -> list[dict]:
    rows = []
    for path in sorted(chapter_dir.glob("*.html")):
        rows.extend(chapter_verses(path, book_id))
    return rows


def align_book(args) -> list[dict]:
    items = load_verse_texts(Path(args.chapter_text_dir), args.book_id) if args.item_level == "verse" else load_chapter_texts(Path(args.chapter_text_dir), args.book_id)
    items = [
        c for c in items
        if int(args.start_chapter) <= int(c["chapter"]) <= int(args.end_chapter)
    ]
    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    aligner = MmsAligner(args.model_path, args.device, args.uroman_path)
    aligned = aligner.align(
        [c["text"] for c in items],
        audio,
        sr,
        [c["name"] for c in items],
        use_star=args.use_star,
        language=args.language,
        raw_transcripts=[c["text"] for c in items],
    )
    by_name = {c["name"]: c for c in items}
    rows = []
    for item in aligned:
        meta = by_name[item["name"]]
        rows.append({
            "book_id": args.book_id,
            "chapter": meta["chapter"],
            "verse": meta.get("verse"),
            "name": item["name"],
            "start_sec": round(float(args.audio_offset_sec) + float(item["start"]), 3),
            "end_sec": round(float(args.audio_offset_sec) + float(item["end"]), 3),
            "duration_sec": item["duration"],
            "text": item["text"],
            "audio_path": args.audio,
        })
    return rows


def rough_chapter_windows(chapters: list[dict], audio_path: str) -> list[dict]:
    info = sf.info(audio_path)
    duration = info.frames / info.samplerate
    weights = [max(len(c["text"]), 1) for c in chapters]
    total = sum(weights)
    t = 0.0
    rows = []
    for c, w in zip(chapters, weights):
        start = t
        t += duration * w / total
        rows.append({
            "book_id": c["name"].split("_")[0],
            "chapter": c["chapter"],
            "name": c["name"],
            "start_sec": round(start, 3),
            "end_sec": round(t, 3),
            "duration_sec": round(t - start, 3),
            "text": c["text"],
            "audio_path": audio_path,
            "alignment": {"method": "text_length_rough_window"},
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--book-id", default="MAT")
    p.add_argument("--audio", default="data/raw/mms_source_zyb/book_audio/MAT.mp3")
    p.add_argument("--chapter-text-dir", default="data/raw/mms_source_zyb/chapter_text/MAT")
    p.add_argument("--out", default="data/manifests/v1_mat_chapter_alignment.jsonl")
    p.add_argument("--model-path", default="/data/duhu/semantic-asr/pretrained_models/mmsalign/model.pt")
    p.add_argument("--uroman-path", default="/data/duhu/semantic-asr/uroman/bin")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--language", default="zyb")
    p.add_argument("--use-star", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--rough-only", action="store_true")
    p.add_argument("--start-chapter", type=int, default=1)
    p.add_argument("--end-chapter", type=int, default=999)
    p.add_argument("--audio-offset-sec", type=float, default=0.0)
    p.add_argument("--item-level", choices=["chapter", "verse"], default="chapter")
    args = p.parse_args()

    if args.prepare_only:
        rows = load_chapter_texts(Path(args.chapter_text_dir), args.book_id)
    elif args.rough_only:
        rows = rough_chapter_windows(load_chapter_texts(Path(args.chapter_text_dir), args.book_id), args.audio)
    else:
        rows = align_book(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
