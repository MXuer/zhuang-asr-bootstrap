#!/usr/bin/env python3
import html
import json
import re
from pathlib import Path


import argparse


NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>')
SPACE_RE = re.compile(r"\s+")
QUOTE_RE = re.compile(r"[\"“”‘’]")
PUNCT_MAP = str.maketrans({
    ",": "，",
    ".": "。",
    ";": "。",
    "?": "？",
    "!": "！",
    ":": "",
})
SENTENCE_RE = re.compile(r"[^。？！]+[。？！]?")
BOOKS = [
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH",
    "PHP", "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS",
    "1PE", "2PE", "1JN", "2JN", "3JN", "JUD", "REV",
]


def verse_texts(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    match = NEXT_DATA_RE.search(raw)
    if not match:
        raise ValueError(f"missing __NEXT_DATA__: {path}")
    data = json.loads(html.unescape(match.group(1)))
    return [
        SPACE_RE.sub(" ", html.unescape(v.get("verse_text") or "")).strip()
        for v in data["props"]["pageProps"]["chapterText"]
        if (v.get("verse_text") or "").strip()
    ]


def split_sentences(text: str) -> list[str]:
    text = QUOTE_RE.sub("", text)
    text = text.translate(PUNCT_MAP)
    return [
        SPACE_RE.sub(" ", m.group(0)).strip(" ，")
        for m in SENTENCE_RE.finditer(text)
        if m.group(0).strip(" ，")
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", default=None)
    parser.add_argument("--out", default="data/manifests/zyb_punct_sentences.tsv")
    args = parser.parse_args()

    root = Path("data/raw/mms_source_zyb/chapter_text")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    books = [args.book] if args.book else BOOKS
    index = 1
    with out.open("w", encoding="utf-8") as f:
        for book in books:
            for path in sorted((root / book).glob("*.html")):
                for verse in verse_texts(path):
                    for sentence in split_sentences(verse):
                        f.write(f"{index:06d}\t{sentence}\n")
                        index += 1
    print(f"wrote {index - 1} rows to {out}")


if __name__ == "__main__":
    main()
