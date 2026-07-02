#!/usr/bin/env python3
import argparse
import html.parser
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


DIVINE_URL = "https://www.divinerevelations.info/documents/bible/zhuang_mp3_bible/zhuang_yongbei_nbp_nt_non_drama/"
BIBLEIS_BASE = "https://live.bible.is/bible/ZYBNBP"
BOOKS = [
    ("MAT", 28), ("MRK", 16), ("LUK", 24), ("JHN", 21), ("ACT", 28), ("ROM", 16),
    ("1CO", 16), ("2CO", 13), ("GAL", 6), ("EPH", 6), ("PHP", 4), ("COL", 4),
    ("1TH", 5), ("2TH", 3), ("1TI", 6), ("2TI", 4), ("TIT", 3), ("PHM", 1),
    ("HEB", 13), ("JAS", 5), ("1PE", 5), ("2PE", 3), ("1JN", 5), ("2JN", 1),
    ("3JN", 1), ("JUD", 1), ("REV", 22),
]


class Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "zhuang-asr-inventory/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def download_file(url, dst):
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "zhuang-asr-inventory/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dst)


def divine_audio_urls():
    p = Links()
    p.feed(fetch(DIVINE_URL).decode("utf-8", "replace"))
    seen = {}
    for h in p.links:
        if h.lower().endswith(".mp3"):
            seen[h] = urllib.parse.urljoin(DIVINE_URL, h)
    return list(seen.values())


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def inventory(limit=None):
    audio_by_book = {book: None for book, _ in BOOKS}
    for url in divine_audio_urls():
        m = re.search(r"/(\d{2})(?:_|%20)", url)
        if m and int(m.group(1)) <= len(BOOKS):
            audio_by_book[BOOKS[int(m.group(1)) - 1][0]] = url

    rows = []
    for book, chapters in BOOKS:
        for chapter in range(1, chapters + 1):
            rows.append({
                "source_id": f"zyb_ybnt_{book.lower()}_{chapter:03d}",
                "language_code": "zyb",
                "language_name": "Zhuang, Yongbei",
                "version_code": "YBNT",
                "book_id": book,
                "chapter": chapter,
                "audio_url": audio_by_book.get(book),
                "text_url": f"{BIBLEIS_BASE}/{book}/{chapter}",
                "provider": "divinerevelations_audio_plus_bibleis_text",
                "license_note": "copyrighted/restricted; research inventory only; verify terms before downloading or redistribution",
                "status": "discovered",
            })
            if limit and len(rows) >= limit:
                return rows
    return rows


def download_text(rows, out_root, rate_limit):
    for row in rows:
        dst = out_root / row["book_id"] / f'{row["chapter"]:03d}.html'
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(fetch(row["text_url"]))
            time.sleep(rate_limit)
        row["text_local_path"] = str(dst)


def download_audio(rows, out_root, rate_limit):
    done = {}
    for row in rows:
        url = row["audio_url"]
        if url not in done:
            dst = out_root / f'{row["book_id"]}.mp3'
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                download_file(url, dst)
                time.sleep(rate_limit)
            done[url] = dst
        row["audio_local_path"] = str(done[url])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/manifests/v1_source_inventory.jsonl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--download-text", action="store_true")
    ap.add_argument("--download-audio", action="store_true")
    ap.add_argument("--text-root", default="data/raw/mms_source_zyb/chapter_text")
    ap.add_argument("--audio-root", default="data/raw/mms_source_zyb/book_audio")
    ap.add_argument("--rate-limit-sec", type=float, default=1.0)
    args = ap.parse_args()

    rows = inventory(args.limit)
    if args.download_text:
        download_text(rows, Path(args.text_root), args.rate_limit_sec)
    if args.download_audio:
        download_audio(rows, Path(args.audio_root), args.rate_limit_sec)
    write_jsonl(Path(args.out), rows)
    audio_books = len({r["book_id"] for r in rows if r["audio_url"]})
    print(f"wrote {len(rows)} chapters; audio book files discovered for {audio_books} books")


if __name__ == "__main__":
    main()
