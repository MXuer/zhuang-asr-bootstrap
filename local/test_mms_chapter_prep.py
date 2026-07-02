#!/usr/bin/env python3
from pathlib import Path

from mms_chapter_align_smoke import load_chapter_texts


def test_load_mat_chapter_texts():
    chapters = load_chapter_texts(Path("data/raw/mms_source_zyb/chapter_text/MAT"))
    assert len(chapters) == 28
    assert chapters[0]["name"] == "MAT_001"
    assert "Yehsuh" in chapters[0]["text"]
    assert chapters[-1]["name"] == "MAT_028"


if __name__ == "__main__":
    test_load_mat_chapter_texts()
    print("ok")
