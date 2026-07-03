#!/usr/bin/env python3
from make_v1_concat_segments import concat_candidates
from make_v1_partial_segments import word_windows


def test_concat_candidates_keeps_adjacent_segments_in_same_book():
    rows = [
        {"utt_id": "A_1", "book_id": "A", "vad_start": 1.0, "vad_end": 2.0, "text": "one。"},
        {"utt_id": "A_2", "book_id": "A", "vad_start": 2.4, "vad_end": 3.0, "text": "two。"},
        {"utt_id": "B_1", "book_id": "B", "vad_start": 1.0, "vad_end": 2.0, "text": "skip。"},
    ]
    got = list(concat_candidates(rows, max_gap=0.5))
    assert len(got) == 1
    assert got[0]["utt_id"] == "A_1_concat2"
    assert got[0]["duration_sec"] == 2.0
    assert got[0]["text"] == "one。two。"


def test_word_windows_returns_front_back_middle_for_long_text():
    got = word_windows("a b c d e f g h", min_words=3)
    assert got == [
        ("front", ["a", "b", "c", "d"]),
        ("back", ["e", "f", "g", "h"]),
        ("middle", ["c", "d", "e", "f"]),
    ]


def test_word_windows_skips_short_text():
    assert word_windows("a b", min_words=3) == []


if __name__ == "__main__":
    test_concat_candidates_keeps_adjacent_segments_in_same_book()
    test_word_windows_returns_front_back_middle_for_long_text()
    test_word_windows_skips_short_text()
    print("ok")
