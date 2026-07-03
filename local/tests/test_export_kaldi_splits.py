#!/usr/bin/env python3
from export_kaldi_splits import assign_splits, source_units


def test_source_units_tracks_clean_concat_and_partial_rows():
    assert source_units({"utt_id": "MAT_000001"}) == {"MAT_000001"}
    assert source_units({"utt_id": "x", "source_utt_id": "MAT_000001"}) == {"MAT_000001"}
    assert source_units({"utt_id": "x", "source_utt_ids": ["MAT_000001", "MAT_000002"]}) == {"MAT_000001", "MAT_000002"}


def test_assign_splits_keeps_heldout_source_units_out_of_train():
    rows = [
        {"utt_id": "MAT_000001", "book_id": "MAT", "wav_path": "/a.wav", "text": "a"},
        {"utt_id": "MAT_000001_part_front", "book_id": "MAT", "source_utt_id": "MAT_000001", "wav_path": "/b.wav", "text": "b"},
        {"utt_id": "MAT_000001_concat2", "book_id": "MAT", "source_utt_ids": ["MAT_000001", "MAT_000002"], "wav_path": "/c.wav", "text": "c"},
        {"utt_id": "MAT_000002", "book_id": "MAT", "wav_path": "/d.wav", "text": "d"},
        {"utt_id": "MAT_000003", "book_id": "MAT", "wav_path": "/e.wav", "text": "e"},
    ]
    chapter_by_unit = {"MAT_000001": "MAT_001", "MAT_000002": "MAT_001", "MAT_000003": "MAT_001"}
    splits = assign_splits(rows, chapter_by_unit, val_per_chapter=1, test_per_chapter=1, seed=1)
    train_units = set().union(*(source_units(r) for r in splits["train"]))
    heldout_units = set().union(*(source_units(r) for r in splits["val"] + splits["test"]))
    assert not (train_units & heldout_units)
    assert splits["val"]
    assert splits["test"]


if __name__ == "__main__":
    test_source_units_tracks_clean_concat_and_partial_rows()
    test_assign_splits_keeps_heldout_source_units_out_of_train()
    print("ok")
