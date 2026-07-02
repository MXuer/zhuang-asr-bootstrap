# Local Pipeline Scripts

- `01_discover_zyb_sources.py`: discover/download source inventory.
- `export_punct_sentences.py`: export per-book punctuation-aware sentence TSV.
- `run_mms_sentence_alignment.py`: align one WAV + TSV with MMS aligner.
- `batch_align_books.py`: run alignment across books; should use qwen3-asr Python directly.
- `mms_runtime/`: local copy of the MMS alignment runtime.
- `mms_chapter_align_smoke.py`: exploratory chapter/verse alignment smoke script.
- `test_mms_chapter_prep.py`: lightweight parser check.

Use `tasks/progress.md` for current state before running more pipeline steps.

