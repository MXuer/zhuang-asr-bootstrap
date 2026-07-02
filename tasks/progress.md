# Progress

## Current State

- Downloaded 27 NT book audio files and 260 chapter HTML files.
- Converted/located 27 book-level WAV files in `data/raw/mms_source_zyb/book_audio_wav/`.
- Extracted MAT punctuation-split text to `data/manifests/MAT_punct_sentences.tsv`.
- MAT alignment completed for both:
  - `data/manifests/MAT_punct_sentences.use_star_false.json`
  - `data/manifests/MAT_punct_sentences.use_star_true.json`
- `2JN` alignment completed for both star modes as a batch smoke test.
- `MRK` completed for `use_star=False`; `use_star=True` batch job dropped and needs a safer resumable runner.
- Added project task files required by `AGENTS.md`.
- Added `docs/plans/2026-07-02-zhuang-asr-data-pipeline.md`.
- Updated `local/export_punct_sentences.py` to preserve Chinese punctuation policy in new exports.
- Updated `local/batch_align_books.py` to use direct qwen3 Python, `.tmp` output, per-book logs, and output validation.
- Added `.gitignore` for large data/model/audio artifacts.

## Important Paths

- Source plan: `zhuang_asr_pipeline_mmsdata_v1_synth_v2.md`
- Runtime aligner copy: `local/mms_runtime/`
- Sentence export: `local/export_punct_sentences.py`
- Single-book alignment: `local/run_mms_sentence_alignment.py`
- Batch alignment attempt: `local/batch_align_books.py`
- Design/implementation plan: `docs/plans/2026-07-02-zhuang-asr-data-pipeline.md`

## Next Step

Clean up scripts enough to support V1 data cleaning:

1. Inspect `ten_vad.py` and decide the smallest local wrapper.
2. Add alignment audit script for existing/new `use_star=True` results.
3. Add V1 clean manifest builder: `use_star=True alignment + ten-vad boundary correction + filters`.
4. Cut WAV segments and write `v1_clean_segments.jsonl`.
