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
- Added local Ten-VAD runtime wrapper and V1 clean scripts.
- Generated full V1 clean data from all 27 `use_star=True` aligned books.
- V1 current output: `data/manifests/v1_clean_segments.jsonl`, `data/wav/v1_clean/`, `reports/v1_clean_audit.md`.
- V1 current audit: 13,544 kept segments, 30 too-long drops, 36.790 hours, 27 books.
- V1 cutting now supports multiprocessing via `--jobs`; use enough workers to target roughly 85%-90% resource utilization.
- Built incomplete-meaning augmentation from V1:
  - Adjacent concat: `data/manifests/v1_pair_concat_segments.jsonl`, 3,228 segments, 16.710 hours, 27 books.
  - Partial span crops: `data/manifests/v1_partial_segments.jsonl`, 39,515 segments, 51.634 hours, 27 books.
  - Audits: `reports/v1_pair_concat_audit.md`, `reports/v1_partial_audit.md`.
- Exported V1 Kaldi splits to `data/kaldi/v1/`:
  - train: 52,676 utterances, 98.036 hours.
  - val: 1,698 utterances, 3.328 hours.
  - test: 1,913 utterances, 3.769 hours.
  - `wav.scp` paths are absolute; train has zero source-unit overlap with val/test.

## Important Paths

- Source plan: `zhuang_asr_pipeline_mmsdata_v1_synth_v2.md`
- Runtime aligner copy: `local/mms_runtime/`
- Sentence export: `local/export_punct_sentences.py`
- Single-book alignment: `local/run_mms_sentence_alignment.py`
- Batch alignment attempt: `local/batch_align_books.py`
- Design/implementation plan: `docs/plans/2026-07-02-zhuang-asr-data-pipeline.md`

## Next Step

Clean up scripts enough to support V1 data cleaning:

1. Add alignment audit script for all books.
2. Review VAD-corrected samples by listening before training.
3. Review augmented samples by listening before training.
4. Start training recipe integration from `data/kaldi/v1/`.
