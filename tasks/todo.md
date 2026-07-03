# Todo

- [x] Record current decisions, lessons, and progress.
- [x] Write the pipeline plan.
- [x] Fix punctuation export to Chinese punctuation policy for all books.
- [x] Make alignment batch runner safer: direct qwen3 Python, per-book logs, temp output, validation before skip.
- [x] Inspect `ten_vad.py` and copy only the minimal VAD interface needed for boundary correction.
- [x] Build V1 clean manifest from currently available `use_star=True` alignment.
- [x] Cut V1 clean WAV segments for current books.
- [x] Generate V1 audit report.
- [x] Document from-scratch pipeline and `qwen3-asr` environment.
- [x] Finish remaining book alignments and rerun V1 clean.
- [ ] Add all-book alignment audit script.
- [ ] Review VAD-corrected samples by listening before training.
- [ ] Prepare WeNet data files from V1 clean manifest.
- [x] Build incomplete-meaning augmentation: adjacent sentence concat and partial-span crops.

## Review

- Current cleanup is documentation-first. No existing alignment outputs were moved or rewritten.
- Incomplete-meaning augmentation is generated as separate dataset versions, not mixed into V1 clean.
