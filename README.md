# Zhuang ASR Bootstrap

Bootstrap pipeline for acquiring and cleaning Yongbei Zhuang (`zyb`) read-speech ASR data.

This repository stores code and documentation only. Downloaded audio/text, generated manifests, model checkpoints, synthesized audio, and voice-conversion outputs are intentionally ignored.

## Layout

- `local/01_discover_zyb_sources.py`: discover/download Bible source inventory.
- `local/export_punct_sentences.py`: export per-book sentence TSV with Chinese punctuation policy.
- `local/run_mms_sentence_alignment.py`: run MMS forced alignment for one book.
- `local/batch_align_books.py`: resumable batch alignment across books.
- `local/mms_runtime/`: local MMS alignment runtime copy.
- `local/synthesis_vc/`: reference scripts for MMS-TTS, OpenVoice V2 conversion, and audio comparison demos.
- `docs/plans/`: implementation plans.
- `tasks/`: project progress, decisions, todo, and lessons.

## Data Policy

Do not commit:

- `data/raw/`
- `data/work/`
- `data/wav/`
- generated manifests under `data/manifests/`
- `outputs/`
- audio/model/checkpoint/archive files

## Current Direction

V1 uses `use_star=True` MMS alignment plus ten-vad boundary correction to create a clean real-speech ASR manifest. Later versions add segment composition, acoustic augmentation, and OpenVoice V2 voice conversion.

