# Pipeline From Scratch

This repo stores code only. Large downloaded/generated data is ignored.

## Environment

Use:

```bash
conda activate qwen3-asr
```

Expected core versions:

```text
torch==2.6.0+cu124
torchaudio==2.6.0+cu124
```

Do not use base Python for MMS forced alignment; it segfaulted on longer `forced_align` runs.

## Stage 0: Discover and Download Sources

V1 source pattern:

- Audio: Divine Revelations mirror of Zhuang Yongbei NT non-drama MP3 files.
- Text: Bible.is `ZYBNBP` chapter HTML pages.

Generate inventory:

```bash
python local/01_discover_zyb_sources.py \
  --out data/manifests/v1_source_inventory.jsonl
```

Download audio only with explicit permission and `no-proxy`:

```bash
no-proxy python local/01_discover_zyb_sources.py \
  --download-audio \
  --out data/manifests/v1_source_inventory.with_audio.jsonl
```

Download text:

```bash
python local/01_discover_zyb_sources.py \
  --download-text \
  --out data/manifests/v1_source_inventory.with_text.jsonl
```

## Stage 1: Prepare 16 kHz Mono WAV

Expected location:

```text
data/raw/mms_source_zyb/book_audio_wav/<BOOK>.wav
```

Example:

```bash
ffmpeg -y -i data/raw/mms_source_zyb/book_audio/MAT.mp3 \
  -ac 1 -ar 16000 \
  data/raw/mms_source_zyb/book_audio_wav/MAT.wav
```

## Stage 2: Export Sentence TSV

```bash
python local/export_punct_sentences.py \
  --book MAT \
  --out data/manifests/MAT_punct_sentences.tsv
```

Text policy:

- Keep `，。？！`.
- Map ASCII punctuation: `, -> ，`, `. ; -> 。`, `? -> ？`, `! -> ！`.
- Remove quotes.

## Stage 3: MMS Forced Alignment

```bash
python local/run_mms_sentence_alignment.py \
  --audio data/raw/mms_source_zyb/book_audio_wav/MAT.wav \
  --transcript data/manifests/MAT_punct_sentences.tsv \
  --out data/manifests/MAT_punct_sentences.use_star_true.json \
  --device cuda:7 \
  --language zyb \
  --use-star
```

Batch:

```bash
python local/batch_align_books.py --stars true --device cuda:7
```

## Stage 4: Ten-VAD Cache

```bash
python local/cache_ten_vad.py --book MAT
```

Outputs:

```text
data/work/vad/MAT.tenvad.npz
data/work/vad/MAT.tenvad.json
```

## Stage 5: VAD Boundary Correction

Boundary correction searches only `[align_start - margin, align_end + margin]` converted to frame indices. It does not scan from frame zero for every segment.

```bash
python local/vad_correct_segments.py \
  --book MAT \
  --alignment data/manifests/MAT_punct_sentences.use_star_true.json \
  --vad-npz data/work/vad/MAT.tenvad.npz \
  --out data/work/v1_clean/MAT.vad_corrected.jsonl
```

## Stage 6: Build V1 Clean Data

```bash
python local/run_v1_clean_pipeline.py \
  --books MAT 2JN \
  --out-manifest data/manifests/v1_clean_segments.jsonl
```

Outputs:

```text
data/manifests/v1_clean_segments.jsonl
data/wav/v1_clean/<BOOK>/*.wav
reports/v1_clean_audit.md
```

