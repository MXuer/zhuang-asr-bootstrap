# Zhuang ASR Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, audited data pipeline from MMS-aligned Yongbei Zhuang Bible audio to V1 clean ASR manifests, then later augmentation versions.

**Architecture:** Keep the pipeline as small command-line scripts under `local/`, with JSONL/TSV files as stage boundaries. Each stage writes an audit summary so training data can be traced back to source audio, text, alignment, VAD correction, and filters.

**Tech Stack:** Python stdlib, `soundfile`, `ffmpeg`, local MMS aligner, `qwen3-asr` conda Python, ten-vad adapter from `/data/duhu/semantic-asr`, OpenVoice V2 only in later versions.

## Global Constraints

- Follow `AGENTS.md`: update `tasks/progress.md`, `tasks/decisions.md`, `tasks/lessons.md`, and `tasks/todo.md`.
- Before large downloads or installs, ask for confirmation and use `no-proxy`.
- Do not move existing data artifacts unless explicitly requested.
- Use `use_star=True` alignment as the current V1 start point.
- Keep `，。？！`; map ASCII punctuation to Chinese equivalents; remove quotes.
- V1 clean contains real speech only. Augmentation and voice conversion are later versions.
- Every stage must have a runnable validation command and a small audit output.

---

## Version Plan

### V1: Clean Real Read Speech

Input:

- `*_punct_sentences.use_star_true.json`
- `data/raw/mms_source_zyb/book_audio_wav/*.wav`
- sentence TSV per book
- ten-vad frame probabilities

Processing:

1. VAD boundary correction:
   - Search `[start - 1.0s, end + 1.0s]`.
   - Move start to first speech frame above threshold, minus small pad.
   - Move end to last speech frame above threshold, plus small pad.
   - Do not remove silence inside the segment.
2. Text normalization:
   - Training text keeps Chinese punctuation policy: `，。？！`.
   - Quotes removed.
   - Preserve raw text separately.
3. Filtering:
   - Drop nonpositive duration.
   - Drop too short segments `< 0.5s`.
   - Drop too long segments `> 30s` unless explicitly whitelisted.
   - Audit chars/sec and seconds/char outliers.
4. Cut WAV:
   - `data/wav/v1_clean/<book>/<utt_id>.wav`
5. Write:
   - `data/manifests/v1_clean_segments.jsonl`
   - `reports/v1_clean_audit.md`

### V1.1: Alignment and VAD Audit

Checks:

- Alignment result row count vs TSV row count.
- Monotonic timestamps per book.
- No negative or zero duration.
- VAD-corrected boundaries stay within source audio.
- Duration histogram and text length histogram.
- Random sample list for listening review.

### V2: Real Data Composition

1. Adjacent segment concatenation:
   - Join 2 or 3 adjacent same-book segments.
   - Keep total duration `<= 30s`.
   - Text concatenates naturally with punctuation.
2. Internal partial spans:
   - Use word-level alignment inside one segment.
   - Produce prefix/suffix/middle spans.
   - Silence inside the selected span is allowed.
   - Segment must contain the selected text acoustically.

### V3: Acoustic Augmentation

Use simple augmentation first:

- speed perturb
- volume gain
- additive noise
- reverb/rir
- optional codec degradation

Do not let augmented data exceed clean data without an explicit experiment setting.

### V4: OpenVoice V2 Voice Conversion

Use after V1/V2/V3 are validated:

- Sample a subset first, not all 35 hours.
- Randomly select speaker references.
- Run ASR roundtrip QA and filter high-CER outputs.
- Keep original clean data as the anchor.

### Training Experiments

Use Thai WeNet pretrained model as one branch:

- `E0`: Thai pretrained + V1 clean.
- `E1`: random init + V1 clean.
- `E2`: Thai pretrained + V1 clean + V2 composition.
- `E3`: Thai pretrained + V1/V2 + V3 acoustic augmentation.
- `E4`: Thai pretrained + V1/V2/V3 + selected OpenVoice.

Evaluation must use real clean held-out data only.

---

## Immediate Implementation Tasks

### Task 1: Stabilize Text Export

**Files:**
- Modify: `local/export_punct_sentences.py`
- Test: inline `python - <<'PY' ... PY`

**Interfaces:**
- Consumes: `data/raw/mms_source_zyb/chapter_text/<BOOK>/*.html`
- Produces: `data/manifests/<BOOK>_punct_sentences.tsv`

- [ ] Update punctuation mapping:

```python
text = text.translate(str.maketrans({
    ",": "，",
    ".": "。",
    ";": "。",
    "?": "？",
    "!": "！",
    ":": "",
}))
```

- [ ] Remove quote characters:

```python
QUOTE_RE = re.compile(r"[\"“”‘’]")
```

- [ ] Split only on `。？！` after mapping.

- [ ] Validate MAT:

```bash
python local/export_punct_sentences.py --book MAT --out data/manifests/MAT_punct_sentences.tsv
python - <<'PY'
from pathlib import Path
rows = Path("data/manifests/MAT_punct_sentences.tsv").read_text(encoding="utf-8").splitlines()
assert rows
assert all("\t" in r for r in rows)
assert not any(':' in r or '"' in r or '“' in r or '”' in r or '‘' in r or '’' in r for r in rows)
print(len(rows))
PY
```

### Task 2: Stabilize Batch Alignment Runner

**Files:**
- Modify: `local/batch_align_books.py`
- Modify: `local/run_mms_sentence_alignment.py`

**Interfaces:**
- Consumes: per-book TSV and WAV.
- Produces: `data/manifests/<BOOK>_punct_sentences.use_star_true.json`

- [ ] Use `/home/duhu/anaconda3/envs/qwen3-asr/bin/python` directly.
- [ ] Write per-book logs to `data/manifests/logs/<BOOK>.use_star_<mode>.log`.
- [ ] Write output to `.tmp` then rename on success.
- [ ] Skip existing nonempty output unless `--overwrite`.
- [ ] After each book, validate monotonic timestamps.
- [ ] Keep `--keep-going`.

### Task 3: Add Alignment Audit

**Files:**
- Create: `local/audit_alignment.py`

**Interfaces:**
- Consumes: `data/manifests/*_punct_sentences.tsv`, alignment JSONL.
- Produces: `reports/alignment_audit.md`, `data/manifests/alignment_audit.jsonl`

- [ ] For each book/star mode, count source TSV rows and aligned rows.
- [ ] Report dropped row ids.
- [ ] Check monotonic timestamps.
- [ ] Check `duration > 0`.
- [ ] Check `duration <= 60s` warning.
- [ ] Print a compact summary table.

### Task 4: Ten-VAD Boundary Correction

**Files:**
- Inspect: `/data/duhu/semantic-asr/semantic_asr/adapters/ten_vad.py`
- Create: `local/vad_correct_segments.py`

**Interfaces:**
- Consumes: `use_star=True` alignment JSONL and source WAV.
- Produces: `data/manifests/v1_segments_vad_corrected.jsonl`

- [ ] Copy only the minimal ten-vad runtime wrapper needed to produce frame probabilities.
- [ ] Cache per-book VAD probabilities under `data/work/vad/<BOOK>.jsonl` or `.npz`.
- [ ] Correct boundaries using margin/pad/threshold parameters.
- [ ] Preserve original alignment fields:

```json
{
  "align_start": 1.23,
  "align_end": 4.56,
  "vad_start": 1.31,
  "vad_end": 4.49
}
```

### Task 5: Build V1 Clean Manifest and WAVs

**Files:**
- Create: `local/make_v1_clean_segments.py`

**Interfaces:**
- Consumes: `v1_segments_vad_corrected.jsonl`
- Produces: `data/manifests/v1_clean_segments.jsonl`, `data/wav/v1_clean/`

- [ ] Filter duration and text outliers.
- [ ] Cut each segment with `ffmpeg`.
- [ ] Write manifest rows with source book, source file, start/end, raw text, training text.
- [ ] Validate that every `wav_path` exists and has positive duration.

### Task 6: V1 Clean Audit Report

**Files:**
- Create: `local/report_v1_clean.py`
- Create: `reports/v1_clean_audit.md`

**Interfaces:**
- Consumes: `data/manifests/v1_clean_segments.jsonl`
- Produces: human-readable audit.

- [ ] Report total hours, segment count, books covered.
- [ ] Report duration percentiles.
- [ ] Report chars/sec percentiles.
- [ ] List filtered/drop counts by reason.
- [ ] Emit 50 random sample ids for listening review.

