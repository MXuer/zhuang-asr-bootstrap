# 壮语 ASR Pipeline 改版：V1 先做 MMS-lab-zyb 风格 ASR 数据，V2 再加合成数据

> 用途：给 Codex / 工程执行代理使用。  
> 目标：先用 `mms-1b-all` 背后的 `zyb` 数据来源思路，获取/复现一版真实朗读 ASR 数据；第二版再加入 `MMS-TTS-zyb + ZhuangBench + OpenVoice V2 + augmentation` 的合成数据池。  
> 重要限定：目前没有看到 Meta 直接公开 `zyb` 的最终 ASR 训练数据包。这里的 V1 应命名为 **MMS-lab-zyb style reproduction / acquisition**，也就是从原始章节音频与文本重建一份 ASR 数据，而不是宣称下载到了 Meta 内部打包训练集。

---

## 0. 新版总体路线

旧路线：

```text
ZhuangBench 文本
  -> MMS-TTS-zyb
  -> OpenVoice V2 多音色
  -> 加噪
  -> WeNet 训练
```

新版路线：

```text
V1: MMS-lab-zyb 风格真实朗读 ASR 数据
  原始邕北壮语章节音频 + 对应文本
    -> 规范化
    -> forced alignment / 章节到 verse 或短句切分
    -> 质量过滤
    -> WeNet V1 baseline
    -> MMS-ASR baseline / roundtrip QA

V2: 加合成数据增强
  ZhuangBench / 自采文本
    -> MMS-TTS-zyb clean synthetic
    -> OpenVoice V2 多音色 synthetic
    -> noise/reverb/speed perturb
    -> 与 V1 数据混合训练
    -> WeNet V2 model
```

推荐版本命名：

| 版本 | 数据核心 | 目标 |
|---|---|---|
| `v1_mmslab_zyb_repro` | 邕北壮语章节朗读音频 + 文本，对齐切分后的真实朗读 ASR 数据 | 先建立真实音频 ASR baseline |
| `v1_mmslab_zyb_repro_aug` | V1 + 噪声/混响/speed perturb | 提高 channel 鲁棒性 |
| `v2_mmslab_plus_mms_tts` | V1 + MMS-TTS-zyb clean synthetic | 增加文本覆盖 |
| `v2_mmslab_plus_openvoice` | V1 + MMS-TTS-zyb + OpenVoice V2 多音色 | 增加说话人音色多样性 |
| `v2_full` | V1 + synthetic clean + synthetic VC + augmentation + 可选真实小数据 | 完整低资源 ASR 实验 |

---

## 1. 核心判断

### 1.1 为什么 V1 先做 MMS-lab-zyb 数据

这是合理的，因为：

1. MMS 的 `zyb` ASR/TTS 能力大概率来自同一类邕北壮语《新约》朗读音频与文本。
2. 这类数据是真人朗读，不是 TTS 合成，声学分布比 `MMS-TTS -> OpenVoice` 更真实。
3. 即使是单说话人/少说话人、宗教域、朗读风格，它仍然比 synthetic-only 更适合做 ASR 冷启动。
4. V1 先跑通真实音频数据的下载、对齐、切分、过滤、训练、评估，对后续 V2 更稳。

### 1.2 V1 的限制

V1 不能被当作通用壮语 ASR 数据集：

- 语种/方言主要应按 `zyb` / 邕北壮语理解，不等于全部壮语。
- 领域是 Bible / 宗教文本，和政务、教育、客服、日常口语差别很大。
- 说话风格是朗读，不是自然对话。
- 说话人数量可能很少。
- 对齐质量需要验证。
- 音频/文本来源涉及第三方版权或 API 条款，非商业实验也不要公开分发数据。

### 1.3 V2 的角色

V2 的合成数据不是替代 V1，而是补充：

- 用 ZhuangBench / 自采文本增加文本覆盖；
- 用 MMS-TTS-zyb 生成标准 `zyb` 合成语音；
- 用 OpenVoice V2 增加音色多样性；
- 用 noise/reverb/speed perturb 增强鲁棒性；
- 训练时降低 synthetic 权重，避免 synthetic-to-real gap。

---

## 2. 新仓库结构

请让 Codex 创建或调整为如下结构：

```text
zhuang-asr-bootstrap/
  README.md
  pyproject.toml
  requirements.txt
  .gitignore

  configs/
    pipeline.yaml
    mms_source_zyb.yaml
    text_norm.yaml
    alignment.yaml
    augmentation.yaml
    wenet/
      train_conformer_char_v1.yaml
      train_conformer_char_v2.yaml

  local/
    00_validate_env.py

    # V1: MMS-lab-zyb style data acquisition/reproduction
    01_prepare_mms_source_inventory.py
    02_fetch_or_import_mms_source_zyb.py
    03_prepare_chapter_transcripts.py
    04_align_and_segment_mms_style.py
    05_audio_stats.py
    06_mms_asr_roundtrip.py
    07_filter_segments.py
    08_make_dataset_splits.py
    09_make_wenet_data.py

    # V2: synthetic data extension
    20_prepare_zhuangbench_text.py
    21_synthesize_mms_tts.py
    22_build_openvoice_refs.py
    23_convert_openvoice_v2.py
    24_augment_audio.py
    25_merge_v1_v2_manifests.py

    # common evaluation/reporting
    90_compute_cer.py
    91_make_report.py
    92_manifest_summary.py

  scripts/
    v1_00_validate.sh
    v1_01_inventory.sh
    v1_02_fetch_or_import.sh
    v1_03_prepare_transcripts.sh
    v1_04_align_segment.sh
    v1_05_filter.sh
    v1_06_make_wenet_data.sh
    v1_07_train_wenet.sh
    v1_08_eval.sh

    v2_01_prepare_text.sh
    v2_02_synth_mms_tts.sh
    v2_03_openvoice.sh
    v2_04_augment.sh
    v2_05_merge.sh
    v2_06_train_wenet.sh
    v2_07_eval.sh

  external/
    fairseq/                # MMS data_prep scripts, optional
    OpenVoice/              # V2 才需要
    wenet/
    uroman/                 # MMS data_prep 可能需要

  data/
    raw/
      mms_source_zyb/
        inventory/
        chapter_audio/
        chapter_text/
        chapter_transcripts/
      zhuangbench/
      refs/
      real_eval/
      noise/
      rir/

    work/
      alignment/
      segments/
      tmp/
      cache/

    wav/
      v1_mmslab_zyb_segments/
      v1_mmslab_zyb_augmented/
      v2_mms_tts_clean/
      v2_openvoice_v2/
      v2_augmented/

    manifests/
      v1_source_inventory.jsonl
      v1_chapters.jsonl
      v1_segments_raw.jsonl
      v1_segments_stats.jsonl
      v1_segments_asr_checked.jsonl
      v1_segments_filtered.jsonl
      v1_train.jsonl
      v1_dev.jsonl
      v1_test.jsonl

      v2_text_manifest.jsonl
      v2_mms_tts_clean.jsonl
      v2_openvoice_v2.jsonl
      v2_augmented.jsonl
      v2_merged_train.jsonl
      v2_merged_dev.jsonl
      v2_merged_test.jsonl

    wenet/
      v1/
        train/wav.scp
        train/text
        dev/wav.scp
        dev/text
        test/wav.scp
        test/text
      v2/
        train/wav.scp
        train/text
        dev/wav.scp
        dev/text
        test/wav.scp
        test/text

  exp/
    wenet_v1_mmslab_zyb/
    wenet_v1_mmslab_zyb_aug/
    wenet_v2_full/
    mms_asr_baseline/

  reports/
    v1_report.md
    v2_report.md
    metrics.csv
```

`.gitignore` 必须包含：

```gitignore
data/raw/
data/work/
data/wav/
data/manifests/*.jsonl
exp/
external/OpenVoice/checkpoints/
*.wav
*.flac
*.mp3
*.m4a
*.opus
*.pt
*.pth
*.safetensors
*.ckpt
.env
```

---

## 3. V1：MMS-lab-zyb 风格数据获取/复现

### 3.1 V1 数据来源模式

V1 支持两种模式。

#### Mode A：导入已有数据包

如果用户已经有一份本地数据包，例如：

```text
audio segments + transcript
或
chapter audio + chapter transcript
或
MMS-style manifest
```

则跳过下载，直接导入并转换为本项目 manifest。

输入示例：

```text
data/raw/mms_source_zyb/imported/
  wav/
  text.tsv
  manifest.jsonl
```

#### Mode B：从章节音频与文本复现

如果没有打包数据，则从原始来源获取：

```text
邕北壮语 / Zhuang, Yongbei / zyb / YBNT 章节音频
+ 对应章节或 verse 文本
```

可能来源：

- BibleBrain / Faith Comes By Hearing API；
- Bible.com / YouVersion 音频与文本页面；
- 已有 podcast feed；
- 用户手工下载的章节音频和文本。

注意：Codex 不要默认写侵入式爬虫。优先实现本地导入器和 API adapter；若需要下载，使用 API key、限速、缓存、断点续传，并遵守来源条款。

---

## 4. V1 manifest 设计

### 4.1 source inventory

`data/manifests/v1_source_inventory.jsonl`

每行一个可下载/可导入的章节资源：

```json
{
  "source_id": "zyb_ybnt_mat_001",
  "language_code": "zyb",
  "language_name": "Zhuang, Yongbei",
  "version_code": "YBNT",
  "book_id": "MAT",
  "chapter": 1,
  "audio_url": null,
  "audio_local_path": "data/raw/mms_source_zyb/chapter_audio/MAT/001.mp3",
  "text_url": null,
  "text_local_path": "data/raw/mms_source_zyb/chapter_text/MAT/001.json",
  "provider": "manual_import",
  "license_note": "research-only; do not redistribute",
  "status": "pending"
}
```

### 4.2 chapter manifest

`data/manifests/v1_chapters.jsonl`

每行一个已经下载/导入并规范化的章节：

```json
{
  "chapter_id": "zyb_ybnt_mat_001",
  "language_code": "zyb",
  "book_id": "MAT",
  "chapter": 1,
  "audio_path": "data/raw/mms_source_zyb/chapter_audio/MAT/001.wav",
  "transcript_path": "data/raw/mms_source_zyb/chapter_transcripts/MAT/001.txt",
  "verse_json_path": "data/raw/mms_source_zyb/chapter_transcripts/MAT/001.verses.jsonl",
  "duration_sec": 312.4,
  "sample_rate": 16000,
  "num_verses": 25,
  "provider": "manual_import"
}
```

### 4.3 segment manifest

`data/manifests/v1_segments_raw.jsonl`

每行一个对齐后的短音频片段：

```json
{
  "utt_id": "v1_zyb_ybnt_mat_001_v0001",
  "dataset_version": "v1_mmslab_zyb_repro",
  "language_code": "zyb",
  "book_id": "MAT",
  "chapter": 1,
  "verse_start": 1,
  "verse_end": 1,
  "text": "raw text",
  "norm_text": "normalized text",
  "wav_path": "data/wav/v1_mmslab_zyb_segments/MAT/001/v0001.wav",
  "start_sec": 0.32,
  "end_sec": 7.14,
  "duration_sec": 6.82,
  "sample_rate": 16000,
  "speaker_id": "unknown_ybnt_reader",
  "source_chapter_id": "zyb_ybnt_mat_001",
  "alignment": {
    "method": "mms_data_prep_forced_alignment",
    "score": null,
    "warnings": []
  },
  "quality": {},
  "split": null
}
```

---

## 5. V1 阶段任务

### 5.1 阶段 V1-0：环境准备

```bash
conda create -n zhuang-asr python=3.10 -y
conda activate zhuang-asr

pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate scipy soundfile librosa pandas tqdm pyyaml jiwer editdistance numpy
pip install datasets sentencepiece unidecode pyloudnorm requests rich

mkdir -p external
cd external
git clone https://github.com/facebookresearch/fairseq.git fairseq
git clone https://github.com/wenet-e2e/wenet.git wenet
# uroman 根据 fairseq MMS data_prep README 安装或 clone
```

验收：

```bash
python local/00_validate_env.py
```

输出：

- Python version
- CUDA available
- torchaudio import ok
- transformers import ok
- fairseq MMS data_prep path exists
- wenet path exists

---

### 5.2 阶段 V1-1：建立 source inventory

脚本：`local/01_prepare_mms_source_inventory.py`

职责：

1. 支持手工 inventory TSV/CSV 导入。
2. 支持扫描本地目录，自动发现 `book/chapter` 音频和文本。
3. 支持可选 provider adapter，例如 BibleBrain API；但第一版不要强依赖 API。
4. 输出 `v1_source_inventory.jsonl`。

本地目录约定：

```text
data/raw/mms_source_zyb/manual/
  audio/
    MAT/001.mp3
    MAT/002.mp3
    ...
  text/
    MAT/001.json
    MAT/002.json
    ...
```

输入命令：

```bash
python local/01_prepare_mms_source_inventory.py \
  --mode local_scan \
  --audio_root data/raw/mms_source_zyb/manual/audio \
  --text_root data/raw/mms_source_zyb/manual/text \
  --language_code zyb \
  --version_code YBNT \
  --out_manifest data/manifests/v1_source_inventory.jsonl
```

验收：

- 每个章节都有 `book_id` 和 `chapter`。
- 音频和文本成对存在。
- 输出缺失列表：`missing_audio`、`missing_text`。
- 如果不足 260 章，也允许 smoke test 继续。

---

### 5.3 阶段 V1-2：下载或导入章节音频/文本

脚本：`local/02_fetch_or_import_mms_source_zyb.py`

职责：

1. `manual_import`：复制/规范化用户本地音频和文本。
2. `api_fetch`：如果配置了 API key 和 fileset id，则通过 API 获取。
3. 对下载实现限速、缓存、checksum、断点续传。
4. 把音频转换为 16 kHz mono wav。
5. 生成 `v1_chapters.jsonl`。

配置：`configs/mms_source_zyb.yaml`

```yaml
language_code: zyb
version_code_candidates: [YBNT]
provider_mode: manual_import

manual_import:
  audio_root: data/raw/mms_source_zyb/manual/audio
  text_root: data/raw/mms_source_zyb/manual/text

api_fetch:
  provider: biblebrain
  api_key_env: BIBLEBRAIN_API_KEY
  fileset_audio_id: null
  fileset_text_id: null
  rate_limit_sec: 1.0

output:
  chapter_audio_root: data/raw/mms_source_zyb/chapter_audio
  chapter_text_root: data/raw/mms_source_zyb/chapter_text
  target_sample_rate: 16000
```

命令：

```bash
python local/02_fetch_or_import_mms_source_zyb.py \
  --in_inventory data/manifests/v1_source_inventory.jsonl \
  --config configs/mms_source_zyb.yaml \
  --out_manifest data/manifests/v1_chapters.jsonl
```

验收：

- 每章音频为 16 kHz mono wav。
- 每章文本可解析为 verse 列表或纯文本。
- 失败章节有 `failed_reason`。
- 不把下载 URL 或 API key 写入公开报告。

---

### 5.4 阶段 V1-3：准备章节 transcript

脚本：`local/03_prepare_chapter_transcripts.py`

职责：

1. 将章节文本 JSON/HTML/TXT 统一为 verse JSONL。
2. 将 verse 文本合并为 chapter transcript。
3. 执行壮文文本规范化。
4. 保留 raw_text 和 norm_text。

文本规范化规则：

```python
import re
import unicodedata

ALLOWED_RE = re.compile(r"[^a-zA-Z'\-\s]")
SPACE_RE = re.compile(r"\s+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
DIGIT_RE = re.compile(r"\d")

def normalize_zhuang_text(text: str) -> str | None:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    if CJK_RE.search(text):
        return None
    # V1 章节文本如果有 verse number，不要进入 norm_text。
    text = DIGIT_RE.sub(" ", text)
    text = ALLOWED_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    if len(text) < 2:
        return None
    return text
```

命令：

```bash
python local/03_prepare_chapter_transcripts.py \
  --in_manifest data/manifests/v1_chapters.jsonl \
  --out_manifest data/manifests/v1_chapters.transcripts.jsonl \
  --out_root data/raw/mms_source_zyb/chapter_transcripts \
  --text_norm configs/text_norm.yaml
```

输出示例：

```text
data/raw/mms_source_zyb/chapter_transcripts/MAT/001.txt
data/raw/mms_source_zyb/chapter_transcripts/MAT/001.verses.jsonl
```

验收：

- 每章 `txt` 是纯 norm_text，适合 forced alignment。
- 每章 `verses.jsonl` 保留 `book_id/chapter/verse/raw_text/norm_text`。
- 被过滤 verse 有统计。

---

### 5.5 阶段 V1-4：forced alignment 与切分

脚本：`local/04_align_and_segment_mms_style.py`

推荐优先调用 MMS data_prep 里的 alignment 工具，而不是自己从零写对齐器。

目标：

```text
chapter wav + chapter transcript
  -> word/char alignment
  -> verse-level 或短句级 segments
  -> v1_segments_raw.jsonl
```

命令草案：

```bash
python local/04_align_and_segment_mms_style.py \
  --in_manifest data/manifests/v1_chapters.transcripts.jsonl \
  --out_manifest data/manifests/v1_segments_raw.jsonl \
  --out_wav_root data/wav/v1_mmslab_zyb_segments \
  --fairseq_mms_data_prep external/fairseq/examples/mms/data_prep \
  --uroman_dir external/uroman \
  --lang zyb \
  --target_sr 16000 \
  --min_segment_sec 0.6 \
  --max_segment_sec 20.0
```

Codex 实现建议：

1. 先在 1 个章节上跑通。
2. 如果 MMS data_prep 的脚本输出已经是 manifest，则写 adapter 转成项目 JSONL。
3. 如果官方脚本只输出 segment wav 和 text，则扫描输出目录重建 manifest。
4. 若 verse-level 对齐不稳定，可以先按 alignment 输出的短句片段训练，不强行绑定 verse id。
5. 若某章对齐失败，把整章标记为 `alignment_failed`，不要手动瞎切。

切分策略：

- 优先 verse-level；
- 如果 verse 太长，按标点或 alignment boundary 再切短；
- 如果 segment 小于 0.6 秒，合并相邻 verse 或丢弃；
- 如果 segment 大于 20 秒，尝试再切，否则丢弃或保留到 long-form 实验，不进第一版 WeNet。

---

### 5.6 阶段 V1-5：音频质量统计

复用脚本：`local/05_audio_stats.py`

命令：

```bash
python local/05_audio_stats.py \
  --in_manifest data/manifests/v1_segments_raw.jsonl \
  --out_manifest data/manifests/v1_segments_stats.jsonl
```

统计字段：

```text
duration_sec
sample_rate
num_channels
peak
rms
clip_ratio
silence_ratio
file_size
```

基础过滤阈值：

```yaml
min_duration_sec: 0.6
max_duration_sec: 20.0
max_clip_ratio: 0.001
min_rms: 0.003
max_silence_ratio: 0.60
```

---

### 5.7 阶段 V1-6：MMS-ASR roundtrip QA

脚本：`local/06_mms_asr_roundtrip.py`

目的：用 `facebook/mms-1b-all` 的 `zyb` adapter 反识别 V1 segment，并与 transcript 计算 CER，作为异常过滤和报告指标。

命令：

```bash
python local/06_mms_asr_roundtrip.py \
  --in_manifest data/manifests/v1_segments_stats.jsonl \
  --out_manifest data/manifests/v1_segments_asr_checked.jsonl \
  --model_id facebook/mms-1b-all \
  --target_lang zyb \
  --device cuda \
  --batch_size 1
```

实现逻辑：

```python
from transformers import AutoProcessor, Wav2Vec2ForCTC
import torch

processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(device)
processor.tokenizer.set_target_lang("zyb")
model.load_adapter("zyb")
model.eval()

inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(**inputs).logits
pred_ids = torch.argmax(logits, dim=-1)[0]
pred_text = processor.decode(pred_ids)
```

注意：MMS-ASR 可能和 V1 数据同源，因此 roundtrip CER 不是独立评估，只是 QA。

建议阈值：

```yaml
v1_roundtrip_max_cer: 0.25
v1_roundtrip_warn_cer: 0.15
```

---

### 5.8 阶段 V1-7：过滤 segments

脚本：`local/07_filter_segments.py`

命令：

```bash
python local/07_filter_segments.py \
  --in_manifest data/manifests/v1_segments_asr_checked.jsonl \
  --out_manifest data/manifests/v1_segments_filtered.jsonl \
  --rejected_manifest data/manifests/v1_segments_rejected.jsonl \
  --config configs/pipeline.yaml
```

拒绝原因：

```text
too_short
too_long
missing_audio
bad_text
rms_too_low
clip_ratio_high
silence_ratio_high
alignment_failed
roundtrip_cer_high
```

输出报告：

```text
reports/v1_filter_summary.csv
reports/v1_filter_summary.md
```

---

### 5.9 阶段 V1-8：切 train/dev/test

脚本：`local/08_make_dataset_splits.py`

推荐使用 book-level split，减少文本/音频风格泄漏。

MMS 风格 split 建议：

```text
dev:  Mark / MRK
test: John / JHN
train: 其他 book
```

如果实际数据没有这些 book，fallback：

```text
按 book_id 分组随机切分，不要按 utterance 随机切。
```

命令：

```bash
python local/08_make_dataset_splits.py \
  --in_manifest data/manifests/v1_segments_filtered.jsonl \
  --out_train data/manifests/v1_train.jsonl \
  --out_dev data/manifests/v1_dev.jsonl \
  --out_test data/manifests/v1_test.jsonl \
  --split_strategy mms_book_split \
  --dev_books MRK \
  --test_books JHN \
  --seed 42
```

验收：

- 同一 `book_id/chapter` 不跨 split。
- 每个 split 有小时数统计。
- train/dev/test 中字符集差异可控。
- dev/test 不参与任何训练和过滤调参之外的模型拟合。

---

### 5.10 阶段 V1-9：生成 WeNet 数据

脚本：`local/09_make_wenet_data.py`

命令：

```bash
python local/09_make_wenet_data.py \
  --train_manifest data/manifests/v1_train.jsonl \
  --dev_manifest data/manifests/v1_dev.jsonl \
  --test_manifest data/manifests/v1_test.jsonl \
  --out_dir data/wenet/v1 \
  --vocab_out data/wenet/v1/units.txt
```

输出：

```text
data/wenet/v1/train/wav.scp
data/wenet/v1/train/text
data/wenet/v1/dev/wav.scp
data/wenet/v1/dev/text
data/wenet/v1/test/wav.scp
data/wenet/v1/test/text
data/wenet/v1/units.txt
```

第一版使用 character-level token。

---

### 5.11 阶段 V1-10：训练 WeNet V1

复制 WeNet recipe：

```bash
mkdir -p recipes/zhuang_v1/s0
cp -r external/wenet/examples/librispeech/s0/* recipes/zhuang_v1/s0/
```

改造：

- 数据目录指向 `data/wenet/v1`。
- 使用 character-level units。
- 低资源先用小 Conformer。
- 训练命令先 smoke test，再 full run。

训练命令草案：

```bash
bash scripts/v1_07_train_wenet.sh \
  --data data/wenet/v1 \
  --exp_dir exp/wenet_v1_mmslab_zyb \
  --config configs/wenet/train_conformer_char_v1.yaml
```

V1 实验矩阵：

| 实验 | 数据 | 说明 |
|---|---|---|
| `v1_e0_mms_asr_baseline` | 不训练，直接 MMS-ASR decode V1 dev/test | QA 与 baseline |
| `v1_e1_wenet_clean` | V1 filtered segments | 第一版 WeNet |
| `v1_e2_wenet_clean_aug` | V1 + noise/reverb/speed | channel augmentation |
| `v1_e3_mms_finetune_optional` | V1 train fine-tune MMS-ASR | 强 baseline，可选 |

---

## 6. V2：加入合成数据

V2 在 V1 跑通后再做。不要一开始就把 synthetic 加进去，否则无法判断 V1 数据本身的质量。

### 6.1 V2 文本来源

优先：

- ZhuangBench `parallel_corpus.json` 的壮文侧；
- ZhuangBench 字典构造的词汇覆盖句；
- 自采目标域壮文文本。

不要把 `test_translation_set.json` 放入训练。

脚本：`local/20_prepare_zhuangbench_text.py`

命令：

```bash
python local/20_prepare_zhuangbench_text.py \
  --zhuangbench_dir data/raw/zhuangbench \
  --out_manifest data/manifests/v2_text_manifest.jsonl \
  --text_norm configs/text_norm.yaml \
  --seed 42
```

---

### 6.2 V2 clean synthetic：MMS-TTS-zyb

脚本：`local/21_synthesize_mms_tts.py`

命令：

```bash
python local/21_synthesize_mms_tts.py \
  --in_manifest data/manifests/v2_text_manifest.jsonl \
  --out_dir data/wav/v2_mms_tts_clean \
  --out_manifest data/manifests/v2_mms_tts_clean.jsonl \
  --model_id facebook/mms-tts-zyb \
  --device cuda \
  --seed 555 \
  --target_sr 16000
```

注意：`facebook/mms-tts-zyb` 是单音色。V2 clean synthetic 的价值是文本覆盖，不是说话人覆盖。

---

### 6.3 V2 多音色 synthetic：OpenVoice V2

脚本：

- `local/22_build_openvoice_refs.py`
- `local/23_convert_openvoice_v2.py`

命令：

```bash
python local/22_build_openvoice_refs.py \
  --refs_dir data/raw/refs \
  --out_manifest data/manifests/v2_openvoice_refs.jsonl

python local/23_convert_openvoice_v2.py \
  --in_manifest data/manifests/v2_mms_tts_clean.jsonl \
  --refs_manifest data/manifests/v2_openvoice_refs.jsonl \
  --out_dir data/wav/v2_openvoice_v2 \
  --out_manifest data/manifests/v2_openvoice_v2.jsonl \
  --max_refs_per_utt 2 \
  --device cuda
```

参考音频必须授权。不要使用公众人物或未授权真人声音。

---

### 6.4 V2 augmentation

脚本：`local/24_augment_audio.py`

命令：

```bash
python local/24_augment_audio.py \
  --in_manifest data/manifests/v2_openvoice_v2.jsonl \
  --out_dir data/wav/v2_augmented \
  --out_manifest data/manifests/v2_augmented.jsonl \
  --config configs/augmentation.yaml \
  --seed 42
```

增强策略：

```yaml
target_sr: 16000
max_aug_per_utt: 1
prob:
  speed: 0.30
  noise: 0.40
  reverb: 0.20
  codec: 0.10
speed_factors: [0.9, 1.0, 1.1]
snr_db: [10, 15, 20]
```

避免大幅 pitch shift，因为可能破坏声调线索。

---

### 6.5 V2 合并 V1 + synthetic

脚本：`local/25_merge_v1_v2_manifests.py`

命令：

```bash
python local/25_merge_v1_v2_manifests.py \
  --v1_train data/manifests/v1_train.jsonl \
  --v1_dev data/manifests/v1_dev.jsonl \
  --v1_test data/manifests/v1_test.jsonl \
  --v2_synth_manifests \
    data/manifests/v2_mms_tts_clean.jsonl \
    data/manifests/v2_openvoice_v2.jsonl \
    data/manifests/v2_augmented.jsonl \
  --out_train data/manifests/v2_merged_train.jsonl \
  --out_dev data/manifests/v2_merged_dev.jsonl \
  --out_test data/manifests/v2_merged_test.jsonl \
  --config configs/pipeline.yaml
```

合并规则：

1. V1 dev/test 保持不变，用于评价真实朗读数据。
2. V2 synthetic dev/test 可以单独生成，但不要混入 V1 dev/test 的主指标。
3. Synthetic 默认只加到 train。
4. 如果需要 synthetic dev/test，单独输出：

```text
data/wenet/v2_synth_eval/dev
```

建议采样权重：

```yaml
train_sampling_weight:
  v1_mmslab_zyb_repro: 0.70
  v2_mms_tts_clean: 0.10
  v2_openvoice_v2: 0.15
  v2_augmented: 0.05
```

如果 V1 数据很少，例如低于 5 小时：

```yaml
train_sampling_weight:
  v1_mmslab_zyb_repro: 0.50
  v2_mms_tts_clean: 0.20
  v2_openvoice_v2: 0.20
  v2_augmented: 0.10
```

如果后续加入真实目标域标注数据，应提升真实目标域权重。

---

## 7. WeNet V2 训练

生成 V2 WeNet 数据：

```bash
python local/09_make_wenet_data.py \
  --train_manifest data/manifests/v2_merged_train.jsonl \
  --dev_manifest data/manifests/v1_dev.jsonl \
  --test_manifest data/manifests/v1_test.jsonl \
  --out_dir data/wenet/v2 \
  --vocab_out data/wenet/v2/units.txt
```

训练：

```bash
bash scripts/v2_06_train_wenet.sh \
  --data data/wenet/v2 \
  --exp_dir exp/wenet_v2_full \
  --config configs/wenet/train_conformer_char_v2.yaml
```

V2 实验矩阵：

| 实验 | 训练数据 | dev/test | 目的 |
|---|---|---|---|
| `v2_e0_v1_clean_baseline` | V1 only | V1 dev/test | 对照 |
| `v2_e1_v1_plus_tts_clean` | V1 + MMS-TTS clean | V1 dev/test + synth dev | 看文本扩展是否有益 |
| `v2_e2_v1_plus_openvoice` | V1 + MMS-TTS + OpenVoice | V1 dev/test + synth multi-speaker | 看音色变体是否有益 |
| `v2_e3_v1_plus_all_aug` | V1 + clean + VC + noise | V1 dev/test + noisy synth | 看鲁棒性 |
| `v2_e4_real_eval_optional` | 同上 | 外部真实壮语 test | 真正判断泛化 |

---

## 8. 评估设计

### 8.1 V1 评估

V1 报告必须包含：

```text
v1_dev_cer
v1_test_cer
per_book_cer
per_duration_bin_cer
roundtrip_mms_asr_cer
wenet_v1_cer
```

注意：如果 MMS-ASR 和 V1 数据同源，MMS-ASR 的结果可能偏乐观。它是 baseline，不是最终真值。

### 8.2 V2 评估

V2 报告必须分开：

```text
V1 dev/test CER          # 真实朗读 Bible 域
Synthetic clean CER      # 合成域
Synthetic OpenVoice CER  # 合成多音色域
Synthetic noisy CER      # 增强域
Real external CER        # 如果有，最重要
```

不要只报一个混合 CER。

### 8.3 外部真实评测集

即使 V1 是真人朗读，也仍然建议做外部真实评测：

```text
10-20 名说话人
每人 3-5 分钟
总计 30-60 分钟
内容：朗读目标域文本 + 自由口语
```

它的作用是判断：模型是否只学会了 Bible 朗读域。

---

## 9. 推荐里程碑

### Milestone A：V1 smoke test

目标：用 2-5 个章节跑通数据链路。

步骤：

```text
1. 手工放入 2-5 个章节音频和文本
2. local_scan 生成 inventory
3. 导入并转 16k wav
4. 章节 transcript 规范化
5. forced alignment + segmentation
6. 过滤
7. 生成 WeNet 数据
8. WeNet smoke train
```

验收：

- 至少 100 个 segment。
- 每个 segment 有 wav 和 norm_text。
- 可以成功训练一个小模型并 decode dev。

### Milestone B：V1 full

目标：尽量处理完整 260 章或可获得章节。

验收：

- 成功章节数。
- 总小时数。
- 过滤后小时数。
- train/dev/test 小时数。
- V1 WeNet CER。

### Milestone C：V2 synthetic clean

目标：只加 MMS-TTS clean synthetic。

验收：

- 合成音频可播放。
- 合成文本与 V1 split 无泄漏。
- V2_e1 与 V1 baseline 对比。

### Milestone D：V2 OpenVoice 多音色

目标：加入 3-8 个授权参考音色。

验收：

- 每个参考音色抽检。
- 反识别 CER 不显著恶化。
- V2_e2 与 V2_e1 对比。

### Milestone E：V2 augmentation

目标：加入轻量噪声/混响/speed。

验收：

- noisy synthetic test 上鲁棒性提升。
- V1 dev/test 不明显下降。

---

## 10. 配置文件草案

### 10.1 `configs/pipeline.yaml`

```yaml
project:
  language_code: zyb
  language_name: Zhuang, Yongbei
  mode: research_noncommercial

text_norm:
  lowercase: true
  unicode_norm: NFKC
  keep_chars_regex: "[a-zA-Z'\\-\\s]"
  remove_digits: true
  remove_cjk: true
  min_chars: 2
  max_chars: 220

v1_filter:
  min_duration_sec: 0.6
  max_duration_sec: 20.0
  max_clip_ratio: 0.001
  min_rms: 0.003
  max_silence_ratio: 0.60
  roundtrip_max_cer: 0.25

v1_split:
  strategy: mms_book_split
  dev_books: [MRK]
  test_books: [JHN]
  fallback_strategy: book_group_random
  seed: 42

v2_synthetic:
  max_refs_per_utt: 2
  keep_mms_clean: true
  openvoice_enabled: true
  augmentation_enabled: true

train_sampling_weight:
  v1_mmslab_zyb_repro: 0.70
  v2_mms_tts_clean: 0.10
  v2_openvoice_v2: 0.15
  v2_augmented: 0.05
```

### 10.2 `configs/alignment.yaml`

```yaml
alignment:
  method: mms_data_prep
  lang: zyb
  target_sr: 16000
  min_segment_sec: 0.6
  max_segment_sec: 20.0
  allow_merge_short_verses: true
  allow_split_long_verses: true
  fail_chapter_on_empty_alignment: true
```

### 10.3 `configs/augmentation.yaml`

```yaml
target_sr: 16000
max_aug_per_utt: 1
prob:
  speed: 0.30
  noise: 0.40
  reverb: 0.20
  codec: 0.10
speed_factors: [0.9, 1.0, 1.1]
snr_db: [10, 15, 20]
gain_db_range: [-6, 3]
forbid_pitch_shift: true
```

---

## 11. 风险与注意事项

### 11.1 不要把 V1 写成“已下载官方 MMS 训练集”

正确表述：

```text
V1 is a reproduction/acquisition of MMS-lab-zyb-style paired ASR data
from available Zhuang, Yongbei chapter audio and text sources.
```

错误表述：

```text
We downloaded Meta's zyb ASR training data.
```

### 11.2 V1 是真实音频，但不是通用语音

V1 比 synthetic 更真实，但仍有强 domain bias：

```text
Bible / New Testament
朗读风格
可能单说话人
zyb / 邕北壮语
```

### 11.3 V2 synthetic 不要盖过 V1

V2 的合成数据是增强，不是主数据。训练采样权重建议 V1 至少 50%-70%。

### 11.4 OpenVoice 可能破坏声调

OpenVoice 主要改音色，不保证保留壮语声调/韵律。V2 中必须保留 MMS-TTS clean，并对 OpenVoice 样本抽检与 roundtrip。

### 11.5 版权与数据治理

即使非商业实验：

- 不公开发布原始章节音频；
- 不公开发布切分后的 segment 数据；
- 不提交 API key；
- 不提交参考人声音频；
- 报告里写清来源和使用限制。

---

## 12. Codex 最终执行顺序

推荐按这个顺序执行，不要跳到 V2：

```text
A. V1 smoke test
  1. 建仓库结构
  2. 准备 2-5 个章节音频/文本
  3. inventory
  4. import/convert
  5. transcript normalization
  6. alignment/segmentation
  7. stats/filter
  8. WeNet data
  9. WeNet smoke training

B. V1 full
  1. 扩展到所有可获得章节
  2. 完整过滤和 split
  3. WeNet V1 baseline
  4. MMS-ASR baseline
  5. v1_report.md

C. V2 synthetic clean
  1. ZhuangBench 文本
  2. MMS-TTS-zyb 合成
  3. merge with V1 train
  4. WeNet V2_e1

D. V2 OpenVoice + augmentation
  1. 授权参考音色
  2. OpenVoice 多音色
  3. augmentation
  4. merge with V1 train
  5. WeNet V2_e2/e3
  6. v2_report.md
```

---

## 13. 最小命令清单

### V1 smoke

```bash
bash scripts/v1_00_validate.sh
bash scripts/v1_01_inventory.sh
bash scripts/v1_02_fetch_or_import.sh
bash scripts/v1_03_prepare_transcripts.sh
bash scripts/v1_04_align_segment.sh
bash scripts/v1_05_filter.sh
bash scripts/v1_06_make_wenet_data.sh
bash scripts/v1_07_train_wenet.sh --smoke true
bash scripts/v1_08_eval.sh
```

### V2 after V1

```bash
bash scripts/v2_01_prepare_text.sh
bash scripts/v2_02_synth_mms_tts.sh
bash scripts/v2_03_openvoice.sh
bash scripts/v2_04_augment.sh
bash scripts/v2_05_merge.sh
bash scripts/v2_06_train_wenet.sh
bash scripts/v2_07_eval.sh
```

---

## 14. 官方参考链接

Codex 实现时优先查这些官方来源：

- MMS fairseq README: `https://github.com/facebookresearch/fairseq/blob/main/examples/mms/README.md`
- MMS data prep README: `https://github.com/facebookresearch/fairseq/blob/main/examples/mms/data_prep/README.md`
- MMS paper: `https://arxiv.org/pdf/2305.13516`
- MMS-ASR Hugging Face: `https://huggingface.co/facebook/mms-1b-all`
- MMS-TTS-zyb Hugging Face: `https://huggingface.co/facebook/mms-tts-zyb`
- Transformers MMS docs: `https://huggingface.co/docs/transformers/en/model_doc/mms`
- BibleBrain Core Concepts: `https://www.faithcomesbyhearing.com/bible-brain/core-concepts`
- ZhuangBench GitHub: `https://github.com/luciusssss/ZhuangBench`
- OpenVoice GitHub: `https://github.com/myshell-ai/OpenVoice`
- WeNet GitHub: `https://github.com/wenet-e2e/wenet`

---

## 15. 最终结论

新版 pipeline 更稳：

```text
V1 先做真实朗读 ASR 数据：
  获取/复现 MMS-lab-zyb 风格数据 -> 对齐切分 -> WeNet baseline

V2 再做合成增强：
  ZhuangBench -> MMS-TTS-zyb -> OpenVoice V2 -> augmentation -> mixed training
```

这样做的最大好处是：

- 第一版模型不依赖 synthetic-only；
- 可以先知道 `zyb` 真人朗读数据的 ASR 上限；
- 第二版合成数据是否有效，可以通过和 V1 baseline 对比判断；
- 后续如果加入真实目标域数据，也能自然接到 V1/V2 框架里。

