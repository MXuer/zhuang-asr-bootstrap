import math
import os
import re
import tempfile
from dataclasses import dataclass

import torch
from torchaudio.models import wav2vec2_model

SPECIAL_ISOS_UROMAN = {
    "ara", "bel", "bul", "deu", "ell", "eng", "fas", "grc", "heb",
    "kaz", "kir", "lav", "lit", "mkd", "mkd2", "oss", "pnt", "pus",
    "rus", "srp", "srp2", "tur", "uig", "ukr", "yid",
}


@dataclass
class Segment:
    label: str
    start: int
    end: int


def normalize_uroman(text: str) -> str:
    text = text.lower()
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()


def get_uroman_tokens(norm_transcripts: list[str], uroman_root_dir: str, iso: str | None = None) -> list[str]:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as fin:
        with tempfile.NamedTemporaryFile(mode="r", encoding="utf-8") as fout:
            for text in norm_transcripts:
                fin.write(text + "\n")
            fin.flush()
            assert os.path.exists(f"{uroman_root_dir}/uroman.pl"), "uroman not found"
            cmd = f"perl {uroman_root_dir}/uroman.pl"
            if iso in SPECIAL_ISOS_UROMAN:
                cmd += f" -l {iso} "
            cmd += f" < {fin.name} > {fout.name}"
            os.system(cmd)
            outtexts = [" ".join(line.strip()) for line in fout]

    assert len(outtexts) == len(norm_transcripts)
    return [normalize_uroman(re.sub(r"\s+", " ", text).strip()) for text in outtexts]


def merge_repeats(path: list[int], idx_to_token_map: dict[int, str]) -> list[Segment]:
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1] == path[i2]:
            i2 += 1
        segments.append(Segment(idx_to_token_map[path[i1]], i1, i2 - 1))
        i1 = i2
    return segments


def time_to_frame(time_s: float) -> int:
    return int(time_s * 50)


def load_model_dict(model_path_name: str):
    if not os.path.exists(model_path_name):
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt",
            model_path_name,
        )
        assert os.path.exists(model_path_name)
    state_dict = torch.load(model_path_name, map_location="cpu")
    model = wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=[
            (512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2),
            (512, 3, 2), (512, 2, 2), (512, 2, 2),
        ],
        extractor_conv_bias=True,
        encoder_embed_dim=1024,
        encoder_projection_dropout=0.0,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=0.0,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=0.1,
        encoder_dropout=0.0,
        encoder_layer_norm_first=True,
        encoder_layer_drop=0.1,
        aux_num_out=31,
    )
    model.load_state_dict(state_dict)
    model.eval()

    dict_path_name = os.path.join(os.path.dirname(model_path_name), "ctc_alignment_mling_uroman_model.dict")
    if not os.path.exists(dict_path_name):
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/dictionary.txt",
            dict_path_name,
        )
        assert os.path.exists(dict_path_name)
    with open(dict_path_name, encoding="utf-8") as fin:
        dictionary = {line.strip(): i for i, line in enumerate(fin)}
    return model, dictionary


def get_spans(tokens: list[str], segments: list[Segment]) -> list[list[Segment]]:
    ltr_idx = 0
    tokens_idx = 0
    intervals = []
    start, end = (0, 0)
    sil = "<blank>"
    for seg_idx, seg in enumerate(segments):
        if tokens_idx == len(tokens):
            assert seg_idx == len(segments) - 1
            assert seg.label == "<blank>"
            continue
        cur_token = tokens[tokens_idx].split(" ")
        ltr = cur_token[ltr_idx]
        if seg.label == "<blank>":
            continue
        assert seg.label == ltr, f'===> {seg.label} <=> {ltr}'
        if ltr_idx == 0:
            start = seg_idx
        if ltr_idx == len(cur_token) - 1:
            ltr_idx = 0
            tokens_idx += 1
            intervals.append((start, seg_idx))
            while tokens_idx < len(tokens) and len(tokens[tokens_idx]) == 0:
                intervals.append((seg_idx, seg_idx))
                tokens_idx += 1
        else:
            ltr_idx += 1

    spans = []
    for idx, (start, end) in enumerate(intervals):
        span = segments[start:end + 1]
        if start > 0:
            prev_seg = segments[start - 1]
            if prev_seg.label == sil:
                pad_start = prev_seg.start if idx == 0 else int((prev_seg.start + prev_seg.end) / 2)
                span = [Segment(sil, pad_start, span[0].start)] + span
        if end + 1 < len(segments):
            next_seg = segments[end + 1]
            if next_seg.label == sil:
                pad_end = next_seg.end if idx == len(intervals) - 1 else math.floor((next_seg.start + next_seg.end) / 2)
                span = span + [Segment(sil, span[-1].end, pad_end)]
        spans.append(span)
    return spans
