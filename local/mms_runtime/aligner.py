import logging

import torch
import torchaudio.functional as F

from mms_runtime.align_utils import (
    get_spans,
    get_uroman_tokens,
    load_model_dict,
    merge_repeats,
    time_to_frame,
)
SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30

logger = logging.getLogger("semantic_asr.mms_runtime.aligner")


class MmsAlignmentFeasibilityError(RuntimeError):
    def __init__(self, reason: str, frame_count: int, target_count: int, repeat_count: int):
        self.reason = reason
        self.frame_count = frame_count
        self.target_count = target_count
        self.repeat_count = repeat_count
        super().__init__(
            f"{reason}: frames={frame_count}, target_chars={target_count}, repeats={repeat_count}"
        )


class MmsAligner:
    def __init__(self, model_path: str, device: str, uroman_path: str = "uroman/bin") -> None:
        logger.info("Loading MMS aligner model from %s", model_path)
        self.model, self.dictionary = load_model_dict(model_path)
        self.dictionary["<star>"] = len(self.dictionary)
        self.model.eval().to(device)
        self.uroman_path = uroman_path
        self.device = device

    def align(
        self,
        transcripts: list[str],
        waveform,
        sample_rate: int,
        names: list[str],
        use_star: bool,
        language: str,
        raw_transcripts: list[str],
        alignment_transcripts: list[str] | None = None,
    ) -> list[dict]:
        items, tokens = self._prepare_items_and_tokens(
            transcripts,
            names,
            language,
            raw_transcripts,
            alignment_transcripts,
        )
        if use_star:
            items, tokens = _insert_gap_stars(items, tokens)
        segments, stride = self.get_alignments(waveform, sample_rate, tokens)
        spans = get_spans(tokens, segments)
        align_segments = []
        for i, item in enumerate(items):
            if item.get("inserted_star"):
                continue
            span = spans[i]
            audio_start = round(span[0].start * stride / 1000, 3)
            audio_end = round(span[-1].end * stride / 1000, 3)
            align_segments.append({
                "start": round(audio_start, 3),
                "end": round(audio_end, 3),
                "duration": round(audio_end - audio_start, 3),
                "clean_text": item["transcript"],
                "text": item["raw_transcript"],
                "name": item["name"],
            })
        return align_segments

    def probe_star_gaps(
        self,
        transcripts: list[str],
        waveform,
        sample_rate: int,
        names: list[str],
        language: str,
        raw_transcripts: list[str],
        alignment_transcripts: list[str] | None = None,
    ) -> list[dict]:
        items, tokens = self._prepare_items_and_tokens(
            transcripts,
            names,
            language,
            raw_transcripts,
            alignment_transcripts,
        )
        if not items:
            return []

        expanded_items, expanded_tokens = _insert_gap_stars(items, tokens)
        segments, stride = self.get_alignments(waveform, sample_rate, expanded_tokens)
        spans = get_spans(expanded_tokens, segments)
        gaps = []
        for item, span in zip(expanded_items, spans):
            if not item.get("inserted_star"):
                continue
            audio_start = round(span[0].start * stride / 1000, 3)
            audio_end = round(span[-1].end * stride / 1000, 3)
            gaps.append({
                "kind": "gap_star",
                "start": round(audio_start, 3),
                "end": round(audio_end, 3),
                "duration": round(audio_end - audio_start, 3),
                "before_token_index": item.get("before_token_index"),
                "after_token_index": item.get("after_token_index"),
            })
        return gaps

    def _prepare_items_and_tokens(
        self,
        transcripts: list[str],
        names: list[str],
        language: str,
        raw_transcripts: list[str],
        alignment_transcripts: list[str] | None = None,
    ) -> tuple[list[dict], list[str]]:
        alignment_transcripts = alignment_transcripts or transcripts
        items = [
            {
                "transcript": transcript,
                "raw_transcript": raw_transcript,
                "alignment_transcript": alignment_transcript,
                "name": name,
                "token_index": index,
                "inserted_star": False,
            }
            for index, (transcript, raw_transcript, alignment_transcript, name) in enumerate(zip(
                transcripts,
                raw_transcripts,
                alignment_transcripts,
                names,
            ))
            if str(transcript).strip() and str(alignment_transcript).strip()
        ]
        tokens = self._uromanize_alignment_tokens(
            [str(item["alignment_transcript"]).strip().lower() for item in items],
            language,
        )
        tokens = _normalize_token_spaces(tokens)
        return _drop_empty_alignment_tokens(items, tokens)

    def _uromanize_alignment_tokens(self, alignment_transcripts: list[str], language: str) -> list[str]:
        uroman_inputs = [token for token in alignment_transcripts if token != "<star>"]
        uroman_tokens = get_uroman_tokens(uroman_inputs, self.uroman_path, language) if uroman_inputs else []
        next_uroman = iter(uroman_tokens)
        return ["<star>" if token == "<star>" else next(next_uroman) for token in alignment_transcripts]

    def get_alignments(self, waveform, sample_rate: int, tokens: list[str]):
        emissions, stride = self.generate_emissions(waveform, sample_rate)
        time_steps, _ = emissions.size()
        if any(token == "<star>" for token in tokens):
            emissions = torch.cat([emissions, torch.zeros(time_steps, 1).to(self.device)], dim=1)

        token_indices = [self.dictionary[c] for c in " ".join(tokens).split(" ") if c in self.dictionary]
        repeat_count = _count_consecutive_repeats(token_indices)
        if not token_indices:
            del emissions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise MmsAlignmentFeasibilityError("empty_target", time_steps, 0, 0)
        if len(token_indices) + repeat_count > time_steps:
            del emissions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise MmsAlignmentFeasibilityError(
                "ctc_target_too_long",
                time_steps,
                len(token_indices),
                repeat_count,
            )
        blank = self.dictionary["<blank>"]
        # torchaudio forced_align has unstable CUDA behavior for some scripts.
        # Keep model emissions on GPU, but run the lightweight DP alignment on CPU.
        alignment_emissions = emissions.to("cpu")
        targets = torch.tensor(token_indices, dtype=torch.int32)
        input_lengths = torch.tensor(alignment_emissions.shape[0]).unsqueeze(-1)
        target_lengths = torch.tensor(targets.shape[0]).unsqueeze(-1)
        path, _ = F.forced_align(
            alignment_emissions.unsqueeze(0),
            targets.unsqueeze(0),
            input_lengths,
            target_lengths,
            blank=blank,
        )
        path = path.squeeze().to("cpu").tolist()
        segments = merge_repeats(path, {v: k for k, v in self.dictionary.items()})
        del alignment_emissions
        del targets
        del emissions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return segments, stride

    def generate_emissions(self, waveform, sample_rate: int):
        waveform = _waveform_to_float_tensor(waveform)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T
        if sample_rate != SAMPLING_FREQ:
            waveform = F.resample(waveform, sample_rate, SAMPLING_FREQ)
            sample_rate = SAMPLING_FREQ
        total_duration = waveform.size(1) / sample_rate

        emissions_arr = []
        with torch.inference_mode():
            i = 0
            while i < total_duration:
                segment_start_time, segment_end_time = (i, i + EMISSION_INTERVAL)
                context = EMISSION_INTERVAL * 0.1
                input_start_time = max(segment_start_time - context, 0)
                input_end_time = min(segment_end_time + context, total_duration)
                waveform_split = waveform[
                    :,
                    int(sample_rate * input_start_time):int(sample_rate * input_end_time),
                ].to(self.device)

                model_outs, _ = self.model(waveform_split)
                emissions = model_outs[0]
                emission_start_frame = time_to_frame(segment_start_time)
                emission_end_frame = time_to_frame(segment_end_time)
                offset = time_to_frame(input_start_time)
                emissions = emissions[emission_start_frame - offset:emission_end_frame - offset, :]
                emissions_arr.append(emissions)
                i += EMISSION_INTERVAL

        emissions = torch.cat(emissions_arr, dim=0).squeeze()
        emissions = torch.log_softmax(emissions, dim=-1)
        stride = float(waveform.size(1) * 1000 / emissions.size(0) / sample_rate)
        del waveform
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return emissions, stride


def _count_consecutive_repeats(token_indices: list[int]) -> int:
    return sum(1 for previous, current in zip(token_indices, token_indices[1:]) if previous == current)


def _waveform_to_float_tensor(waveform) -> torch.Tensor:
    tensor = torch.as_tensor(waveform)
    if tensor.is_floating_point():
        return tensor.to(dtype=torch.float32)
    if tensor.dtype == torch.bool:
        return tensor.to(dtype=torch.float32)
    info = torch.iinfo(tensor.dtype)
    scale = float(max(abs(info.min), abs(info.max)))
    return tensor.to(dtype=torch.float32) / scale


def _normalize_token_spaces(tokens: list[str]) -> list[str]:
    return [
        "<star>" if token == "<star>" else " ".join(str(token).split())
        for token in tokens
    ]


def _drop_empty_alignment_tokens(items: list[dict], tokens: list[str]) -> tuple[list[dict], list[str]]:
    kept_items = []
    kept_tokens = []
    for item, token in zip(items, tokens):
        if token == "<star>" or str(token).strip():
            kept_items.append(item)
            kept_tokens.append(token)
    return kept_items, kept_tokens


def _insert_gap_stars(items: list[dict], tokens: list[str]) -> tuple[list[dict], list[str]]:
    expanded_items = []
    expanded_tokens = []
    for index, (item, token) in enumerate(zip(items, tokens)):
        if index == 0:
            expanded_items.append({
                "inserted_star": True,
                "before_token_index": None,
                "after_token_index": item.get("token_index"),
            })
            expanded_tokens.append("<star>")
        expanded_items.append(item)
        expanded_tokens.append(token)
        next_item = items[index + 1] if index + 1 < len(items) else None
        expanded_items.append({
            "inserted_star": True,
            "before_token_index": item.get("token_index"),
            "after_token_index": next_item.get("token_index") if next_item else None,
        })
        expanded_tokens.append("<star>")
    return expanded_items, expanded_tokens
