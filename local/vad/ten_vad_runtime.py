#!/usr/bin/env python3
from dataclasses import dataclass

import numpy as np
import soundfile as sf


@dataclass
class TenVadConfig:
    sampling_rate: int = 16000
    hop_size: int = 256
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 0


class TenVadAdapter:
    def __init__(self, config: TenVadConfig | None = None):
        from ten_vad import TenVad

        self.config = config or TenVadConfig()
        self.model = TenVad(self.config.hop_size, self.config.threshold)

    def detect(self, wav_path: str) -> dict:
        wav, sample_rate = sf.read(wav_path, dtype="int16")
        if sample_rate != self.config.sampling_rate:
            raise ValueError(f"Ten-VAD expects {self.config.sampling_rate} Hz audio, got {sample_rate} Hz")
        wav = _mono(wav)
        probabilities = self._frame_probabilities(wav)
        timestamps = get_speech_timestamps(
            probabilities,
            sampling_rate=self.config.sampling_rate,
            threshold=self.config.threshold,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            window_size_samples=self.config.hop_size,
            speech_pad_ms=self.config.speech_pad_ms,
            total_samples=len(wav),
        )
        frame_ms = round(self.config.hop_size / self.config.sampling_rate * 1000, 3)
        return {
            "timestamps": timestamps,
            "frame_speech_probs": {
                "frame_shift_ms": frame_ms,
                "frame_length_ms": frame_ms,
                "probs": probabilities,
            },
        }

    def _frame_probabilities(self, wav: np.ndarray) -> list[float]:
        frame_count = len(wav) // self.config.hop_size
        probabilities = []
        for i in range(frame_count):
            frame = wav[i * self.config.hop_size: (i + 1) * self.config.hop_size]
            probability, _ = self.model.process(np.ascontiguousarray(frame))
            probabilities.append(float(probability))
        return probabilities


def get_speech_timestamps(
    probabilities,
    sampling_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 256,
    speech_pad_ms: int = 0,
    total_samples: int | None = None,
) -> list[tuple[float, float]]:
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    speech_pad_samples = int(sampling_rate * speech_pad_ms / 1000)
    neg_threshold = threshold - 0.15

    triggered = False
    temp_end = 0
    current_speech = {}
    speeches = []

    for i, probability in enumerate(probabilities):
        current_sample = i * window_size_samples
        if probability >= threshold and temp_end:
            temp_end = 0

        if probability >= threshold and not triggered:
            triggered = True
            current_speech = {"start": current_sample}
            continue

        if probability < neg_threshold and triggered:
            if not temp_end:
                temp_end = current_sample
            if current_sample - temp_end >= min_silence_samples:
                current_speech["end"] = temp_end
                if current_speech["end"] - current_speech["start"] >= min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                triggered = False
                temp_end = 0

    if triggered and current_speech:
        current_speech["end"] = len(probabilities) * window_size_samples
        if current_speech["end"] - current_speech["start"] >= min_speech_samples:
            speeches.append(current_speech)

    max_samples = total_samples or len(probabilities) * window_size_samples
    segments = []
    for speech in speeches:
        start = max(0, int(speech["start"]) - speech_pad_samples)
        end = min(max_samples, int(speech["end"]) + speech_pad_samples)
        if start < end:
            segments.append((round(start / sampling_rate, 3), round(end / sampling_rate, 3)))
    return segments


def _mono(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return wav
    return wav.mean(axis=1).astype(np.int16)

