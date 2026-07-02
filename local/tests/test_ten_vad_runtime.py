#!/usr/bin/env python3
from vad.ten_vad_runtime import get_speech_timestamps


def test_get_speech_timestamps_merges_short_silence():
    probabilities = [0.0] * 5 + [0.8] * 20 + [0.1] * 3 + [0.8] * 20 + [0.0] * 20
    segments = get_speech_timestamps(
        probabilities,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=100,
        window_size_samples=256,
    )
    assert len(segments) == 1
    assert segments[0][0] == 0.08


def test_get_speech_timestamps_drops_too_short_speech():
    probabilities = [0.0] * 5 + [0.8] * 3 + [0.0] * 20
    segments = get_speech_timestamps(
        probabilities,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=100,
        window_size_samples=256,
    )
    assert segments == []


if __name__ == "__main__":
    test_get_speech_timestamps_merges_short_silence()
    test_get_speech_timestamps_drops_too_short_speech()
    print("ok")
