"""Speech-to-text transcription using faster-whisper."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from faster_whisper import WhisperModel

from src.utils.config import (
    DEFAULT_ASR_COMPUTE_TYPE,
    DEFAULT_ASR_DEVICE,
    DEFAULT_ASR_MODEL_SIZE,
)


class ASRError(RuntimeError):
    """Raised when transcription fails."""


@lru_cache(maxsize=4)
def _load_model(
    model_size: str = DEFAULT_ASR_MODEL_SIZE,
    device: str = DEFAULT_ASR_DEVICE,
    compute_type: str = DEFAULT_ASR_COMPUTE_TYPE,
) -> WhisperModel:
    """Load and cache a Whisper model."""
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio(
    audio_path: str,
    model_size: str = DEFAULT_ASR_MODEL_SIZE,
    device: str = DEFAULT_ASR_DEVICE,
    compute_type: str = DEFAULT_ASR_COMPUTE_TYPE,
) -> dict[str, Any]:
    """Transcribe an audio file to text.

    Returns:
        {
          "raw_transcript": str,
          "language": str,
          "confidence": None
        }
    """
    try:
        model = _load_model(model_size=model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(audio_path, vad_filter=True)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()

        return {
            "raw_transcript": transcript,
            "language": info.language if info and info.language else "unknown",
            "confidence": None,
        }
    except Exception as exc:
        raise ASRError(f"Transcription failed for '{audio_path}': {exc}") from exc
