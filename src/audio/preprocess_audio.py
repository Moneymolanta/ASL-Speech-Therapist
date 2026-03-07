"""Audio preprocessing for ASR compatibility.

This baseline converts any input WAV to mono 16kHz WAV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write
from scipy.signal import resample_poly

from src.utils.config import DEFAULT_SAMPLE_RATE


class AudioPreprocessError(RuntimeError):
    """Raised when audio preprocessing fails."""


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multichannel audio to mono by averaging channels."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def _to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert audio to float32 in range [-1, 1] when possible."""
    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float32)

    if audio.dtype == np.int16:
        return (audio.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if audio.dtype == np.int32:
        return (audio.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    if audio.dtype == np.uint8:
        return ((audio.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)

    return audio.astype(np.float32)


def preprocess_audio_to_mono16k(audio_path: str, overwrite: bool = False) -> str:
    """Convert a WAV file to mono 16kHz WAV for ASR.

    Args:
        audio_path: Path to a WAV file.
        overwrite: If True, overwrite original file. Else write `<stem>_processed.wav`.

    Returns:
        Path to processed audio WAV.

    Raises:
        AudioPreprocessError: If processing fails.
    """
    input_path = Path(audio_path)
    if not input_path.exists():
        raise AudioPreprocessError(f"Audio file not found: {audio_path}")

    if input_path.suffix.lower() != ".wav":
        raise AudioPreprocessError("Baseline preprocessor currently supports WAV input only.")

    output_path = input_path if overwrite else input_path.with_name(f"{input_path.stem}_processed.wav")

    try:
        sample_rate, audio = wav_read(str(input_path))
        mono = _to_mono(audio)
        mono_f32 = _to_float32(mono)

        if sample_rate != DEFAULT_SAMPLE_RATE:
            mono_f32 = resample_poly(mono_f32, DEFAULT_SAMPLE_RATE, sample_rate)

        # Save as int16 WAV
        output_int16 = np.clip(mono_f32 * 32767.0, -32768.0, 32767.0).astype(np.int16)
        wav_write(str(output_path), DEFAULT_SAMPLE_RATE, output_int16)

        return str(output_path)
    except Exception as exc:
        raise AudioPreprocessError(f"Failed to preprocess audio: {exc}") from exc
