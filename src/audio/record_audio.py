"""Microphone recording utilities for the ASL Speech Therapy baseline."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

from src.utils.config import (
    AUDIO_OUTPUT_DIR,
    DEFAULT_CHANNELS,
    DEFAULT_DTYPE,
    DEFAULT_RECORD_SECONDS,
    DEFAULT_SAMPLE_RATE,
)


class MicrophoneRecordingError(RuntimeError):
    """Raised when microphone recording fails."""


def _wait_for_enter(stop_event: threading.Event) -> None:
    """Block until Enter is pressed, then set the stop event."""
    try:
        input("Press Enter to stop recording early...\n")
        stop_event.set()
    except EOFError:
        # Non-interactive environment: ignore and use duration-only mode.
        return


def record_from_microphone(
    duration: Optional[float] = DEFAULT_RECORD_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    output_dir: Path | str = AUDIO_OUTPUT_DIR,
    stop_on_enter: bool = True,
) -> str:
    """Record audio from the default microphone and save as WAV.

    Recording stops when either:
    1) configured duration is reached, or
    2) user presses Enter (when stop_on_enter=True and stdin is interactive)

    Args:
        duration: Maximum seconds to record. If None, requires Enter to stop.
        sample_rate: Recording sample rate in Hz.
        channels: Number of channels (1=mono).
        output_dir: Folder where WAV output will be written.
        stop_on_enter: If True, allow Enter key to stop early.

    Returns:
        Path to saved WAV file as a string.

    Raises:
        MicrophoneRecordingError: On device/recording/write failures.
        ValueError: If duration is invalid.
    """
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive or None")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    wav_path = output_path / f"mic_recording_{timestamp}.wav"

    frames = []
    stop_event = threading.Event()

    def callback(indata: np.ndarray, _frames: int, _time, status) -> None:
        if status:
            # status is best-effort diagnostic info from PortAudio
            print(f"[microphone warning] {status}")
        frames.append(indata.copy())
        if stop_event.is_set():
            raise sd.CallbackStop

    enter_thread = None
    if stop_on_enter:
        enter_thread = threading.Thread(target=_wait_for_enter, args=(stop_event,), daemon=True)
        enter_thread.start()

    print("Starting microphone recording...")
    print("If your OS prompts for microphone permission, click Allow.")

    start_time = time.monotonic()

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype=DEFAULT_DTYPE,
            callback=callback,
        ):
            while True:
                if stop_event.is_set():
                    break
                if duration is not None and (time.monotonic() - start_time) >= duration:
                    break
                time.sleep(0.05)

        stop_event.set()

        if not frames:
            raise MicrophoneRecordingError(
                "No audio frames were captured. Check microphone permissions or input device."
            )

        audio = np.concatenate(frames, axis=0)
        wav_write(str(wav_path), sample_rate, audio)

    except sd.PortAudioError as exc:
        raise MicrophoneRecordingError(
            "Could not access microphone device. Verify audio input permissions/device settings."
        ) from exc
    except Exception as exc:
        if isinstance(exc, MicrophoneRecordingError):
            raise
        raise MicrophoneRecordingError(f"Microphone recording failed: {exc}") from exc

    print(f"Saved recording to: {wav_path}")
    return str(wav_path)
