"""Audio pipeline:

microphone/file audio -> ASR -> normalization -> learned English->ASL model inference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    DEFAULT_ASR_COMPUTE_TYPE,
    DEFAULT_ASR_DEVICE,
    DEFAULT_ASR_MODEL_SIZE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_RECORD_SECONDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASL Speech Therapist audio pipeline: audio -> ASR -> learned English->ASL model"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--mic", action="store_true", help="Record audio from microphone.")
    input_group.add_argument("--audio_file", type=str, help="Path to an existing WAV file.")

    parser.add_argument("--duration", type=float, default=DEFAULT_RECORD_SECONDS)
    parser.add_argument("--model_size", type=str, default=DEFAULT_ASR_MODEL_SIZE)
    parser.add_argument("--asr_device", type=str, default=DEFAULT_ASR_DEVICE)
    parser.add_argument("--compute_type", type=str, default=DEFAULT_ASR_COMPUTE_TYPE)

    parser.add_argument(
        "--checkpoint",
        default=str(Path(DEFAULT_CHECKPOINT_DIR) / "best_model.pt"),
        help="Path to trained English->ASL checkpoint",
    )
    parser.add_argument("--device", default="cpu", help="Model inference device: cpu or cuda")
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--keep_fillers", action="store_true")
    parser.add_argument(
        "--use_fallback",
        action="store_true",
        help="Use debug fallback rules instead of learned model",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include raw token/id generation details in output.",
    )

    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> dict:
    from src.asl.fallback_rules import fallback_text_to_gloss
    from src.audio.asr import transcribe_audio
    from src.audio.preprocess_audio import preprocess_audio_to_mono16k
    #from src.audio.record_audio import record_from_microphone
    from src.models.inference import load_inference_bundle, predict_gloss
    from src.nlp.normalize_text import normalize_text

    if args.mic:
        from src.audio.record_audio import record_from_microphone
        audio_path = record_from_microphone()
    else:
        audio_path = args.audio_file

    processed_audio_path = preprocess_audio_to_mono16k(audio_path)

    asr_result = transcribe_audio(
        processed_audio_path,
        model_size=args.model_size,
        device=args.asr_device,
        compute_type=args.compute_type,
    )

    normalization_result = normalize_text(
        asr_result["raw_transcript"],
        remove_fillers=not args.keep_fillers,
    )

    if args.use_fallback:
        fallback = fallback_text_to_gloss(normalization_result["tokens"])
        return {
            "audio_path": audio_path,
            "processed_audio_path": processed_audio_path,
            "raw_transcript": asr_result["raw_transcript"],
            "language": asr_result["language"],
            "confidence": asr_result["confidence"],
            "clean_text": normalization_result["clean_text"],
            "predicted_gloss_tokens": fallback["predicted_gloss_tokens"],
            "predicted_gloss_text": fallback["predicted_gloss_text"],
            "model_name": "fallback_rules",
            "used_fallback": True,
            "empty_after_postprocess": len(fallback["predicted_gloss_tokens"]) == 0,
        }

    bundle = load_inference_bundle(args.checkpoint, device=args.device)
    prediction = predict_gloss(
        normalization_result["clean_text"],
        bundle=bundle,
        device=args.device,
        max_len=args.max_len,
        debug=args.debug,
    )

    output = {
        "audio_path": audio_path,
        "processed_audio_path": processed_audio_path,
        "raw_transcript": asr_result["raw_transcript"],
        "language": asr_result["language"],
        "confidence": asr_result["confidence"],
    }
    output.update(prediction.to_dict())
    return output


def main() -> None:
    args = parse_args()

    try:
        output = run_pipeline(args)
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
