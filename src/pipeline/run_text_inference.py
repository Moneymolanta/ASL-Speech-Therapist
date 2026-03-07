"""CLI for English text -> learned ASL gloss inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DEFAULT_CHECKPOINT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learned English->ASL gloss inference.")
    parser.add_argument("--text", default=None, help="Input English text")
    parser.add_argument(
        "--text_file",
        default=None,
        help="Optional text file with one input phrase per line.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(Path(DEFAULT_CHECKPOINT_DIR) / "best_model.pt"),
        help="Path to trained checkpoint (.pt)",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--max_len", type=int, default=32)
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
    parser.add_argument(
        "--include_fallback_compare",
        action="store_true",
        help="When using model inference, also include fallback output for comparison.",
    )
    return parser.parse_args()


def _load_inputs(args: argparse.Namespace) -> list[str]:
    if args.text_file:
        lines: list[str] = []
        with Path(args.text_file).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    lines.append(line)
        if not lines:
            raise ValueError("text_file did not contain any non-empty lines.")
        return lines

    if args.text is None:
        raise ValueError("Provide --text or --text_file.")
    return [args.text]


def main() -> None:
    args = parse_args()

    from src.asl.fallback_rules import fallback_text_to_gloss
    from src.models.inference import load_inference_bundle, predict_gloss
    from src.nlp.normalize_text import normalize_text

    inputs = _load_inputs(args)

    if args.use_fallback:
        outputs = []
        for text in inputs:
            norm = normalize_text(text)
            fallback = fallback_text_to_gloss(norm["tokens"])
            outputs.append(
                {
                    "input_text": text,
                    "clean_text": norm["clean_text"],
                    "predicted_gloss_tokens": fallback["predicted_gloss_tokens"],
                    "predicted_gloss_text": fallback["predicted_gloss_text"],
                    "model_name": "fallback_rules",
                    "used_fallback": True,
                    "empty_after_postprocess": len(fallback["predicted_gloss_tokens"]) == 0,
                }
            )

        if args.text_file:
            print(json.dumps(outputs, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(outputs[0], indent=2, ensure_ascii=False))
        return

    bundle = load_inference_bundle(args.checkpoint, device=args.device)

    outputs = []
    for text in inputs:
        prediction = predict_gloss(
            text,
            bundle=bundle,
            device=args.device,
            max_len=args.max_len,
            debug=args.debug,
        )
        payload = prediction.to_dict()
        payload["input_text"] = text

        if args.include_fallback_compare:
            norm = normalize_text(text)
            payload["fallback_compare"] = fallback_text_to_gloss(norm["tokens"])

        outputs.append(payload)

    if args.text_file:
        print(json.dumps(outputs, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(outputs[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
