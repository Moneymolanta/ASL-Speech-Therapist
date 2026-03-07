# ASL Speech Therapist

## Project Shift: Baseline -> ML Translation Scaffold

The original baseline focused on:

`audio -> ASR -> normalization -> mostly rule-based gloss`

This refactor changes the core design to match the real research/engineering goal:

`audio -> ASR -> normalized English text -> learned English-to-ASL model -> structured ASL output`

The key architectural decision is that **English-to-ASL is now modeled as a learned sequence prediction task**, not a dictionary/rule system.

## New Architecture

```text
src/
  audio/
    record_audio.py
    preprocess_audio.py
    asr.py

  nlp/
    normalize_text.py
    text_to_gloss.py            # deprecated compatibility wrapper

  asl/
    schema.py
    postprocess_gloss.py
    fallback_rules.py           # debug/baseline only

  models/
    english_to_asl_model.py
    tokenizer_utils.py
    inference.py

  data/
    dataset.py
    preprocess_dataset.py
    collate.py

  training/
    train.py
    evaluate.py
    losses.py

  pipeline/
    run_audio_pipeline.py
    run_text_inference.py

  utils/
    config.py
    io.py
    seed.py

data/
  examples/
    toy_asl_pairs.json

requirements.txt
README.md
```

## Why This Design Changed

Rule-based gloss conversion is useful as a quick baseline, but it does not scale to real translation quality or richer ASL representations.

This refactor introduces:

- trainable seq2seq model scaffolding
- paired dataset loading pipeline
- training and validation loops with checkpoints
- model-based text and audio inference
- structured output schema for downstream modules

## Current Model Baseline

The default model is a **small Transformer encoder-decoder** (`EnglishToASLTransformer`) implemented with PyTorch.

Default toy-data hyperparameters are intentionally small:

- `d_model=64`
- `nhead=2`
- `num_encoder_layers=1`
- `num_decoder_layers=1`
- `dim_feedforward=128`

Input: normalized English token ids  
Output: ASL gloss token sequence

## Installation

Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Format

Paired records are expected in JSON/JSONL/CSV with fields:

```json
{
  "english": "can you help me today",
  "gloss": "TODAY YOU HELP ME"
}
```

A toy dataset is included at:

`data/examples/toy_asl_pairs.json`

## How To Train (Toy Dataset)

```bash
python src/training/train.py \
  --dataset data/examples/toy_asl_pairs.json \
  --epochs 25 \
  --batch_size 4 \
  --device cpu
```

This will:

- preprocess paired records
- split train/validation
- build source/target vocabularies
- train with teacher forcing
- evaluate each epoch
- save best checkpoint to `checkpoints/best_model.pt`

## Overfit Sanity-Check Mode

Use this to verify the model can memorize tiny data:

```bash
python src/training/train.py \
  --dataset data/examples/toy_asl_pairs.json \
  --epochs 200 \
  --batch_size 2 \
  --device cpu \
  --max_train_samples 2 \
  --no_val
```

Options:

- `--max_train_samples N`: truncate training split to first N examples.
- `--no_val`: skip validation completely and checkpoint by train loss.

## How To Run Text Inference

Single input:

```bash
python src/pipeline/run_text_inference.py \
  --text "can you help me today" \
  --checkpoint checkpoints/best_model.pt \
  --device cpu
```

Batch input from file (one phrase per line):

```bash
python src/pipeline/run_text_inference.py \
  --text_file data/examples/inference_inputs.txt \
  --checkpoint checkpoints/best_model.pt \
  --device cpu
```

Debug mode (inspect raw generation internals):

```bash
python src/pipeline/run_text_inference.py \
  --text "can you help me today" \
  --checkpoint checkpoints/best_model.pt \
  --device cpu \
  --debug
```

Debug JSON fields include:

- `normalized_input_text`
- `source_tokens`
- `source_ids`
- `raw_generated_ids`
- `raw_decoded_tokens`
- `cleaned_gloss_tokens`
- `empty_after_postprocess`

Compare learned model output vs fallback in one run:

```bash
python src/pipeline/run_text_inference.py \
  --text "can you help me today" \
  --checkpoint checkpoints/best_model.pt \
  --include_fallback_compare
```

## How To Run Audio Inference

Microphone mode:

```bash
python src/pipeline/run_audio_pipeline.py --mic --checkpoint checkpoints/best_model.pt
```

Audio file mode:

```bash
python src/pipeline/run_audio_pipeline.py --audio_file example.wav --checkpoint checkpoints/best_model.pt
```

Audio debug mode:

```bash
python src/pipeline/run_audio_pipeline.py \
  --audio_file example.wav \
  --checkpoint checkpoints/best_model.pt \
  --debug
```

Pipeline:

`mic/audio -> preprocess (mono 16kHz) -> Whisper ASR -> normalize -> learned model inference`

## Fallback Rules (Debug Only)

A fallback module exists in `src/asl/fallback_rules.py` for debugging and comparison.

It is **not** the main translation path. Use explicitly:

```bash
python src/pipeline/run_text_inference.py --text "hello" --use_fallback
```

## Debugging Workflow (Recommended)

1. Train on toy dataset and confirm checkpoint is produced.
2. Run inference on exact training phrases.
3. Use `--debug` and check whether model emits `<eos>` immediately.
4. If needed, run overfit mode with `--max_train_samples 2 --no_val`.
5. Compare learned output and fallback output using `--include_fallback_compare`.

## Current Limitations

- Toy dataset is tiny and only for sanity checks.
- Model quality is limited until trained on real paired English-ASL resources.
- Output target is gloss text, not sign video/animation.
- No phoneme-level pronunciation scoring yet.
- No computer-vision sign feedback yet.

## Future Extensions

1. Real dataset integration (WLASL/How2Sign-aligned gloss resources, cleaned parallel pairs)
2. Better translation modeling (larger transformer, pretrained encoders, beam search)
3. Sign retrieval / animation backend integration
4. Visual signing feedback via computer vision modules
5. Pronunciation and phoneme analysis modules connected to ASR front-end
6. Richer ASL representations beyond gloss (non-manual markers, timing, sign IDs)

## Notes

- Microphone permission prompts are handled by OS/browser/device settings through `sounddevice`.
- For notebook/Colab workflows, audio file mode is usually more reliable than direct mic capture.
- All pipeline scripts return JSON-serializable outputs for easier integration with future components.
