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

Datasets available:

- **Toy dataset** (16 pairs): `data/examples/toy_asl_pairs.json` (for sanity checks only)
- **Expanded dataset** (396 pairs): `data/asl_gloss_pairs_v2.json` (recommended for training)

The expanded dataset covers diverse conversational phrases, questions, commands, and learning contexts with proper ASL grammar (topic-comment structure, WH-questions, copula/article deletion).

## How To Train

**Recommended: Train on expanded dataset with improved hyperparameters**

```bash
python src/training/train.py \
  --dataset data/asl_gloss_pairs_v2.json \
  --epochs 80 --batch_size 16 --lr 5e-4 \
  --d_model 128 --nhead 4 \
  --num_encoder_layers 2 --num_decoder_layers 2 \
  --dim_feedforward 256 --dropout 0.15 \
  --grad_clip 1.0 --label_smoothing 0.1 \
  --warmup_epochs 8 --val_split 0.15 --device cpu
```

This configuration:

- Uses 85/15 train/validation split on 396 pairs
- Larger model (128-dim embeddings, 2 encoder/decoder layers)
- Cosine annealing learning rate with 8-epoch warmup
- Gradient clipping (max norm 1.0) for stability
- Label smoothing (0.1) for better generalization
- Takes ~30-40 minutes on CPU, saves checkpoint to `checkpoints/best_model.pt`

**Toy Dataset (for quick sanity checks)**

```bash
python src/training/train.py \
  --dataset data/examples/toy_asl_pairs.json \
  --epochs 25 \
  --batch_size 4 \
  --device cpu
```

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

### Interactive Testing (Recommended)

Test the model interactively with any English phrase:

```bash
python test_examples.py --checkpoint checkpoints/best_model.pt --beam_width 5
```

This launches an interactive prompt:

```
📝 English text: where do you work
✓ Translation:
   ASL Gloss: YOU WORK WHERE
   Tokens: ['YOU', 'WORK', 'WHERE']

📝 English text: i love music
✓ Translation:
   ASL Gloss: I LOVE MUSIC
   Tokens: ['I', 'LOVE', 'MUSIC']
```

Type `exit` or `quit` to stop.

### Batch Testing (Predefined Categories)

Test on predefined phrase categories with accuracy metrics:

```bash
# All categories
python test_batch.py --checkpoint checkpoints/best_model.pt --beam_width 3

# Specific category (greetings, questions, emotions, learning, daily_activities, etc.)
python test_batch.py --checkpoint checkpoints/best_model.pt --category greetings

# Show all results (not just failures)
python test_batch.py --checkpoint checkpoints/best_model.pt --show_all
```

### Single Input

```bash
python src/pipeline/run_text_inference.py \
  --text "can you help me today" \
  --checkpoint checkpoints/best_model.pt \
  --device cpu
```

### With Beam Search

Test with different beam widths (higher = better quality, slower):

```bash
# Greedy decoding (fastest)
python src/pipeline/run_text_inference.py \
  --text "where is the library" \
  --checkpoint checkpoints/best_model.pt

# Beam width 3 (recommended)
python src/pipeline/run_text_inference.py \
  --text "where is the library" \
  --checkpoint checkpoints/best_model.pt --beam_width 3

# Beam width 5 (slowest, best quality)
python src/pipeline/run_text_inference.py \
  --text "where is the library" \
  --checkpoint checkpoints/best_model.pt --beam_width 5
```

### Full Evaluation

Run full evaluation on entire dataset with BLEU scores:

```bash
python src/training/evaluate_checkpoint.py \
  --checkpoint checkpoints/best_model.pt \
  --dataset data/asl_gloss_pairs_v2.json \
  --beam_width 3 \
  --show_examples 20
```

Output:

```
Corpus BLEU: 0.8674
Exact match accuracy: 326/396 (82.3%)
1-gram precision: 0.9335
2-gram precision: 0.8849
...
```

### Debug Mode

```bash
python src/pipeline/run_text_inference.py \
  --text "can you help me today" \
  --checkpoint checkpoints/best_model.pt \
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

## Model Performance

**Current Results (Expanded Dataset)**

- **Corpus BLEU**: 0.8674
- **Exact Match Accuracy**: 82.3% (326/396 examples)
- **1-gram Precision**: 0.9335
- **2-gram Precision**: 0.8849
- **Model Size**: 803K parameters
- **Training Time**: ~40 minutes on CPU (80 epochs)

The model successfully learns ASL grammatical patterns including:

- Topic-comment structure (time references first)
- WH-question fronting (interrogatives at end)
- Copula/article deletion
- Word reordering for ASL syntax

## Current Limitations

- Gloss text output only (not sign video/animation)
- No phoneme-level pronunciation scoring yet
- No computer-vision sign feedback yet
- Greedy + beam search decoding (no advanced techniques like length normalization tuning)
- Limited to ~400 training examples

## Future Extensions

1. **Data**: Integration with ASLG-PC12 corpus (~87K pairs) for large-scale training
2. **Modeling**: Subword tokenization (BPE), larger models, pretrained encoders
3. **Decoding**: Advanced beam search, diverse beam search, minimum risk training
4. **Integration**: Sign video generation, computer vision feedback, pronunciation scoring
5. **Representation**: Non-manual markers, spatial positioning, classifier expressions
6. **Personalization**: LoRA-based fine-tuning for individual user adaptation

## Notes

- Microphone permission prompts are handled by OS/browser/device settings through `sounddevice`.
- For notebook/Colab workflows, audio file mode is usually more reliable than direct mic capture.
- All pipeline scripts return JSON-serializable outputs for easier integration with future components.
