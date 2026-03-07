"""Central configuration for the ASL Speech Therapist ML-ready baseline."""

from __future__ import annotations

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_TOY_DATASET_PATH = PROJECT_ROOT / "data" / "examples" / "toy_asl_pairs.json"

# Audio defaults
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1
DEFAULT_RECORD_SECONDS = 5
DEFAULT_DTYPE = "int16"

# ASR defaults
DEFAULT_ASR_MODEL_SIZE = "base"
DEFAULT_ASR_DEVICE = "cpu"
DEFAULT_ASR_COMPUTE_TYPE = "int8"

# Text normalization defaults
DEFAULT_REMOVE_FILLERS = True
FILLER_WORDS = {
    "um",
    "uh",
    "like",
    "hmm",
    "erm",
}

# Debug fallback stopwords only (fallback is not main system).
GLOSS_STOPWORDS = {
    "a",
    "an",
    "the",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "as",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "because",
    "that",
    "this",
    "these",
    "those",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "may",
    "might",
    "must",
}
