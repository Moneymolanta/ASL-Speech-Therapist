"""Inference helpers for the learned English->ASL model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.asl.postprocess_gloss import clean_gloss_tokens, to_gloss_text
from src.asl.schema import ASLPrediction
from src.models.english_to_asl_model import EnglishToASLTransformer
from src.models.tokenizer_utils import SimpleWhitespaceTokenizer, Vocab
from src.nlp.normalize_text import normalize_text


@dataclass
class InferenceBundle:
    """Loaded model artifacts used by inference scripts and pipelines."""

    model: EnglishToASLTransformer
    src_tokenizer: SimpleWhitespaceTokenizer
    tgt_tokenizer: SimpleWhitespaceTokenizer
    src_vocab: Vocab
    tgt_vocab: Vocab
    model_name: str


def load_inference_bundle(checkpoint_path: str, device: str = "cpu") -> InferenceBundle:
    """Load model/vocabs/tokenizers from a saved training checkpoint."""
    payload = torch.load(checkpoint_path, map_location=device)

    model_config = payload["model_config"]
    src_vocab = Vocab.from_dict(payload["src_vocab"])
    tgt_vocab = Vocab.from_dict(payload["tgt_vocab"])

    src_tokenizer = SimpleWhitespaceTokenizer(**payload["src_tokenizer"])
    tgt_tokenizer = SimpleWhitespaceTokenizer(**payload["tgt_tokenizer"])

    # Backward compatible with checkpoints saved before src/tgt pad split.
    src_pad_idx = model_config.get("src_pad_idx", src_vocab.pad_idx)
    tgt_pad_idx = model_config.get("tgt_pad_idx", tgt_vocab.pad_idx)

    model = EnglishToASLTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_encoder_layers=model_config["num_encoder_layers"],
        num_decoder_layers=model_config["num_decoder_layers"],
        dim_feedforward=model_config["dim_feedforward"],
        dropout=model_config["dropout"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    model_name = payload.get("model_name", "english_to_asl_transformer")

    return InferenceBundle(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        model_name=model_name,
    )


def predict_gloss(
    text: str,
    bundle: InferenceBundle,
    device: str = "cpu",
    max_len: int = 32,
    debug: bool = False,
) -> ASLPrediction:
    """Run end-to-end text inference using the learned translation model."""
    normalized = normalize_text(text)
    clean_text = normalized["clean_text"]

    src_tokens = bundle.src_tokenizer.tokenize(clean_text)
    src_ids = [bundle.src_vocab.bos_idx]
    src_ids += bundle.src_vocab.encode(src_tokens)
    src_ids += [bundle.src_vocab.eos_idx]

    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    generated = bundle.model.generate(
        src_ids=src_tensor,
        bos_idx=bundle.tgt_vocab.bos_idx,
        eos_idx=bundle.tgt_vocab.eos_idx,
        max_len=max_len,
    )

    generated_ids = generated.squeeze(0).tolist()
    raw_tokens = bundle.tgt_vocab.decode(generated_ids)
    gloss_tokens = clean_gloss_tokens(raw_tokens)
    empty_after_postprocess = len(gloss_tokens) == 0

    debug_info = None
    if debug or empty_after_postprocess:
        debug_info = {
            "normalized_input_text": clean_text,
            "source_tokens": src_tokens,
            "source_ids": src_ids,
            "raw_generated_ids": generated_ids,
            "raw_decoded_tokens": raw_tokens,
            "cleaned_gloss_tokens": gloss_tokens,
        }

    return ASLPrediction(
        clean_text=clean_text,
        predicted_gloss_tokens=gloss_tokens,
        predicted_gloss_text=to_gloss_text(gloss_tokens),
        model_name=bundle.model_name,
        used_fallback=False,
        empty_after_postprocess=empty_after_postprocess,
        debug_info=debug_info,
    )
