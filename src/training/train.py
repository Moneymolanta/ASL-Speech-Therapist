"""Training entry point for English->ASL gloss seq2seq model.

Improvements over the original baseline:
- Learning rate scheduling (cosine annealing with warmup)
- Gradient clipping to prevent exploding gradients
- Label smoothing for better generalization
- BLEU score evaluation during validation
- Scaled-up default hyperparameters for real dataset training
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_TOY_DATASET_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train English->ASL gloss transformer.")
    parser.add_argument("--dataset", default=DEFAULT_TOY_DATASET_PATH, help="Path to paired dataset file")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default=DEFAULT_CHECKPOINT_DIR)

    # Scaled-up model defaults for real dataset.
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--tiny_model",
        action="store_true",
        help="Force tiny toy-data-friendly architecture defaults.",
    )

    # Training improvements
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor.")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs for LR scheduling.")

    parser.add_argument("--max_src_len", type=int, default=64)
    parser.add_argument("--max_tgt_len", type=int, default=64)

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="If set, truncate training set to first N samples (debug/overfit mode).",
    )
    parser.add_argument(
        "--no_val",
        action="store_true",
        help="Disable validation loop and checkpoint on train loss.",
    )
    return parser.parse_args()


def split_records(records: list[dict[str, str]], val_split: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Randomly split records into train and validation sets."""
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")

    shuffled = records[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    split_idx = max(1, int(len(shuffled) * (1 - val_split)))
    train_records = shuffled[:split_idx]
    val_records = shuffled[split_idx:] or shuffled[:1]
    return train_records, val_records


def maybe_apply_tiny_model(args: argparse.Namespace) -> None:
    """Apply tiny model config if explicitly requested."""
    if not args.tiny_model:
        return
    args.d_model = 64
    args.nhead = 2
    args.num_encoder_layers = 1
    args.num_decoder_layers = 1
    args.dim_feedforward = 128


def main() -> None:
    args = parse_args()
    maybe_apply_tiny_model(args)

    import math

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from src.data.collate import TranslationCollator
    from src.data.dataset import EnglishASLDataset, load_paired_records
    from src.data.preprocess_dataset import preprocess_records
    from src.models.english_to_asl_model import EnglishToASLTransformer
    from src.models.tokenizer_utils import SimpleWhitespaceTokenizer, build_vocabs
    from src.training.evaluate import evaluate_model
    from src.training.losses import seq2seq_cross_entropy
    from src.training.metrics import compute_bleu
    from src.utils.seed import set_seed

    set_seed(args.seed)

    records = load_paired_records(args.dataset)
    records = preprocess_records(records)

    if args.no_val:
        train_records = records
        val_records: list[dict[str, str]] = []
    else:
        train_records, val_records = split_records(records, val_split=args.val_split, seed=args.seed)

    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("max_train_samples must be > 0")
        train_records = train_records[: args.max_train_samples]

    if not train_records:
        raise ValueError("Training set is empty after filtering/truncation.")

    print(f"Dataset: {len(train_records)} train, {len(val_records)} val samples")

    src_tokenizer = SimpleWhitespaceTokenizer(lowercase=True)
    tgt_tokenizer = SimpleWhitespaceTokenizer(lowercase=False)

    # Build vocabs using training split only.
    src_vocab, tgt_vocab = build_vocabs(train_records, src_tokenizer, tgt_tokenizer)
    print(f"Vocabulary: {len(src_vocab)} source tokens, {len(tgt_vocab)} target tokens")

    train_dataset = EnglishASLDataset(
        train_records,
        src_tokenizer,
        tgt_tokenizer,
        src_vocab,
        tgt_vocab,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
    )

    collator = TranslationCollator(pad_idx=tgt_vocab.pad_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    if args.no_val:
        val_loader = None
    else:
        val_dataset = EnglishASLDataset(
            val_records,
            src_tokenizer,
            tgt_tokenizer,
            src_vocab,
            tgt_vocab,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = EnglishToASLTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_idx=src_vocab.pad_idx,
        tgt_pad_idx=tgt_vocab.pad_idx,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # Cosine annealing with warmup
    def lr_lambda(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Label smoothing cross-entropy
    label_smooth_loss = nn.CrossEntropyLoss(
        ignore_index=tgt_vocab.pad_idx,
        label_smoothing=args.label_smoothing,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "best_model.pt"

    best_metric = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            src_ids = batch["src_ids"].to(args.device)
            tgt_input_ids = batch["tgt_input_ids"].to(args.device)
            tgt_output_ids = batch["tgt_output_ids"].to(args.device)

            optimizer.zero_grad()
            logits = model(src_ids=src_ids, tgt_input_ids=tgt_input_ids)

            # Use label smoothing loss
            vocab_size = logits.size(-1)
            loss = label_smooth_loss(logits.reshape(-1, vocab_size), tgt_output_ids.reshape(-1))
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)

        metrics: dict[str, float | int | str | bool | None] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "no_val": args.no_val,
            "train_size": len(train_records),
        }

        if args.no_val:
            monitor_value = train_loss
            metrics["monitor_metric"] = "train_loss"
            metrics["monitor_value"] = round(monitor_value, 4)
        else:
            assert val_loader is not None
            val_metrics = evaluate_model(model, val_loader, device=args.device, pad_idx=tgt_vocab.pad_idx)
            metrics.update({k: round(v, 4) if isinstance(v, float) else v for k, v in val_metrics.items()})
            monitor_value = val_metrics["val_loss"]
            metrics["monitor_metric"] = "val_loss"
            metrics["monitor_value"] = round(monitor_value, 4)

            # Compute BLEU on a sample of validation pairs every 10 epochs
            if epoch % 10 == 0 or epoch == args.epochs:
                bleu_refs = []
                bleu_hyps = []
                model.eval()
                with torch.no_grad():
                    for rec in val_records[:20]:
                        src_tokens = src_tokenizer.tokenize(rec["english"])
                        src_ids_list = [src_vocab.bos_idx] + src_vocab.encode(src_tokens) + [src_vocab.eos_idx]
                        src_tensor = torch.tensor([src_ids_list], dtype=torch.long, device=args.device)

                        gen = model.generate(src_tensor, tgt_vocab.bos_idx, tgt_vocab.eos_idx, max_len=32)
                        gen_ids = gen.squeeze(0).tolist()
                        raw_tokens = tgt_vocab.decode(gen_ids)
                        from src.asl.postprocess_gloss import clean_gloss_tokens
                        pred_tokens = clean_gloss_tokens(raw_tokens)

                        ref_tokens = tgt_tokenizer.tokenize(rec["gloss"])
                        ref_tokens = [t.upper() for t in ref_tokens]

                        bleu_refs.append(ref_tokens)
                        bleu_hyps.append(pred_tokens)

                from src.training.metrics import corpus_bleu
                bleu_result = corpus_bleu(bleu_refs, bleu_hyps)
                metrics["val_bleu"] = round(bleu_result["corpus_bleu"], 4)

        print(json.dumps(metrics))

        if monitor_value < best_metric:
            best_metric = monitor_value
            torch.save(
                {
                    "model_name": "english_to_asl_transformer",
                    "model_config": {
                        "d_model": args.d_model,
                        "nhead": args.nhead,
                        "num_encoder_layers": args.num_encoder_layers,
                        "num_decoder_layers": args.num_decoder_layers,
                        "dim_feedforward": args.dim_feedforward,
                        "dropout": args.dropout,
                        "src_pad_idx": src_vocab.pad_idx,
                        "tgt_pad_idx": tgt_vocab.pad_idx,
                    },
                    "src_vocab": src_vocab.to_dict(),
                    "tgt_vocab": tgt_vocab.to_dict(),
                    "src_tokenizer": {"lowercase": src_tokenizer.lowercase},
                    "tgt_tokenizer": {"lowercase": tgt_tokenizer.lowercase},
                    "model_state_dict": model.state_dict(),
                    "best_metric": best_metric,
                    "best_metric_name": "train_loss" if args.no_val else "val_loss",
                    "train_size": len(train_records),
                    "val_size": len(val_records),
                    "no_val": args.no_val,
                },
                checkpoint_path,
            )

    print(f"\nTraining complete. Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
