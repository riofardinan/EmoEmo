"""Fine-tune BERT on GoEmotions and reproduce Table 4 of Demszky et al. (2020).

  python train.py --output_dir ./output/run1

The original is TensorFlow 1.x (bert_classifier.py) and will not run on a
modern Python. This is a PyTorch port that keeps every hyperparameter,
the exact optimizer, and the exact evaluation protocol. See config.py for a
field-by-field mapping back to the original flags.
"""

import argparse
import dataclasses
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from config import Config
from data import build_datasets, load_emotions
from metrics import compute_metrics, format_table
from model import BertForMultiLabelEmotion
from optimization import create_optimizer_and_schedule

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # huggingface_hub/httpx log every HTTP request at INFO, which buries the
    # training output.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@torch.no_grad()
def predict(model, loader, device):
    """Returns (probabilities, gold labels) for a whole split."""
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        logits, _ = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
        )
        all_probs.append(torch.sigmoid(logits).float().cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    for f in dataclasses.fields(Config):
        if f.type is bool or isinstance(getattr(cfg, f.name), bool):
            parser.add_argument(f"--{f.name}", type=lambda v: v.lower() == "true",
                                default=None)
        elif f.name == "exclude_from_weight_decay":
            continue
        else:
            parser.add_argument(f"--{f.name}", type=type(getattr(cfg, f.name)),
                                default=None)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Truncate the training set. For smoke tests only.")
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k != "max_train_samples"}
    cfg = dataclasses.replace(cfg, **overrides)
    return cfg, args.max_train_samples


def main():
    cfg, max_train_samples = parse_args()
    setup_logging(cfg.output_dir)
    set_seed(cfg.seed)

    logger.info("Config: %s", json.dumps(dataclasses.asdict(cfg), indent=2))

    device = torch.device(
        cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    if device.type != cfg.device:
        logger.warning("CUDA unavailable, falling back to %s", device)

    emotions = load_emotions(cfg.emotion_file)
    num_labels = len(emotions)
    logger.info("Loaded %d emotion labels: %s", num_labels, emotions)
    if num_labels != 28:
        logger.warning(
            "Expected 28 labels (27 emotions + neutral) to match the paper, got %d",
            num_labels,
        )

    tokenizer = BertTokenizerFast.from_pretrained(
        cfg.model_name, do_lower_case=cfg.do_lower_case
    )
    train_ds, dev_ds, test_ds = build_datasets(cfg, tokenizer, emotions)
    logger.info(
        "Split sizes — train %d, dev %d, test %d (paper: 43410 / 5426 / 5427)",
        len(train_ds), len(dev_ds), len(test_ds),
    )

    if max_train_samples is not None:
        train_ds = torch.utils.data.Subset(train_ds, range(max_train_samples))
        logger.warning("SMOKE TEST: training on %d examples only", len(train_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dev_loader = DataLoader(dev_ds, batch_size=cfg.eval_batch_size,
                            num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size,
                             num_workers=cfg.num_workers)

    model = BertForMultiLabelEmotion(
        cfg.model_name, num_labels, cfg.classifier_dropout, cfg.multilabel
    ).to(device)

    # The original computes steps as len(train) / batch_size * epochs, ignoring
    # the dropped remainder; we match that so warmup lands on the same step.
    num_train_steps = int(len(train_ds) / cfg.train_batch_size * cfg.num_train_epochs)
    optimizer, scheduler, num_warmup_steps = create_optimizer_and_schedule(
        model, cfg, num_train_steps
    )
    logger.info("Num training steps = %d (warmup %d)", num_train_steps, num_warmup_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and device.type == "cuda")

    global_step = 0
    num_epochs = int(np.ceil(cfg.num_train_epochs))
    for epoch in range(num_epochs):
        model.train()
        running_loss, seen, t0 = 0.0, 0, time.time()
        for batch in train_loader:
            if global_step >= num_train_steps:
                break
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                _, loss = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch["token_type_ids"].to(device),
                    labels=batch["labels"].to(device),
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # clip_by_global_norm(grads, 1.0) in create_optimizer.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            running_loss += loss.item()
            seen += 1
            if global_step % 200 == 0:
                logger.info(
                    "epoch %d step %d/%d loss %.4f lr %.2e (%.1f steps/s)",
                    epoch + 1, global_step, num_train_steps, running_loss / seen,
                    scheduler.get_last_lr()[0], seen / (time.time() - t0),
                )

        dev_probs, dev_true = predict(model, dev_loader, device)
        dev_res = compute_metrics(dev_probs, dev_true, emotions,
                                  cfg.eval_prob_threshold)
        logger.info(
            "== epoch %d — dev macro-F1 %.4f | micro-F1 %.4f | loss %.4f",
            epoch + 1, dev_res["macro_f1"], dev_res["micro_f1"],
            running_loss / max(1, seen),
        )
        if global_step >= num_train_steps:
            break

    # The paper evaluates once, on the test set, with the final model —
    # no checkpoint selection on dev.
    test_probs, test_true = predict(model, test_loader, device)
    test_res = compute_metrics(test_probs, test_true, emotions,
                               cfg.eval_prob_threshold)

    logger.info("\n%s", format_table(test_res, emotions))
    logger.info(
        "Paper Table 4 reference: macro P .40 / R .63 / F1 .46 (std .18/.24/.19)"
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "test_results.json"), "w") as f:
        json.dump(test_res, f, indent=2)
    np.save(os.path.join(cfg.output_dir, "test_probs.npy"), test_probs)
    with open(os.path.join(cfg.output_dir, "test_table4.txt"), "w") as f:
        f.write(format_table(test_res, emotions) + "\n")
    torch.save(
        {"model_state_dict": model.state_dict(),
         "config": dataclasses.asdict(cfg),
         "emotions": emotions},
        os.path.join(cfg.output_dir, "model.pt"),
    )
    logger.info("Wrote results and checkpoint to %s", cfg.output_dir)


if __name__ == "__main__":
    main()
