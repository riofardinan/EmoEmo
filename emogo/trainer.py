"""Runs one incremental experiment and records the per-task metric curves.

Reporting follows EmoGrowth's tables: for each metric we log the value after
every task, then summarise as
  * Avg. Acc — the mean across all tasks (the "Avg. Acc" columns), and
  * Last Acc — the value after the final task, over all classes.
"""

import json
import logging
import os
import random
import time

import numpy as np
import torch
from transformers import BertTokenizerFast

from utils.config import Config, effective_config
from utils.data_manager import GoEmotionsDataManager
from utils.factory import get_model
from utils.metrics import METRIC_NAMES
from utils.vad import build_affective_matrix, load_vad_lexicon

logger = logging.getLogger(__name__)

# Methods that need the affective (VAD) teacher.
_NEEDS_AFFECTIVE = {"aesl", "clif"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(run_dir: str):
    os.makedirs(run_dir, exist_ok=True)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    # Deliberately not %(filename)s: methods inherit their training loop from
    # models/finetune.py, so the filename would name the wrong method. The
    # learner prints its own class name instead.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(run_dir, "run.log")),
            logging.StreamHandler(),
        ],
    )
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def train(cfg: Config):
    run_dir = cfg.run_dir
    setup_logging(run_dir)
    set_seed(cfg.seed)

    logger.info("Run: %s", cfg.run_name)
    # Only the settings this method actually reads — other methods' knobs would
    # be misleading noise here.
    logger.info("Config: %s", json.dumps(effective_config(cfg), indent=2))

    tokenizer = BertTokenizerFast.from_pretrained(
        cfg.model_name, do_lower_case=cfg.do_lower_case
    )
    data_manager = GoEmotionsDataManager(cfg, tokenizer)

    if cfg.method.lower() in _NEEDS_AFFECTIVE:
        lexicon = load_vad_lexicon(cfg.vad_lexicon_path)
        for split in ("train", "test"):
            data_manager.attach_affective(
                split,
                build_affective_matrix(
                    data_manager._splits[split]["texts"], lexicon, cfg.use_vad_dims
                ),
            )

    model = get_model(cfg.method, cfg)
    logger.info(
        "Method: %s -> %s.%s",
        cfg.method, type(model).__module__, type(model).__name__,
    )

    curves = {"map": []}
    curves.update({name: [] for name in METRIC_NAMES})

    for task in range(data_manager.nb_tasks):
        t0 = time.time()
        model.incremental_train(data_manager)
        mean_ap, metrics, outputs, labels = model.eval_task()
        model.after_task()

        curves["map"].append(float(mean_ap))
        for name, value in metrics:
            curves[name].append(value)

        logger.info(
            "== Task %d/%d done in %.1fs — mAP %.4f | maF1 %.4f | miF1 %.4f",
            task + 1, data_manager.nb_tasks, time.time() - t0,
            mean_ap, curves["macrof1"][-1], curves["microf1"][-1],
        )
        logger.info("mAP curve:    %s", [round(v, 4) for v in curves["map"]])
        logger.info("maF1 curve:   %s", curves["macrof1"])
        logger.info("miF1 curve:   %s", curves["microf1"])

        np.save(os.path.join(run_dir, f"task{task}_logits.npy"), outputs)
        np.save(os.path.join(run_dir, f"task{task}_labels.npy"), labels)

    summary = {
        "run_name": cfg.run_name,
        "config": effective_config(cfg),
        "class_order": data_manager.ordered_emotions,
        "task_sizes": [data_manager.get_task_size(t)
                       for t in range(data_manager.nb_tasks)],
        "curves": curves,
        "avg_acc": {k: float(np.mean(v)) for k, v in curves.items()},
        "last_acc": {k: float(v[-1]) for k, v in curves.items()},
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # A csv laid out like EmoGrowth's: one row per metric, one column per task,
    # with the across-task mean in the final column.
    names = ["map"] + METRIC_NAMES
    table = np.zeros((len(names), data_manager.nb_tasks + 1))
    for i, name in enumerate(names):
        table[i, : data_manager.nb_tasks] = curves[name]
        table[i, data_manager.nb_tasks] = np.mean(curves[name])
    np.savetxt(
        os.path.join(run_dir, "results.csv"), table, delimiter=",", fmt="%.4f",
        header=",".join([f"task{t}" for t in range(data_manager.nb_tasks)] + ["avg"]),
        comments="# rows: " + ",".join(names) + "\n",
    )

    logger.info(
        "FINAL — Avg mAP %.4f | Avg maF1 %.4f | Avg miF1 %.4f",
        summary["avg_acc"]["map"], summary["avg_acc"]["macrof1"],
        summary["avg_acc"]["microf1"],
    )
    logger.info(
        "FINAL — Last mAP %.4f | Last maF1 %.4f | Last miF1 %.4f",
        summary["last_acc"]["map"], summary["last_acc"]["macrof1"],
        summary["last_acc"]["microf1"],
    )
    logger.info("Results written to %s", run_dir)
    return summary
