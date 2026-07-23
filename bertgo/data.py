"""GoEmotions data loading, mirroring DataProcessor in bert_classifier.py."""

import os
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _require(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            "The GoEmotions .tsv files and emotions.txt ship inside this "
            "folder under data/. If you copied only the .py files, copy data/ "
            "across too, or point --data_dir at wherever they live."
        )
    return path


def load_emotions(emotion_file: str) -> List[str]:
    """Reads the emotion vocabulary, one label per line.

    The shipped emotions.txt already ends with `neutral`, which is why the
    original code leaves --add_neutral at False. Returns 28 labels.
    """
    with open(_require(emotion_file), encoding="utf-8") as f:
        emotions = f.read().splitlines()
    return [e for e in emotions if e.strip()]


class GoEmotionsDataset(Dataset):
    """A .tsv split, tokenized once up front.

    The .tsv files have no header and three columns: text, a comma-separated
    list of emotion ids, and the comment id. Ids index into emotions.txt.
    """

    def __init__(self, path: str, tokenizer, num_labels: int, max_seq_length: int):
        df = pd.read_csv(
            _require(path),
            sep="\t",
            encoding="utf-8",
            header=None,
            names=["text", "labels", "id"],
            dtype={"text": str},
        )
        # The original processor guards against rare encoding errors that make
        # pandas hand back a float instead of a string.
        self.texts = [t if isinstance(t, str) else "" for t in df["text"].tolist()]
        self.ids = df["id"].tolist()

        self.labels = np.zeros((len(df), num_labels), dtype=np.float32)
        for i, raw in enumerate(df["labels"].tolist()):
            for idx in str(raw).split(","):
                self.labels[i, int(idx)] = 1.0

        # BERT truncates to max_seq_length - 2 wordpieces and adds [CLS]/[SEP],
        # which is exactly what truncation=True + max_length does here.
        encoded = tokenizer(
            self.texts,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
        )
        self.input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        self.token_type_ids = torch.tensor(encoded["token_type_ids"], dtype=torch.long)
        self.label_tensor = torch.from_numpy(self.labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "token_type_ids": self.token_type_ids[i],
            "labels": self.label_tensor[i],
        }


def build_datasets(cfg, tokenizer, emotions):
    """Returns (train, dev, test) datasets for the given config."""
    splits = {}
    for name, fname in (
        ("train", cfg.train_fname),
        ("dev", cfg.dev_fname),
        ("test", cfg.test_fname),
    ):
        path = os.path.join(cfg.data_dir, fname)
        splits[name] = GoEmotionsDataset(
            path, tokenizer, len(emotions), cfg.max_seq_length
        )
    return splits["train"], splits["dev"], splits["test"]
