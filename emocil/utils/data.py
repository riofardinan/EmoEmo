"""Datasets, label spaces and annotation masks for the RP1 and RP2 pilots.

Pair `xedgo` (RP1): two datasets, each with its own inventory:
  go   GoEmotions, 27 emotions + neutral, official train/dev/test.
  xed  XED English, 8 Plutchik emotions + neutral. No official split: an
       80/10/10 split by SHA-256 of the normalised text (NFKC, casefold,
       whitespace collapsed), so normalised duplicates share a partition.
       Öhman & Koufakou (2026) describe the same kind of split; their exact
       normalisation is not known, so the partitions need not coincide.

A label space lays both inventories out on one output head:
  separate  37 outputs: go's 28 then xed's 9. Same-named labels stay apart.
  merge     30 outputs: go's 28, then xed's anticipation and trust. xed's
            anger, disgust, fear, joy, sadness, surprise and neutral share the
            go output of the same name.

Pair `ekfine` (RP2): two "datasets" cut from GoEmotions alone, so the text
domain is identical. GoEmotions train is halved by SHA-256 of the normalised
text: half 0 is `ekman`, labelled at the Ekman level (anger, disgust, fear,
joy, sadness, surprise, neutral), each Ekman label being the OR of its fine
members in data/goemotions/ekman_mapping.json; half 1 is `fine`, with the 28
GoEmotions labels. Both are evaluated on the full GoEmotions dev and test sets,
at their own level. All seven Ekman names also exist among the fine labels:
  separate  35 outputs: fine's 28 then ekman's 7.
  merge     28 outputs: every Ekman label shares the fine output of its name.
Ekman disgust and neutral have the same scope as their fine namesakes; anger
covers 3 fine labels, fear 2, joy 12, sadness 5, surprise 4.

Every row carries a mask over the union: a column is annotated (mask 1) only if
it belongs to the row's own dataset. Labels of the other inventory are unknown,
not negative, and are never supervised.
"""

import hashlib
import os
import re
import unicodedata

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # emocil/
DATA = os.path.join(ROOT, "data")

XED_LABELS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness",
              "surprise", "trust", "neutral"]
EKMAN_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
BRIGHTER_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
# BRIGHTER English leaves disgust unannotated (empty in every split), so its
# inventory has five labels and disgust is masked, never negative, for its rows.
BRIGHTER_INVENTORY = {"b_eng": [l for l in BRIGHTER_LABELS if l != "disgust"],
                      "b_hin": BRIGHTER_LABELS, "b_sun": BRIGHTER_LABELS}
SHARED_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
# pair -> (base dataset laid out first on the union head, the other dataset)
PAIRS = {"xedgo": ("go", "xed"), "ekfine": ("fine", "ekman"),
         "enghin": ("b_hin", "b_eng"), "engsun": ("b_sun", "b_eng")}


def go_labels():
    with open(os.path.join(DATA, "goemotions", "emotions.txt"), encoding="utf-8") as f:
        labels = [l.strip() for l in f if l.strip()]
    assert len(labels) == 28
    return labels


def load_go(split):
    """Read a GoEmotions .tsv exactly as ../bertgo/data.py does (pandas, with
    CSV quote handling: texts containing quotes are quoted and escaped)."""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA, "goemotions", f"{split}.tsv"), sep="\t",
                     encoding="utf-8", header=None, names=["text", "labels", "id"],
                     dtype={"text": str})
    texts = [t if isinstance(t, str) else "" for t in df["text"].tolist()]
    y = np.zeros((len(texts), 28), dtype=np.float32)
    for i, raw in enumerate(df["labels"].tolist()):
        for l in str(raw).split(","):
            y[i, int(l)] = 1.0
    return texts, y


def load_xed_all():
    """All English XED texts with 9-dim labels. Texts that appear both in the
    emotion file and the neutral file are dropped; duplicates are kept once."""
    rows = {}
    with open(os.path.join(DATA, "xed", "en-annotated.tsv"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            text, labs = line.split("\t")
            rows[text] = {int(x) for x in labs.split(",") if x.strip()}
    neutral = set()
    with open(os.path.join(DATA, "xed", "neu_en.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                neutral.add(line.split("\t", 1)[1])
    clash = neutral & set(rows)
    texts, y = [], []
    for text, labs in rows.items():
        if text in clash:
            continue
        v = np.zeros(9, dtype=np.float32)
        for l in labs:                          # 1..8 = anger..trust
            v[l - 1] = 1.0
        texts.append(text); y.append(v)
    for text in sorted(neutral - clash):
        v = np.zeros(9, dtype=np.float32); v[8] = 1.0
        texts.append(text); y.append(v)
    return texts, np.stack(y)


def _normalise(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def xed_partition(text):
    h = int(hashlib.sha256(_normalise(text).encode("utf-8")).hexdigest(), 16) % 10
    return "train" if h < 8 else ("dev" if h == 8 else "test")


def load_xed(split):
    texts, y = load_xed_all()
    keep = [i for i, t in enumerate(texts) if xed_partition(t) == split]
    return [texts[i] for i in keep], y[keep]


def go_half(text):
    return int(hashlib.sha256(_normalise(text).encode("utf-8")).hexdigest(), 16) % 2


def to_ekman(y_fine):
    """Ekman-level labels from the 28 fine labels via the official grouping."""
    import json
    go = go_labels()
    with open(os.path.join(DATA, "goemotions", "ekman_mapping.json"), encoding="utf-8") as f:
        groups = json.load(f)
    y = np.zeros((len(y_fine), len(EKMAN_LABELS)), dtype=np.float32)
    for j, e in enumerate(EKMAN_LABELS):
        members = groups[e] if e != "neutral" else ["neutral"]
        y[:, j] = y_fine[:, [go.index(m) for m in members]].max(1)
    return y


def load_level(level, split):
    texts, y = load_go(split)
    if split == "train":
        keep = [i for i, t in enumerate(texts) if go_half(t) == (0 if level == "ekman" else 1)]
        texts, y = [texts[i] for i in keep], y[keep]
    return (texts, to_ekman(y)) if level == "ekman" else (texts, y)


def load_brighter(dataset, split):
    """BRIGHTER (Muhammad et al., 2025), data/brighter/<lang>_<split>.tsv, converted
    from the CC-BY-4.0 parquet release; unannotated labels are empty cells."""
    import pandas as pd
    lang = dataset[2:]
    df = pd.read_csv(os.path.join(DATA, "brighter", f"{lang}_{split}.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    labs = BRIGHTER_INVENTORY[dataset]
    assert all((df[l] != "").all() for l in labs), f"unannotated cells in {dataset} {split}"
    return df["text"].tolist(), df[labs].astype(int).to_numpy(dtype=np.float32)


def load(dataset, split):
    if dataset == "go":
        return load_go(split)
    if dataset == "xed":
        return load_xed(split)
    if dataset.startswith("b_"):
        return load_brighter(dataset, split)
    return load_level(dataset, split)


def labels_of(dataset):
    if dataset.startswith("b_"):
        return BRIGHTER_INVENTORY[dataset]
    return {"go": go_labels(), "fine": go_labels(), "xed": XED_LABELS, "ekman": EKMAN_LABELS}[dataset]


def shared_emotions(pair):
    """The six shared emotion names that both datasets of a pair annotate."""
    b, o = PAIRS[pair]
    return [e for e in SHARED_EMOTIONS if e in labels_of(b) and e in labels_of(o)]


class LabelSpace:
    """Where each dataset's labels sit on the union head. For pair xedgo this
    is exactly the RP1 layout (go first, then xed)."""

    def __init__(self, space, pair="xedgo"):
        b, o = PAIRS[pair]
        base, other = labels_of(b), labels_of(o)
        nb = len(base)
        if space == "separate":
            self.names = [f"{b}:{l}" for l in base] + [f"{o}:{l}" for l in other]
            self.cols = {b: list(range(nb)), o: list(range(nb, nb + len(other)))}
        elif space == "merge":
            extra = [l for l in other if l not in base]
            self.names = list(base) + extra
            self.cols = {b: list(range(nb)),
                         o: [base.index(l) if l in base else nb + extra.index(l) for l in other]}
        else:
            raise ValueError(space)
        self.space = space
        self.size = len(self.names)

    def native(self, dataset):
        """Stage-1 layout: the dataset's own labels only, in its own order."""
        return list(range(len(labels_of(dataset))))


class Encoded:
    """Tokenised rows with union-layout targets and annotation masks."""

    def __init__(self, texts, y_native, cols, size, tokenizer, max_len=50):
        enc = tokenizer(texts, truncation=True, max_length=max_len)
        self.ids, self.types = enc["input_ids"], enc["token_type_ids"]
        n = len(texts)
        self.y = np.zeros((n, size), dtype=np.float32)
        self.m = np.zeros((n, size), dtype=np.float32)
        self.y[:, cols] = y_native
        self.m[:, cols] = 1.0

    def __len__(self):
        return len(self.ids)

    def subset(self, idx):
        out = Encoded.__new__(Encoded)
        out.ids = [self.ids[i] for i in idx]; out.types = [self.types[i] for i in idx]
        out.y, out.m = self.y[idx], self.m[idx]
        return out

    @staticmethod
    def concat(parts):
        out = Encoded.__new__(Encoded)
        out.ids = [x for p in parts for x in p.ids]
        out.types = [x for p in parts for x in p.types]
        out.y = np.concatenate([p.y for p in parts]); out.m = np.concatenate([p.m for p in parts])
        return out


def batches(enc, batch_size, shuffle, rng=None, drop_last=False):
    """Yields padded tensors (input_ids, attention_mask, token_type_ids, y, m, idx)."""
    order = rng.permutation(len(enc)) if shuffle else np.arange(len(enc))
    stop = len(order) - (len(order) % batch_size if drop_last else 0)
    for s in range(0, stop, batch_size):
        idx = order[s:s + batch_size]
        if len(idx) == 0:
            break
        L = max(len(enc.ids[i]) for i in idx)
        ids = torch.zeros((len(idx), L), dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        typ = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            n = len(enc.ids[i])
            ids[r, :n] = torch.tensor(enc.ids[i]); att[r, :n] = 1
            typ[r, :n] = torch.tensor(enc.types[i])
        yield ids, att, typ, torch.from_numpy(enc.y[idx]), torch.from_numpy(enc.m[idx]), idx
