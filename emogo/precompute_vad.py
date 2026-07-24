"""Precompute per-sample affective (valence/arousal/dominance) vectors.

AESL distils relations from a second teacher living in affective space
(Section 3.6, Figure 3). Video27/Brain27 ship 14 human-rated appraisal
dimensions per stimulus and Audio28 ships 11; GoEmotions ships none, so the
signal has to be derived from the text. Two sources are supported, and they are
the two arms of an ablation:

  --source lexicon   Mean NRC-VAD score over the tokens found in the lexicon.
                     Cheap and transparent, but ignores composition (negation,
                     intensifiers) and leaves 0.4% of texts with a zero vector.

  --source emobank   A BERT regressor fine-tuned on EmoBank (10k sentences with
                     human V/A/D ratings at the *sentence* level), then applied
                     to GoEmotions. Same three dimensions, but every text gets a
                     vector and sentence composition is accounted for.

Neither uses the gold emotion labels: the future-missing setting assumes no
access to unseen classes, and a label-derived affective signal would smuggle
them back in.

Run once per source; training then just loads the cached .npy.

    python precompute_vad.py --source lexicon
    python precompute_vad.py --source emobank --emobank ~/Downloads/emobank.csv

Output: data/vad_<source>_{train,test}.npy, each [n_samples, 3] float32,
standardised — the paper standardises its affective ratings before RKD
(Appendix B.3).
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def read_texts(split: str):
    path = os.path.join(DATA, f"{split}.tsv")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    return [r[0] if r[0] else "" for r in rows]


def standardise(x: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per dimension, as the paper does before RKD."""
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return ((x - mu) / sd).astype(np.float32)


# --------------------------------------------------------------- lexicon arm

def build_lexicon(train_texts, test_texts, lexicon_path):
    from utils.vad import build_affective_matrix, load_vad_lexicon

    lexicon = load_vad_lexicon(lexicon_path)
    dims = (0, 1, 2)
    train = build_affective_matrix(train_texts, lexicon, dims)
    test = build_affective_matrix(test_texts, lexicon, dims)

    empty = int((np.abs(train).sum(axis=1) == 0).sum())
    print(f"lexicon: {empty}/{len(train)} training texts matched no lexicon "
          f"entry and get a zero vector")
    return train, test


# --------------------------------------------------------------- EmoBank arm

def build_emobank(train_texts, test_texts, emobank_path, model_name,
                  epochs, batch_size, lr, max_len, device, seed):
    """Fine-tune a BERT regressor on EmoBank, then score GoEmotions."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import BertModel, BertTokenizerFast

    torch.manual_seed(seed)
    np.random.seed(seed)

    rows = list(csv.DictReader(open(emobank_path, encoding="utf-8")))
    texts = [r["text"] for r in rows]
    vad = np.array([[float(r["V"]), float(r["A"]), float(r["D"])] for r in rows],
                   dtype=np.float32)
    # EmoBank is a 1-5 scale; centre it so the regression head starts sane.
    vad_mu, vad_sd = vad.mean(0, keepdims=True), vad.std(0, keepdims=True)
    y = (vad - vad_mu) / vad_sd
    is_train = np.array([r["split"] != "test" for r in rows])
    print(f"EmoBank: {len(texts)} sentences, {int(is_train.sum())} train / "
          f"{int((~is_train).sum())} test")

    tok = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)

    def encode(batch_texts):
        enc = tok(batch_texts, max_length=max_len, truncation=True,
                  padding="max_length", return_token_type_ids=True)
        return (torch.tensor(enc["input_ids"]),
                torch.tensor(enc["attention_mask"]),
                torch.tensor(enc["token_type_ids"]))

    ids, mask, types = encode(texts)
    targets = torch.from_numpy(y)

    tr = torch.nonzero(torch.from_numpy(is_train), as_tuple=True)[0]
    te = torch.nonzero(torch.from_numpy(~is_train), as_tuple=True)[0]

    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertModel.from_pretrained(model_name)
            self.drop = nn.Dropout(0.1)
            self.head = nn.Linear(self.bert.config.hidden_size, 3)

        def forward(self, a, b, c):
            pooled = self.bert(input_ids=a, attention_mask=b,
                               token_type_ids=c).pooler_output
            return self.head(self.drop(pooled))

    model = Regressor().to(device)
    loader = DataLoader(
        TensorDataset(ids[tr], mask[tr], types[tr], targets[tr]),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = len(loader) * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: max(0.0, 1.0 - s / max(1, total))
    )
    lossf = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        run = 0.0
        for a, b, c, t in loader:
            a, b, c, t = (v.to(device) for v in (a, b, c, t))
            loss = lossf(model(a, b, c), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
        print(f"  epoch {epoch + 1}/{epochs} — train MSE {run / len(loader):.4f}")

    # Held-out quality, so the proxy is not taken on faith.
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(te), 128):
            sl = te[i:i + 128]
            preds.append(model(ids[sl].to(device), mask[sl].to(device),
                               types[sl].to(device)).cpu())
        preds = torch.cat(preds).numpy()
    gold = y[te.numpy()]
    for d, name in enumerate("VAD"):
        r = np.corrcoef(preds[:, d], gold[:, d])[0, 1]
        print(f"  EmoBank held-out Pearson r ({name}) = {r:.3f}")

    def score(text_list):
        out = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(text_list), 128):
                a, b, c = encode(text_list[i:i + 128])
                out.append(model(a.to(device), b.to(device),
                                 c.to(device)).cpu().numpy())
        return np.concatenate(out)

    return score(train_texts), score(test_texts)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source", choices=["lexicon", "emobank"], required=True)
    ap.add_argument("--lexicon", default=os.path.join(DATA,
                                                      "NRC-VAD-Lexicon-v2.1.txt"))
    ap.add_argument("--emobank", default=os.path.expanduser(
        "~/Downloads/emobank.csv"))
    ap.add_argument("--model_name", default="bert-base-cased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1993)
    args = ap.parse_args()

    train_texts, test_texts = read_texts("train"), read_texts("test")
    print(f"GoEmotions: {len(train_texts)} train / {len(test_texts)} test")

    if args.source == "lexicon":
        train, test = build_lexicon(train_texts, test_texts, args.lexicon)
    else:
        import torch
        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA unavailable, falling back to CPU (this will be slow)")
            device = "cpu"
        if not os.path.isfile(args.emobank):
            raise FileNotFoundError(
                f"EmoBank not found at {args.emobank}. Download emobank.csv "
                f"from https://github.com/JULIELab/EmoBank and pass --emobank."
            )
        train, test = build_emobank(
            train_texts, test_texts, args.emobank, args.model_name,
            args.epochs, args.batch_size, args.lr, args.max_len, device,
            args.seed,
        )

    # Standardise using the training split's statistics only.
    mu, sd = train.mean(0, keepdims=True), train.std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    train = ((train - mu) / sd).astype(np.float32)
    test = ((test - mu) / sd).astype(np.float32)

    for split, arr in (("train", train), ("test", test)):
        path = os.path.join(DATA, f"vad_{args.source}_{split}.npy")
        np.save(path, arr)
        print(f"wrote {path}  {arr.shape}  "
              f"mean {arr.mean():+.3f}  std {arr.std():.3f}")


if __name__ == "__main__":
    main()
