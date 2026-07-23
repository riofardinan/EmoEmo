"""Per-sample affective (valence / arousal / dominance) vectors.

AESL's relation-based knowledge distillation uses a second teacher living in
the affective dimension space (Section 3.6, Figure 3). Video27 and Audio28 come
with human valence-arousal ratings per stimulus; GoEmotions does not, so we
derive them from the comment text with the NRC-VAD lexicon: look up every token,
average the entries that are found.

Deriving VAD from the gold emotion labels instead would be a mistake — the
future-missing setting assumes no access to unseen classes, and label-derived
VAD would smuggle them in. Text-derived VAD stays label-independent.
"""

import csv
import logging
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")


def load_vad_lexicon(path: str) -> Dict[str, Tuple[float, float, float]]:
    """NRC-VAD-Lexicon v2.1: tab-separated, header `term valence arousal dominance`."""
    lexicon = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                lexicon[row["term"].lower()] = (
                    float(row["valence"]),
                    float(row["arousal"]),
                    float(row["dominance"]),
                )
            except (KeyError, ValueError, TypeError):
                continue
    logger.info("Loaded %d VAD entries from %s", len(lexicon), path)
    return lexicon


def text_to_vad(text: str, lexicon, use_dims: Sequence[int]) -> np.ndarray:
    """Mean VAD over the tokens present in the lexicon; zeros if none match."""
    vectors = [
        [lexicon[tok][d] for d in use_dims]
        for tok in _TOKEN_RE.findall(text.lower())
        if tok in lexicon
    ]
    if not vectors:
        return np.zeros(len(use_dims), dtype=np.float32)
    return np.mean(np.asarray(vectors, dtype=np.float32), axis=0)


def build_affective_matrix(texts: List[str], lexicon,
                           use_dims: Sequence[int]) -> np.ndarray:
    matrix = np.zeros((len(texts), len(use_dims)), dtype=np.float32)
    for i, text in enumerate(texts):
        matrix[i] = text_to_vad(text, lexicon, use_dims)
    covered = int(np.count_nonzero(np.abs(matrix).sum(axis=1)))
    logger.info(
        "Built affective matrix %s — %d/%d texts matched at least one lexicon entry",
        matrix.shape, covered, len(texts),
    )
    return matrix
