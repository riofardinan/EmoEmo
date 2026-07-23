"""Class-incremental protocols.

EmoGrowth evaluates under two families of protocol (Section 4.1):
  (1) split all classes into equal increments  -> "B0-Ik"
  (2) train a base model on many classes first -> "Bn-Ik"

GoEmotions has 27 emotions + neutral = 28 classes, which is exactly the size of
the paper's Audio28 dataset, so we can reuse its four protocols verbatim:
B0-I7, B0-I4, B16-I3, B16-I2. The 27-class protocols from Video27/Brain27
(B0-I9, B0-I3, B15-I3, B15-I2) are here too, for a run with neutral dropped.
"""

from typing import List

# name -> (init_cls, increment, expected total classes)
PROTOCOLS = {
    # 28 classes: 27 emotions + neutral (mirrors Audio28 in the paper)
    "B0-I7": (7, 7, 28),
    "B0-I4": (4, 4, 28),
    "B16-I3": (16, 3, 28),
    "B16-I2": (16, 2, 28),
    # 27 classes: neutral dropped (mirrors Video27 / Brain27)
    "B0-I9": (9, 9, 27),
    "B0-I3": (3, 3, 27),
    "B15-I3": (15, 3, 27),
    "B15-I2": (15, 2, 27),
}


def build_increments(init_cls: int, increment: int, total_class: int) -> List[int]:
    """Sizes of each incremental task.

    Follows DataManager in EmoGrowth: keep appending `increment` while it still
    fits. If the classes do not divide evenly the remainder is left out, which
    is what the original does — so we assert instead of silently dropping
    classes.
    """
    increments = [init_cls]
    while sum(increments) + increment <= total_class:
        increments.append(increment)
    if sum(increments) != total_class:
        raise ValueError(
            f"init_cls={init_cls}, increment={increment} covers only "
            f"{sum(increments)} of {total_class} classes. Pick a protocol that "
            f"divides evenly, e.g. one of {sorted(PROTOCOLS)}."
        )
    return increments


def resolve_protocol(name: str):
    if name not in PROTOCOLS:
        raise KeyError(f"Unknown protocol '{name}'. Known: {sorted(PROTOCOLS)}")
    init_cls, increment, total_class = PROTOCOLS[name]
    return init_cls, increment, total_class
