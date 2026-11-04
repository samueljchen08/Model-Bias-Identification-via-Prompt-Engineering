"""ALLURE (Hasanbeig et al., 2023): put the judge's own worst mistakes in context.

The N pool items where the zero-shot judge deviated most from the human median are
shown as demos, each carrying the human label. Selection reads labels for the whole
pool even though only N items reach the prompt; the write-up states this caveat.
"""

from __future__ import annotations

import random

import dspy


def select_deviant_demos(
    pool: list[dspy.Example], pool_preds: dict[str, int | None], n: int, seed: int
) -> list[dspy.Example]:
    rng = random.Random(seed)
    scored = []
    for ex in pool:
        p = pool_preds.get(ex.value_association_id)
        if p is None:
            continue
        scored.append((abs(int(p) - int(ex.harmful_rating)), rng.random(), ex))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [ex for _, _, ex in scored[:n]]
