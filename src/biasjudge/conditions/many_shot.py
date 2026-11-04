"""Many-shot: N random human-labeled items shown in context (Song et al., 2024)."""

from __future__ import annotations

import random

import dspy


def select_random_demos(pool: list[dspy.Example], n: int, seed: int) -> list[dspy.Example]:
    return random.Random(seed).sample(list(pool), n)
