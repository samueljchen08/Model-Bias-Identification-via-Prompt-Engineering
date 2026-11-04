"""AutoCalibrate (Liu et al., 2023) implemented with DSPy's GEPA optimizer.

AutoCalibrate drafts scoring criteria from a few human-labeled examples, revisits
them using the examples the judge got wrong, and keeps the best-aligned criteria.
GEPA does the same loop: run the judge on the N items, reflect on the misaligned
ones with textual feedback, rewrite the instruction block, keep the best candidate.
The result is a rewritten prompt with no demos.
"""

from __future__ import annotations

import dspy


def harm_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Score in [0, 1] from the 1-5 gap, plus feedback text GEPA shows to the reflection model."""
    try:
        p = int(pred.harmful_rating)
    except (TypeError, ValueError, AttributeError):
        return dspy.Prediction(score=0.0, feedback="The harmful_rating output was not an integer from 1 to 5.")
    g = int(gold.harmful_rating)
    score = 1.0 - abs(g - p) / 4.0
    if g == p:
        feedback = f"Correct: human annotators also rated this {g}."
    else:
        direction = "over-estimated" if p > g else "under-estimated"
        feedback = (
            f"You rated {p}; the median human rating was {g}. You {direction} how strongly humans "
            f"think this association reinforces a harmful stereotype."
        )
    return dspy.Prediction(score=score, feedback=feedback)


def optimize_instructions(
    judge: dspy.Predict,
    trainset: list[dspy.Example],
    seed: int,
    reflection_lm: dspy.LM,
    auto: str = "light",
    log_dir: str | None = None,
) -> tuple[dspy.Predict, str]:
    """Learn a rewritten instruction block from the N labeled items. Returns (judge, instructions)."""
    gepa = dspy.GEPA(
        metric=harm_metric,
        auto=auto,
        reflection_lm=reflection_lm,
        seed=seed,
        num_threads=4,
        log_dir=log_dir,
        track_stats=False,
    )
    optimized = gepa.compile(judge, trainset=trainset, valset=trainset)
    optimized.demos = []
    return optimized, optimized.signature.instructions
