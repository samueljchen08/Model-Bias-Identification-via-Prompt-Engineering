import math

import pytest

import pandas as pd

from biasjudge.metrics import compute_metrics, per_attribute_delta


def test_perfect_agreement():
    m = compute_metrics([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert m["spearman"] == pytest.approx(1) and m["pearson"] == pytest.approx(1) and m["mae"] == 0
    assert m["f1"] == 1 and m["kappa"] == 1 and m["accuracy"] == 1
    assert m["n"] == 5 and m["n_failed"] == 0
    assert m["pred_five_rate"] == 0.2 and m["human_five_rate"] == 0.2


def test_failed_predictions_are_dropped():
    m = compute_metrics([1, 2, 3, 4, 5, 4], [1, 2, 3, 4, 5, None])
    assert m["n"] == 5 and m["n_failed"] == 1


def test_binary_metrics():
    m = compute_metrics([4, 4, 2, 2], [4, 2, 4, 2])  # tp=1 fn=1 fp=1 tn=1
    assert m["accuracy"] == 0.5 and m["f1"] == 0.5 and m["kappa"] == 0


def test_constant_predictions_give_nan_correlation():
    m = compute_metrics([1, 2, 3, 4], [3, 3, 3, 3])
    assert math.isnan(m["spearman"]) and m["mae"] == 1.0


def test_per_attribute_delta():
    df = pd.DataFrame(
        {"base_attribute": ["age", "age", "gender", "x"], "human_median": [4, 2, 3, 1], "harmful_rating": [3, 2, 4, None]}
    )
    d = per_attribute_delta(df).set_index("base_attribute")
    assert d.loc["age", "mean_signed_error"] == -0.5 and d.loc["age", "n"] == 2
    assert d.loc["gender", "mean_signed_error"] == 1
    assert "x" not in d.index
