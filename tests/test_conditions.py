import dspy

from biasjudge.conditions.allure import select_deviant_demos
from biasjudge.conditions.autocalibrate import harm_metric
from biasjudge.conditions.many_shot import select_random_demos


def _pool(n=20):
    return [
        dspy.Example(value_association_id=f"v{i}", statement=f"s{i}", harmful_rating=(i % 5) + 1, realism="yes")
        .with_inputs("statement")
        for i in range(n)
    ]


def _ids(exs):
    return [x.value_association_id for x in exs]


def test_random_demos_deterministic_and_sized():
    a = select_random_demos(_pool(), 5, seed=1)
    b = select_random_demos(_pool(), 5, seed=1)
    c = select_random_demos(_pool(), 5, seed=2)
    assert len(a) == 5 and _ids(a) == _ids(b) and _ids(a) != _ids(c)
    assert len(set(_ids(a))) == 5


def test_deviant_demos_picks_largest_gap_and_skips_failures():
    pool = _pool(6)  # human ratings 1,2,3,4,5,1
    preds = {"v0": 5, "v1": 2, "v2": None, "v3": 1, "v4": 4, "v5": 1}  # gaps 4,0,-,3,1,0
    sel = select_deviant_demos(pool, preds, 2, seed=0)
    assert _ids(sel) == ["v0", "v3"]
    assert sel[0].harmful_rating == 1  # demo carries the human label, not the judge's


def test_deviant_demos_tie_break_is_seeded():
    pool = _pool(10)
    preds = {f"v{i}": 3 for i in range(10)}
    a = select_deviant_demos(pool, preds, 3, seed=0)
    b = select_deviant_demos(pool, preds, 3, seed=0)
    assert _ids(a) == _ids(b)
    assert all(abs(x.harmful_rating - 3) == 2 for x in a)  # only gap-2 items (ratings 1 and 5) qualify


def test_harm_metric_score_and_feedback():
    gold = dspy.Example(statement="s", harmful_rating=4, realism="yes").with_inputs("statement")
    m = harm_metric(gold, dspy.Prediction(harmful_rating=2, realism="no"))
    assert m.score == 0.5 and "4" in m.feedback and "2" in m.feedback and "under-estimated" in m.feedback
    assert harm_metric(gold, dspy.Prediction(harmful_rating=4, realism="yes")).score == 1.0
    assert harm_metric(gold, dspy.Prediction(harmful_rating="?", realism="yes")).score == 0.0
    assert harm_metric(gold, dspy.Prediction(harmful_rating=5, realism="yes"), None, "self", None).score == 0.75
