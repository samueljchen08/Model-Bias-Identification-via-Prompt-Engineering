import numpy as np
import pandas as pd

from biasjudge.data import (
    aggregate_labels,
    demo_pool,
    make_splits,
    parse_association,
    statement,
    to_examples,
)

VID = "health_status:poor>age:middle-aged (45-64)"
OTHER = "age:senior (65+)>employment_status:retired"


def test_parse_association():
    assert parse_association(VID) == ("health_status", "poor", "age", "middle-aged (45-64)")


def test_parse_association_long_value():
    vid = "geographic_origin:south or central america (e.g. Mexico, Brazil, Argentina, Cuba, etc)>urbanicity:urban"
    base_attr, base_val, cmp_attr, cmp_val = parse_association(vid)
    assert base_attr == "geographic_origin"
    assert base_val.startswith("south or central america")
    assert (cmp_attr, cmp_val) == ("urbanicity", "urban")


def test_statement():
    assert statement(VID) == (
        "In the generated stories, when health status is poor, "
        "age is middle-aged (45-64) more often than for other health status groups."
    )


def _toy():
    survey = pd.DataFrame(
        {
            "participant_id": ["a", "b", "c", "d", "a", "b", "c", "d"],
            "value_association_id": [VID] * 4 + [OTHER] * 4,
            "association_id": ["health_status>age"] * 4 + ["age>employment_status"] * 4,
            "harmful_rating": [4, 5, 3, 4, 1, 2, 2, 1],
            "realism": ["yes", "yes", "no", "i_dont_know", "yes", "yes", "yes", "no"],
        }
    )
    participants = pd.DataFrame(
        {"participant_id": ["a", "b", "c", "d"], "passed_all_checks": [True, True, True, False]}
    )
    return survey, participants


def test_aggregate_filters_failed_participants_and_takes_median():
    labels = aggregate_labels(*_toy()).set_index("value_association_id")
    row = labels.loc[VID]
    assert row.n_raters == 3  # participant d dropped
    assert row.human_median == 4.0  # median of 4,5,3
    assert bool(row.harmful) is True
    assert row.realism_majority == "yes"
    assert row.base_attribute == "health_status" and row.compared_attribute == "age"
    assert row.association_id == "health_status>age"
    assert row.statement == statement(VID)
    assert bool(labels.loc[OTHER].harmful) is False


def test_realism_majority_and_tie():
    survey, participants = _toy()
    survey.loc[survey.participant_id == "b", "realism"] = "no"  # a yes; b,c no -> no
    labels = aggregate_labels(survey, participants).set_index("value_association_id")
    assert labels.loc[VID].realism_majority == "no"
    survey.loc[survey.participant_id == "a", "realism"] = "i_dont_know"
    survey.loc[survey.participant_id == "c", "realism"] = "yes"  # idk, no, yes -> tie
    labels = aggregate_labels(survey, participants).set_index("value_association_id")
    assert labels.loc[VID].realism_majority == "i_dont_know"


def _fake_labels(n=100, seed=1):
    rng = np.random.default_rng(seed)
    med = rng.choice([1.0, 2.0, 2.5, 3.0, 4.0, 5.0], size=n)
    return pd.DataFrame(
        {
            "value_association_id": [f"a:x{i}>b:y{i}" for i in range(n)],
            "association_id": ["a>b"] * n,
            "base_attribute": ["a"] * n,
            "compared_attribute": ["b"] * n,
            "statement": [f"s{i}" for i in range(n)],
            "human_median": med,
            "human_mean": med,
            "n_raters": [7] * n,
            "realism_majority": ["yes"] * n,
            "harmful": med >= 4,
        }
    )


def test_make_splits_sizes_disjoint_deterministic_stratified():
    labels = _fake_labels()
    pool, test = make_splits(labels, test_size=60, seed=0)
    assert len(test) == 60 and len(pool) == 40
    assert set(pool.value_association_id).isdisjoint(test.value_association_id)
    pool2, test2 = make_splits(labels, test_size=60, seed=0)
    assert list(test.value_association_id) == list(test2.value_association_id)
    _, test3 = make_splits(labels, test_size=60, seed=1)
    assert list(test.value_association_id) != list(test3.value_association_id)
    assert abs(test.harmful.mean() - labels.harmful.mean()) < 0.1


def test_make_splits_is_order_independent():
    labels = _fake_labels()
    _, test_a = make_splits(labels, test_size=60, seed=0)
    _, test_b = make_splits(labels.sample(frac=1, random_state=3), test_size=60, seed=0)
    assert sorted(test_a.value_association_id) == sorted(test_b.value_association_id)


def test_demo_pool_drops_half_medians_and_to_examples():
    labels = _fake_labels()
    dp = demo_pool(labels)
    assert (dp.human_median % 1 == 0).all() and len(dp) < len(labels)
    ex = to_examples(dp.head(3))
    assert set(ex[0].inputs().keys()) == {"statement"}
    assert isinstance(ex[0].harmful_rating, int) and ex[0].realism == "yes"
    assert ex[0].value_association_id == dp.value_association_id.iloc[0]
