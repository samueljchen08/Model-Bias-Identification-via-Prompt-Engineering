from biasjudge.run import result_name


def test_result_name():
    assert result_name("many_shot", 5, 1, "test", None) == "many_shot_n5_s1_test"
    assert result_name("zero_shot", 0, 0, "pool", 200) == "zero_shot_n0_s0_pool_limit200"


def test_failure_counts():
    import pandas as pd

    from biasjudge.run import failure_counts

    preds = pd.DataFrame(
        {
            "harmful_rating": [3, None, None, None, 4],
            "error_kind": [None, "content_filter", "content_filter", "rate_limit", None],
            "attempts": [1, 4, 4, 4, 2],
        }
    )
    assert failure_counts(preds) == {
        "content_filter": 2,
        "other": 1,
        "other_kinds": {"rate_limit": 1},
        "retried_items": 4,
    }
