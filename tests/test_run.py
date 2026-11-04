from biasjudge.run import result_name


def test_result_name():
    assert result_name("many_shot", 5, 1, "test", None) == "many_shot_n5_s1_test"
    assert result_name("zero_shot", 0, 0, "pool", 200) == "zero_shot_n0_s0_pool_limit200"
