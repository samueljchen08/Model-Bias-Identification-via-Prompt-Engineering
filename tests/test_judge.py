import dspy

from biasjudge.judge import Settings, build_judge, error_kind, is_reasoning_model, judge_items, make_lm, max_tokens_of
from biasjudge.prompt import HUMAN_INSTRUCTIONS, JudgeAssociation


def test_signature_carries_human_instructions():
    assert JudgeAssociation.instructions == HUMAN_INSTRUCTIONS
    assert "reinforces a harmful stereotype" in HUMAN_INSTRUCTIONS
    assert "Strongly disagree" in HUMAN_INSTRUCTIONS
    assert "Literacy status" in HUMAN_INSTRUCTIONS
    assert "widely held belief about a social group" in HUMAN_INSTRUCTIONS
    assert set(JudgeAssociation.input_fields) == {"statement"}
    assert set(JudgeAssociation.output_fields) == {"harmful_rating", "realism"}


def test_is_reasoning_model():
    assert is_reasoning_model("openai/gpt-5.4-nano")
    assert is_reasoning_model("openai/gpt-5.6-luna")
    assert not is_reasoning_model("bedrock/claude-haiku-4-5")
    assert not is_reasoning_model("gpt-4o")


def test_make_lm_parameters():
    s = Settings(api_key="k", base_url="https://parley.api.mit.edu/v1", model="bedrock/claude-haiku-4-5")
    lm = make_lm(s)
    assert lm.model == "openai/bedrock/claude-haiku-4-5"
    assert lm.kwargs["temperature"] == 0 and lm.kwargs["max_tokens"] == 2048
    assert lm.kwargs["api_base"] == s.base_url and lm.kwargs["api_key"] == "k"

    r = make_lm(Settings(api_key="k", base_url=s.base_url, model="openai/gpt-5.4-nano"))
    assert r.kwargs["temperature"] == 1.0 and max_tokens_of(r) == 16000
    assert r.kwargs["reasoning_effort"] == "low"

    refl = make_lm(s, role="reflection")
    assert refl.kwargs["temperature"] == 1.0 and max_tokens_of(refl) == 8000
    assert refl.model == "openai/bedrock/claude-haiku-4-5"
    refl2 = make_lm(Settings(api_key="k", base_url=s.base_url, model="llama", reflection_model="gpt-5-mini"), role="reflection")
    assert refl2.model == "openai/gpt-5-mini" and max_tokens_of(refl2) == 16000


def test_build_judge_overrides():
    demo = dspy.Example(statement="x", harmful_rating=4, realism="yes").with_inputs("statement")
    j = build_judge(instructions="NEW", demos=[demo])
    assert j.signature.instructions == "NEW" and j.demos == [demo]
    assert build_judge().signature.instructions == HUMAN_INSTRUCTIONS
    assert build_judge().demos == []


def test_judge_items_parses_and_records_errors():
    lm = dspy.utils.DummyLM(
        [{"harmful_rating": "4", "realism": "yes"}, {"harmful_rating": "banana", "realism": "yes"}]
    )
    ex = [
        dspy.Example(value_association_id="v1", statement="s1").with_inputs("statement"),
        dspy.Example(value_association_id="v2", statement="s2").with_inputs("statement"),
    ]
    with dspy.context(lm=lm):
        recs = judge_items(build_judge(), ex, num_threads=1, attempts=1)
    assert recs[0] == {
        "value_association_id": "v1",
        "statement": "s1",
        "harmful_rating": 4,
        "realism": "yes",
        "error": None,
        "error_kind": None,
        "attempts": 1,
    }
    assert recs[1]["harmful_rating"] is None and recs[1]["error"] and recs[1]["attempts"] == 1


def test_error_kind_buckets():
    class LMBillingError(Exception):
        pass

    assert error_kind(LMBillingError("x")) == "billing"
    assert error_kind(ValueError("Insufficient tokens. Free model token balance is exhausted")) == "billing"
    assert error_kind(ValueError("The LM returned an empty or null response.")) == "content_filter"
    assert error_kind(RuntimeError("Bedrock API error: Too many tokens, please wait")) == "rate_limit"
    assert error_kind(KeyError("boom")) == "KeyError"


def test_judge_items_retries_blocked_responses(monkeypatch):
    """A guardrail-blanked response (empty output) is retried with a new rollout_id; a billing error is not."""
    import biasjudge.judge as judge_mod

    monkeypatch.setattr(judge_mod, "RETRY_SLEEP", 0)
    calls: list[int | None] = []

    class FlakyJudge:
        def __init__(self, outcomes):
            self.outcomes = list(outcomes)

        def __call__(self, statement, config=None):
            calls.append((config or {}).get("rollout_id"))
            o = self.outcomes.pop(0)
            if isinstance(o, Exception):
                raise o
            return dspy.Prediction(harmful_rating=o, realism="no")

    ex = dspy.Example(value_association_id="v1", statement="s1").with_inputs("statement")
    blocked = ValueError("The LM returned an empty or null response.")
    rec = judge_mod._one(FlakyJudge([blocked, blocked, 3]), ex)
    assert rec["harmful_rating"] == 3 and rec["attempts"] == 3 and rec["error"] is None
    assert calls == [None, 1, 2]

    calls.clear()
    rec = judge_mod._one(FlakyJudge([blocked] * 4), ex)
    assert rec["harmful_rating"] is None and rec["error_kind"] == "content_filter" and rec["attempts"] == 4

    calls.clear()
    rec = judge_mod._one(FlakyJudge([ValueError("Insufficient tokens. Free model token balance is exhausted")] * 4), ex)
    assert rec["error_kind"] == "billing" and rec["attempts"] == 1 and calls == [None]
