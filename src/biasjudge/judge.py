"""LM factory for MIT Parley and the judge module shared by every condition."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import dspy
from dotenv import load_dotenv

from .prompt import JudgeAssociation

DEFAULT_BASE_URL = "https://parley.api.mit.edu/v1"
DEFAULT_MODEL = "openai/gpt-5.4-nano"


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str


def load_settings() -> Settings:
    load_dotenv()
    key = os.environ.get("PARLEY_API_KEY")
    if not key:
        raise RuntimeError("PARLEY_API_KEY is not set. Copy .env.example to .env and paste your Parley key.")
    return Settings(
        api_key=key,
        base_url=os.environ.get("PARLEY_BASE_URL", DEFAULT_BASE_URL),
        model=os.environ.get("JUDGE_MODEL", DEFAULT_MODEL),
    )


def is_reasoning_model(model: str) -> bool:
    """OpenAI reasoning models need temperature 1.0 and a large max_tokens in DSPy."""
    name = model.split("/")[-1].lower()
    return name.startswith(("gpt-5", "o1", "o3", "o4"))


def make_lm(settings: Settings, role: str = "judge") -> dspy.LM:
    """role='judge': deterministic scoring. role='reflection': GEPA's instruction-rewriting model."""
    kwargs: dict = dict(api_base=settings.base_url, api_key=settings.api_key, cache=True)
    if role == "reflection":
        kwargs.update(temperature=1.0, max_tokens=16000)
    elif is_reasoning_model(settings.model):
        kwargs.update(temperature=1.0, max_tokens=16000, reasoning_effort="low")
    else:
        kwargs.update(temperature=0, max_tokens=128)
    # Parley is OpenAI-compatible, so litellm needs its own "openai/" provider prefix in front of
    # Parley's model id (which already carries a vendor prefix such as "openai/" or "bedrock/").
    return dspy.LM("openai/" + settings.model, **kwargs)


def max_tokens_of(lm: dspy.LM) -> int | None:
    """DSPy stores the limit as max_completion_tokens for OpenAI reasoning models."""
    return lm.kwargs.get("max_tokens", lm.kwargs.get("max_completion_tokens"))


def lm_metadata(lm: dspy.LM) -> dict:
    return {
        "model": lm.model,
        "temperature": lm.kwargs.get("temperature"),
        "max_tokens": max_tokens_of(lm),
        "reasoning_effort": lm.kwargs.get("reasoning_effort"),
    }


def build_judge(instructions: str | None = None, demos: list[dspy.Example] | None = None) -> dspy.Predict:
    """The judge: dspy.Predict over the human-study signature, optionally with rewritten instructions or demos."""
    sig = JudgeAssociation if instructions is None else JudgeAssociation.with_instructions(instructions)
    judge = dspy.Predict(sig)
    judge.demos = list(demos or [])
    return judge


def _one(judge: dspy.Predict, ex: dspy.Example) -> dict:
    rec = {
        "value_association_id": ex.value_association_id,
        "statement": ex.statement,
        "harmful_rating": None,
        "realism": None,
        "error": None,
    }
    try:
        out = judge(statement=ex.statement)
        rec["harmful_rating"] = int(out.harmful_rating)
        rec["realism"] = str(out.realism)
    except Exception as e:  # noqa: BLE001 - one bad item must not kill a 1,000-item run
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
    return rec


def judge_items(judge: dspy.Predict, examples: list[dspy.Example], num_threads: int = 8) -> list[dict]:
    """Score every example, in parallel. Worker threads reuse the LM active in the calling thread."""
    lm = dspy.settings.lm

    def work(ex: dspy.Example) -> dict:
        with dspy.context(lm=lm):
            return _one(judge, ex)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        return list(pool.map(work, examples))
