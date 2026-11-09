"""LM factory for MIT Parley and the judge module shared by every condition."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import dspy
from dotenv import load_dotenv

from .prompt import JudgeAssociation

DEFAULT_BASE_URL = "https://parley.api.mit.edu/v1"
DEFAULT_MODEL = "gpt-5-mini"


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    reflection_model: str | None = None  # GEPA's instruction-rewriting model; defaults to the judge model


def load_settings() -> Settings:
    load_dotenv()
    key = os.environ.get("PARLEY_API_KEY")
    if not key:
        raise RuntimeError("PARLEY_API_KEY is not set. Copy .env.example to .env and paste your Parley key.")
    return Settings(
        api_key=key,
        base_url=os.environ.get("PARLEY_BASE_URL", DEFAULT_BASE_URL),
        model=os.environ.get("JUDGE_MODEL", DEFAULT_MODEL),
        reflection_model=os.environ.get("REFLECTION_MODEL") or None,
    )


def is_reasoning_model(model: str) -> bool:
    """OpenAI reasoning models need temperature 1.0 and a large max_tokens in DSPy."""
    name = model.split("/")[-1].lower()
    return name.startswith(("gpt-5", "o1", "o3", "o4"))


def make_lm(settings: Settings, role: str = "judge") -> dspy.LM:
    """role='judge': deterministic scoring. role='reflection': GEPA's instruction-rewriting model."""
    kwargs: dict = dict(api_base=settings.base_url, api_key=settings.api_key, cache=True)
    model = settings.model
    if role == "reflection":
        model = settings.reflection_model or settings.model
        # reasoning models insist on >= 16k; other models (e.g. Llama on Parley) cap at 8,192
        kwargs.update(temperature=1.0, max_tokens=16000 if is_reasoning_model(model) else 8000)
    elif is_reasoning_model(settings.model):
        kwargs.update(temperature=1.0, max_tokens=16000, reasoning_effort="low")
    else:
        # 2,048 not 128: AutoCalibrate's rewritten prompts make Llama reason for several hundred tokens
        # before it emits the two fields; the plain prompt never needed more than ~30 output tokens.
        kwargs.update(temperature=0, max_tokens=2048)
    # Parley is OpenAI-compatible, so litellm needs its own "openai/" provider prefix in front of
    # Parley's model id (which already carries a vendor prefix such as "openai/" or "bedrock/").
    return dspy.LM("openai/" + model, **kwargs)


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


MAX_ATTEMPTS = 4  # 1 call + 3 retries
RETRY_SLEEP = 2.0  # seconds, multiplied by the attempt number


def error_kind(e: Exception) -> str:
    """Bucket a judge failure. 'content_filter' = Parley's moderation guardrail blanked the response
    (it arrives as finish_reason=content_filter with null content, which DSPy reports as an empty response).
    The guardrail is stochastic even at temperature 0, so those are worth retrying; billing errors are not."""
    name = type(e).__name__
    msg = str(e)
    if "LMBillingError" in name or "Insufficient tokens" in msg:
        return "billing"
    if "empty or null response" in msg or "content_filter" in msg:
        return "content_filter"
    if "Too many tokens" in msg or "RateLimit" in name or "429" in msg:
        return "rate_limit"
    return name


def _one(judge: dspy.Predict, ex: dspy.Example, attempts: int = MAX_ATTEMPTS) -> dict:
    rec = {
        "value_association_id": ex.value_association_id,
        "statement": ex.statement,
        "harmful_rating": None,
        "realism": None,
        "error": None,
        "error_kind": None,
        "attempts": 0,
    }
    for k in range(attempts):
        rec["attempts"] = k + 1
        try:
            # rollout_id changes DSPy's cache key, so a retry is a fresh request rather than the cached failure
            out = judge(statement=ex.statement, config={"rollout_id": k} if k else {})
            rec["harmful_rating"] = int(out.harmful_rating)
            rec["realism"] = str(out.realism)
            rec["error"] = rec["error_kind"] = None
            return rec
        except Exception as e:  # noqa: BLE001 - one bad item must not kill a 1,000-item run
            rec["error"] = f"{type(e).__name__}: {e}"[:500]
            rec["error_kind"] = error_kind(e)
            if rec["error_kind"] == "billing":
                break
            if k + 1 < attempts:
                time.sleep(RETRY_SLEEP * (k + 1))
    return rec


def judge_items(
    judge: dspy.Predict, examples: list[dspy.Example], num_threads: int = 8, attempts: int = MAX_ATTEMPTS
) -> list[dict]:
    """Score every example, in parallel. Worker threads reuse the LM active in the calling thread."""
    lm = dspy.settings.lm

    def work(ex: dspy.Example) -> dict:
        with dspy.context(lm=lm):
            return _one(judge, ex, attempts)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        return list(pool.map(work, examples))
