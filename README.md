# Model Bias Identification via Prompt Engineering

Can an LLM judge be prompt-engineered, with only N human-labeled examples, into agreeing
with human annotators about which demographic associations are harmful stereotypes?

We take the [StereoTales](https://huggingface.co/datasets/giskardai/StereoTales) human study
(1,580 associations, each rated by ~8 UK annotators on a 1–5 "reinforces a harmful stereotype"
scale), give an LLM judge the *same* instructions the humans saw, and measure agreement with the
human median as a function of N ∈ {0, 5, 10, 25, 50} under four conditions:

| Condition | What the N labels do | Owner |
|---|---|---|
| Zero-shot | nothing | Ruoxi |
| Many-shot | N random labeled items in context ([Song et al.](https://arxiv.org/abs/2406.11629)) | Jefferson |
| ALLURE | the N items the zero-shot judge got most wrong, in context ([Hasanbeig et al.](https://arxiv.org/abs/2309.13701)) | Maeve |
| AutoCalibrate | N items drive DSPy's GEPA optimizer to rewrite the scoring instructions ([Liu et al.](https://arxiv.org/abs/2309.13308)) | Sam |

**Results report: `docs/report.md`** (interim, 33 of 38 runs: zero-shot, many-shot and ALLURE
complete on the free Llama judge except three N = 50 cells; AutoCalibrate done for N = 5 and
two seeds of N = 10. Two free keys' monthly quotas are spent, and since Nov 2026 Parley's free
Llama has a moderation guardrail that blanks part of each AutoCalibrate run, see report §6).
Design spec: `docs/superpowers/specs/2026-11-03-llm-bias-judge-design.md`.
Implementation plan: `docs/superpowers/plans/2026-11-03-llm-bias-judge.md`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # then paste your MIT Parley key (https://parley.mit.edu → Parley API)
pytest                      # offline unit tests, no API calls
```

`.env` holds `PARLEY_API_KEY`, `PARLEY_BASE_URL` (default `https://parley.api.mit.edu/v1`) and
`JUDGE_MODEL` (a Parley model id as listed by `/v1/models`, e.g. `gpt-5-mini`, `claude-haiku-4-5`, `llama-4-maverick-17b`; the free Llama model has a monthly token allowance).

## Running

```bash
python scripts/parley_smoke.py                 # 0. check the key, list models, one judge call
python scripts/anchor.py                       #    the dataset's own 23 judges vs humans on our test split
python -m biasjudge.run --condition zero_shot --n 0 --split test --limit 50   # 1. small smoke run
bash scripts/run_all.sh                        # 2. the full grid (≈40k judge calls, all cached)
python -m biasjudge.analyze                    # 3. results/summary*.csv + results/metric_vs_n.png
```

One run = one (condition, N, seed, split). Each writes `results/<name>.jsonl` (one row per item
with the judge's rating next to the human median) and `results/<name>.summary.json` (metrics,
the exact instructions used, demo ids, optimizer info). ALLURE needs
`zero_shot_n0_s0_pool` to exist first; `run_all.sh` orders this correctly.

## How it fits together

```
src/biasjudge/
  data.py          HF download → per-item human median, "harmful" flag, item sentence, pool/test split
  prompt.py        the human-study instructions as a DSPy signature (statement → harmful_rating, realism)
  judge.py         Parley LM factory (handles gpt-5 reasoning-model constraints), build_judge, judge_items
  metrics.py       Spearman, Pearson, MAE, F1/κ/accuracy on harmful ≥ 4, per-attribute signed error
  conditions/      many_shot (random demos), allure (most-deviant demos), autocalibrate (GEPA instructions)
  run.py           CLI for one run;  analyze.py  aggregates summaries
scripts/           parley_smoke.py, anchor.py, run_all.sh
```

Splits: 580-item supervision pool (the only source of the N labeled examples; demos use items
with integer medians) and a 1,000-item test set (all reported numbers), stratified on harmful,
seed 0. For a given (N, seed) many-shot and AutoCalibrate get the identical N items.

## Caveats to carry into the write-up

- The human panel is UK-only and 89.9% white; "agreement with humans" means agreement with that panel.
- ALLURE's *selection* reads labels for the whole pool, even though only N items reach the prompt.
- AutoCalibrate selects its best instruction on the same N items it learns from (no held-out set at N = 5).
- One judge model, one repetition; the paper used three repetitions with randomized question order.
  gpt-5 family models run at temperature 1.0 (DSPy requirement) instead of 0.
- Half-integer medians (~9% of items) stay in the test set but are excluded from the demo pool.
