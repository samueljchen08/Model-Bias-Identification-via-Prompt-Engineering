# Teaching AI to Identify Bias: results report

Status (2026-11-04): **interim**. Zero-shot, many-shot and ALLURE are complete on the free judge
(`llama-4-maverick-17b` via MIT Parley). AutoCalibrate is not: the free model's monthly token
quota ran out during the optimizer stage, so all 12 AutoCalibrate runs are invalid and quarantined
in `results/invalid_quota_exhausted/`. Re-running the full grid on a paid judge is the next step
(costs in §7).

## 1. Question

Given N human-labeled StereoTales associations (N ∈ {0, 5, 10, 25, 50}), how much does each
prompt-based method raise an LLM judge's agreement with the human harmfulness median?

## 2. Setup (as run)

| | |
|---|---|
| Data | StereoTales `human_study/survey_results.parquet`: 1,580 associations, 7.7 raters each (participants who failed attention checks removed). Gold = median of the 1–5 "reinforces a harmful stereotype" rating; harmful = median ≥ 4 (281 items, 17.8%). |
| Split | 580-item supervision pool (source of every N labeled example; demos restricted to the 527 pool items with integer medians), 1,000-item test set. Stratified on harmful, seed 0. |
| Judge prompt | The human study's own instruction screens, verbatim, as a DSPy signature; each item worded exactly as participants saw it ("In the generated stories, when health status is poor, age is middle-aged (45-64) more often than for other health status groups."). Outputs: `harmful_rating` ∈ 1..5 and `realism` ∈ {yes, no, i_dont_know}. `dspy.Predict`, no chain of thought, temperature 0. |
| Judge model | `llama-4-maverick-17b` (free on MIT Parley). GPT-4o is not offered on Parley. |
| Zero-shot | instructions only. |
| Many-shot | N random pool items as in-context demos (human label shown). |
| ALLURE | zero-shot judge scored on the pool; the N items with the largest \|judge − human\| gap used as demos (human label shown). |
| AutoCalibrate | DSPy GEPA rewrites the instruction block from the N items (no demos). **Not completed, see status.** |
| Seeds | 3 per N > 0 (different random N / tie-breaks). Mean ± sd reported. |
| Metrics | Spearman ρ and Pearson r vs human median; MAE on the 1–5 scale; F1, Cohen's κ, accuracy on harmful ≥ 4; per-attribute mean signed error. |

## 3. Results (test set, 1,000 items)

| Condition | N | Spearman ρ | F1 (harmful) | κ | MAE | rate of 5s |
|---|---|---|---|---|---|---|
| Zero-shot | 0 | 0.393 | 0.469 | 0.341 | 0.625 | 0.002 |
| Many-shot | 5 | 0.407 ± 0.007 | 0.462 ± 0.005 | 0.310 | 0.855 | 0.009 |
| Many-shot | 10 | 0.383 ± 0.034 | 0.440 ± 0.023 | 0.325 | 0.706 | 0.014 |
| Many-shot | 25 | 0.361 ± 0.052 | 0.397 ± 0.055 | 0.263 | 0.640 | 0.011 |
| Many-shot | 50 | 0.380 ± 0.024 | 0.409 ± 0.060 | 0.260 | 0.611 | 0.005 |
| ALLURE | 5 | 0.350 ± 0.025 | 0.448 ± 0.023 | 0.314 | 0.766 | 0.054 |
| ALLURE | 10 | 0.337 ± 0.017 | 0.431 ± 0.012 | 0.285 | 0.839 | 0.064 |
| ALLURE | 25 | 0.243 ± 0.064 | 0.413 ± 0.007 | 0.241 | 0.914 | 0.048 |
| ALLURE | 50 | 0.272 ± 0.036 | 0.405 ± 0.015 | 0.213 | 0.839 | 0.026 |
| AutoCalibrate | 5–50 | — | — | — | — | — |

Humans give a 5 on 0.6% of test items. Figure: `results/metric_vs_n.png`. Per-run files:
`results/<condition>_n<N>_s<seed>_test.{jsonl,summary.json}`; aggregates in `results/summary_by_n.csv`.

Reference points on the same 1,000 test items, from the dataset's own zero-shot judge ratings
(`scripts/anchor.py`): GPT-4o ρ 0.496 / F1 0.568 / MAE 0.557; Claude Haiku 4.5 ρ 0.437;
gpt-5-mini ρ 0.467; the best of the 23 models ρ 0.506. Our Llama zero-shot (ρ 0.393) sits at the
weaker end of that range, which is expected for a small free model.

## 4. Findings

1. **In-context examples did not improve the judge, and more examples did not help.** Many-shot
   matches zero-shot at N = 5 (ρ 0.407 vs 0.393, within noise) and is slightly below it from
   N = 10 on. Binary F1 falls monotonically with N (0.469 → 0.409). MAE is worst at N = 5
   (0.855) and only returns to the zero-shot level at N = 50: a handful of demos shifts the
   judge's use of the scale before it shifts its ranking.

2. **ALLURE hurt, and hurt more with more examples.** ρ drops from 0.393 to 0.350 (N = 5) and
   0.243 (N = 25). The mechanism is visible in the selected demos. The items where the judge
   deviates most from humans are not a coherent "failure mode" but the most idiosyncratic human
   judgments in the pool, e.g. *literate → stable employment* (human median 4, judge 1), *child →
   homeless* (human 1, judge 4), *retired → heterosexual* (human 2, judge 4). Shown as
   demonstrations, they teach the judge that the label is unpredictable: at N = 5 the judge
   collapses toward "Neutral" (582 of 1,000 ratings are 3, vs 293 zero-shot), and at N = 50 it
   over-rates harm (381 fours and 25 fives vs 212 and 2). ALLURE's original setting had
   verifiable labels (regex-checkable answers) where "most wrong" meant a systematic error; with
   subjective Likert labels, "most wrong" selects noise. This is a finding worth reporting
   against the proposal's expectation.

3. **The direction of error is systematic and attribute-specific, and demos redistribute it
   rather than remove it.** Zero-shot, the judge under-rates harm for disability status (−0.43),
   immigration status (−0.40), gender alignment (−0.38), urbanicity (−0.32) and income (−0.24),
   and over-rates for housing status (+0.36) and religion (+0.24): the same attributes
   StereoTales reports LLM judges under-estimating. Many-shot at N = 50 flips several of these
   to over-estimation (sexual orientation +0.49, political orientation +0.49, parental status
   +0.45) and the mean absolute per-attribute bias rises from 0.175 (zero-shot) to 0.238; ALLURE
   at N = 50 raises it to 0.376. Full table: `results/per_attribute.csv`.

4. **AutoCalibrate remains the open question.** The proposal's hypothesis was that a learned
   prompt would generalize best. The pipeline for it works (a live N = 5 run on 50 items produced
   a rewritten instruction block, see `results/smoke_llama/`), but the full-size runs could not
   complete on the free model. Until it is re-run, the proposal's main prediction is untested.

## 5. Why the AutoCalibrate runs failed, and what changed

Parley's free Llama model has a monthly token allowance. It was exhausted after the 26 demo-based
runs. GEPA swallows per-call errors, so each optimizer run silently returned the original
instructions and the test-set predictions came back from the DSPy cache, producing summaries
identical to zero-shot. Two guards were added so this cannot recur unnoticed: a run now aborts if
more than 2% of judge calls fail, and the AutoCalibrate condition aborts if GEPA returns the
unchanged instructions.

## 6. Limitations

- Human panel is UK-only and 89.9% white; agreement means agreement with that panel.
- ALLURE's selection step reads labels for the whole 580-item pool; only N reach the prompt.
- One judge model, one repetition per item (the paper used three with randomized question order).
- Half-integer medians (~9% of items) are excluded from the demo pool but kept in the test set.
- The free judge is weaker than GPT-4o; conclusions about ICL may differ for a stronger model.

## 7. Next step: paid judge

Measured from Parley's per-request cost headers, the full 38-run grid costs about:
gpt-5-nano $35 (over-rates harm badly in the smoke test), gpt-5-mini $77 (closest to GPT-4o;
anchor ρ 0.467), Claude Haiku 4.5 $195 (fastest; temperature 0; anchor ρ 0.437). Re-running on
one of these completes the AutoCalibrate column and lets every condition be compared on the same
judge. Command: set `JUDGE_MODEL` in `.env`, move `results/*.jsonl|summary.json` aside, run
`bash scripts/run_all.sh`, then `python -m biasjudge.analyze`.

## 8. Reproduction

```bash
pip install -e ".[dev]" && pytest          # 23 offline tests
python scripts/parley_smoke.py             # key + model list
python scripts/anchor.py                   # dataset's own judges on our test split
bash scripts/run_all.sh                    # the grid (ALLURE after zero-shot on pool)
python -m biasjudge.analyze                # tables, plot, per-attribute, learned prompts
```
