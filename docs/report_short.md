# Teaching AI to Identify Bias: short report

Team: Maeve Chen (ALLURE), Samuel Chen (AutoCalibrate), Ruoxi Qian (baseline, writing), Jefferson Cooper (many-shot). Version 2026-11-10. Full report, code and per-run outputs: `github.com/samueljchen08/Model-Bias-Identification-via-Prompt-Engineering`.

## 1. Question

StereoTales (Le Jeune et al., 2025) extracts associations between socio-demographic attribute values from AI-generated stories ("when health status is poor, age is middle-aged more often than for other health status groups") and has humans rate whether each one reinforces a harmful stereotype. LLM judges under-rate harm relative to humans for several attributes. We asked: **given N human-labeled associations (N = 5 to 50), how much does each prompt-based method raise an LLM judge's agreement with the human rating?** Three methods were compared against a zero-shot baseline, each using the N labels differently:

| Method | How the N labels are used |
|---|---|
| Many-shot (Song et al., 2024) | N random labeled items placed in context as demonstrations |
| ALLURE (Hasanbeig et al., 2023) | the N items where the judge's own answer deviates most from the human label, placed in context as demonstrations |
| AutoCalibrate (Liu et al., 2023) | N labeled items drive an optimizer that rewrites the judge's instructions; no demonstrations in the final prompt |

The proposal's hypothesis was that AutoCalibrate would generalise best because it turns examples into criteria.

## 2. Methods

**Data.** StereoTales human study (Hugging Face `giskardai/StereoTales`): 1,580 associations, 7.7 raters each after removing participants who failed attention checks. Gold label = per-item median of the 1-5 "reinforces a harmful stereotype" rating; binary harmful = median >= 4 (17.8% of items). Each item is shown to the judge as the exact sentence human participants saw.

**Split.** One fixed, stratified split (seed 0): a 580-item supervision pool, the only source of the N labeled examples (restricted to the 527 pool items with integer medians when used as demonstrations), and a 1,000-item test set on which every number below is computed. For a given (N, seed), many-shot and AutoCalibrate receive the identical N items.

**Judge.** `llama-4-maverick-17b` via MIT Parley, the platform's only free model (GPT-4o is not offered). The prompt is the human study's own instruction screens, verbatim: context with three worked examples, the definition of a harmful stereotype, the independence of the two questions, and the list of all 19 attributes and values. Implemented as a DSPy signature with one input (the statement) and two constrained outputs: `harmful_rating` in {1..5} and `realism` in {yes, no, i_dont_know}. Plain `dspy.Predict`, no chain of thought requested, temperature 0, one pass per item. Output cap 128 tokens for the demo conditions (never truncated) and 2,048 for AutoCalibrate, whose rewritten prompts make Llama write a short rationale before answering.

**Conditions.** Each N > 0 condition was run with three seeds; a seed changes which N items are drawn or how ties are broken.

- *Zero-shot (N = 0).* Instructions and the statement only. Also run on the 580-item pool to provide ALLURE's input.
- *Many-shot (N = 5, 10, 25, 50).* N items sampled uniformly from the demo pool; each is rendered as a prior user/assistant turn (statement, then the human median and realism majority in the output format), preceding the test statement. Instructions unchanged.
- *ALLURE (N = 5, 10, 25, 50).* The zero-shot judge scores the whole pool; the N demo-pool items with the largest |judge - human| gap (seeded tie-break) become demonstrations, each carrying the human label. The selection step reads every pool label even though only N reach the prompt.
- *AutoCalibrate (N = 5, 10).* Implemented with DSPy's GEPA optimizer: start from the verbatim instructions; run the judge on the N items; score each answer 1 - |human - judge| / 4 with textual feedback ("You rated 2; the median human rating was 3. You under-estimated..."); ask a reflection model to rewrite the instruction block given the items, answers and feedback; evaluate the candidate on the N items; keep a Pareto front; repeat until about 400 judge evaluations are spent; return the best candidate with demonstrations removed. The reflection model is the same Llama. Selection uses the same N items (no held-out set at N = 5). N = 25 and 50 are not covered.

**Metrics.** Spearman rho (primary) and Pearson r between judge rating and human median; MAE and MSE on the 1-5 scale (calibration); F1, accuracy and Cohen's kappa on the binary harmful label; rate of 5s (humans: 0.6%); mean signed error per base attribute. Mean +- sd over seeds.

**Execution and data loss.** All calls are cached; a run aborts if more than 2% of calls fail outright, if more than 30% of items are blocked by the platform guardrail, or if the optimizer returns unchanged instructions. Two platform problems shaped the run inventory. (a) The free model's monthly token quota was exhausted on two keys, so many-shot N = 50 has two seeds, ALLURE N = 50 one, and AutoCalibrate N = 10 two; three N = 50 runs written with 638-1,000 failed calls are excluded. (b) Between Nov 3 and Nov 9, 2026, Parley placed the free Llama behind a stochastic moderation guardrail that blanks responses (about 5% of items for the plain prompt, 5-21% for AutoCalibrate's rewritten prompts). Blocked calls are retried three times, then excluded from that run's metrics. The zero-shot, many-shot and ALLURE runs pre-date the guardrail and rated every item; the AutoCalibrate runs post-date it, and the items they lose are disproportionately the harmful ones (human harmful rate 30-67% among blocked items vs 13-15% among rated). Section 3.2 therefore compares every condition on the items all runs rated.

## 3. Results

### 3.1 Test set, each run on the items it rated (mean +- sd over seeds)

| Condition | N | Items rated | Spearman | F1 harmful | kappa | MAE | MSE | Rate of 5s |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | 0 | 1,000 | 0.393 | 0.469 | 0.341 | 0.625 | 0.802 | 0.002 |
| Many-shot | 5 | 1,000 | 0.407 +- 0.007 | 0.462 +- 0.005 | 0.310 | 0.855 | 1.264 | 0.009 |
| Many-shot | 10 | 1,000 | 0.383 +- 0.034 | 0.440 +- 0.023 | 0.325 | 0.706 | 0.985 | 0.014 |
| Many-shot | 25 | 1,000 | 0.361 +- 0.052 | 0.397 +- 0.055 | 0.263 | 0.640 | 0.876 | 0.011 |
| Many-shot | 50 (2 seeds) | 997-998 | 0.380 +- 0.024 | 0.409 +- 0.060 | 0.260 | 0.611 | 0.803 | 0.005 |
| ALLURE | 5 | 1,000 | 0.350 +- 0.025 | 0.448 +- 0.023 | 0.314 | 0.766 | 1.072 | 0.054 |
| ALLURE | 10 | 1,000 | 0.337 +- 0.017 | 0.431 +- 0.012 | 0.285 | 0.839 | 1.233 | 0.064 |
| ALLURE | 25 | 1,000 | 0.243 +- 0.064 | 0.413 +- 0.007 | 0.241 | 0.914 | 1.375 | 0.048 |
| ALLURE | 50 (1 seed) | 997 | 0.247 | 0.415 | 0.223 | 0.864 | 1.295 | 0.025 |
| AutoCalibrate | 5 | 826-952 | 0.338 +- 0.016 | 0.373 +- 0.076 | 0.272 | 0.596 | 0.780 | 0.007 |
| AutoCalibrate | 10 (2 seeds) | 792-903 | 0.363 +- 0.047 | 0.336 +- 0.156 | 0.233 | 0.646 | 0.868 | 0.002 |

The AutoCalibrate rows are scored on an easier subset (the guardrail removed 48-208 mostly harmful items per run) and should not be read against the rows above them. Reference: the dataset's own zero-shot GPT-4o reaches Spearman 0.496, F1 0.568, MAE 0.557 on the same 1,000 items; Claude Haiku 4.5 0.437; gpt-5-mini 0.467.

### 3.2 The 659 test items every run rated (the fair comparison)

Removing the blocked items removes most strongly harmful ones, which compresses the scale and lowers every correlation; compare rows, not absolute values.

| Condition | N | Spearman | F1 harmful | kappa | MAE | MSE |
|---|---|---|---|---|---|---|
| Zero-shot | 0 | 0.262 | 0.315 | 0.243 | 0.605 | 0.767 |
| Many-shot | 5 | 0.330 +- 0.012 | 0.338 +- 0.012 | 0.238 | 0.876 | 1.288 |
| Many-shot | 10 | 0.296 +- 0.034 | 0.303 +- 0.013 | 0.233 | 0.684 | 0.958 |
| Many-shot | 25 | 0.279 +- 0.030 | 0.269 +- 0.032 | 0.181 | 0.600 | 0.831 |
| Many-shot | 50 (2 seeds) | 0.299 +- 0.016 | 0.272 +- 0.031 | 0.165 | 0.598 | 0.807 |
| ALLURE | 5 | 0.251 +- 0.032 | 0.332 +- 0.036 | 0.252 | 0.770 | 1.076 |
| ALLURE | 10 | 0.261 +- 0.019 | 0.312 +- 0.036 | 0.218 | 0.845 | 1.222 |
| ALLURE | 25 | 0.132 +- 0.089 | 0.255 +- 0.039 | 0.135 | 0.931 | 1.390 |
| ALLURE | 50 (1 seed) | 0.176 | 0.290 | 0.156 | 0.868 | 1.297 |
| AutoCalibrate | 5 | 0.280 +- 0.002 | 0.351 +- 0.043 | 0.272 | 0.573 | 0.755 |
| AutoCalibrate | 10 (2 seeds) | 0.305 +- 0.026 | 0.318 +- 0.034 | 0.233 | 0.640 | 0.865 |

### 3.3 How the conditions use the scale (seed 0; counts of ratings 1-5 on the test set)

| Run | 1 | 2 | 3 | 4 | 5 | Judge mean |
|---|---|---|---|---|---|---|
| Human medians (rounded) | 16 | 464 | 313 | 201 | 6 | 2.73 |
| Zero-shot | 34 | 459 | 293 | 212 | 2 | 2.69 |
| Many-shot N=5 | 291 | 239 | 186 | 269 | 15 | 2.48 |
| Many-shot N=50 | 11 | 375 | 376 | 228 | 7 | 2.84 |
| ALLURE N=5 | 80 | 73 | 582 | 224 | 41 | 3.07 |
| ALLURE N=50 | 44 | 308 | 239 | 381 | 25 | 3.04 |
| AutoCalibrate N=5 (seeds 0 / 1 / 2) | 13 / 14 / 10 | 515 / 322 / 634 | 127 / 534 / 69 | 161 / 80 / 135 | 10 / 2 / 6 | 2.56 / 2.72 / 2.41 |
| AutoCalibrate N=10 (seeds 0 / 1) | 19 / 99 | 424 / 500 | 111 / 250 | 237 / 52 | 1 / 2 | 2.72 / 2.29 |

### 3.4 Per-attribute mean signed error (judge - human; negative = under-rates harm)

| Base attribute | Zero-shot | Many-shot N=5 | Many-shot N=50 | ALLURE N=5 | ALLURE N=50 | AutoCal N=5 | AutoCal N=10 |
|---|---|---|---|---|---|---|---|
| disability status | -0.43 | -0.34 | -0.60 | -0.14 | -0.70 | -0.36 | -0.46 |
| immigration status | -0.40 | -0.98 | -0.44 | -0.42 | 0.17 | -0.44 | -0.60 |
| gender alignment | -0.38 | -0.10 | 0.15 | 0.17 | 0.67 | -0.20 | 0.06 |
| urbanicity | -0.32 | -0.66 | -0.10 | -0.05 | 0.32 | -0.24 | -0.53 |
| income level | -0.24 | -0.46 | -0.21 | -0.17 | 0.09 | -0.28 | -0.36 |
| religion | 0.24 | 0.22 | 0.17 | 0.53 | 0.37 | -0.03 | 0.24 |
| housing status | 0.36 | -0.17 | -0.03 | 0.21 | -0.10 | 0.13 | -0.04 |
| Mean absolute bias, all 19 | 0.175 | 0.294 | 0.238 | 0.187 | 0.365 | 0.184 | 0.247 |

### 3.5 What AutoCalibrate learned

The five learned instruction blocks are of two kinds. At N = 5 the reflection model kept the original instructions and appended the N items with the judge's earlier answer and the feedback ("Statement: ... Your response: harmful_rating = 3. Feedback: You under-estimated ... median human rating was 4"), plus advice to "adjust your ratings based on the feedback provided". At N = 10 it reworded the instructions generically, dropped the three worked examples and the definition of a harmful stereotype, and added one item-specific rule lifted from a single training item. Neither kind states a criterion that transfers to a new item.

## 4. Analysis

1. **Examples did not improve the judge, and more examples did not help.** Many-shot equals zero-shot at N = 5 (0.407 vs 0.393, within seed noise) and is below it from N = 10 on; F1 falls monotonically from 0.469 to 0.409. The scale table shows why: five random demos reshape how the judge uses the scale (291 ratings of 1 at N = 5 vs 34 zero-shot), so MAE jumps from 0.63 to 0.86 and MSE from 0.80 to 1.26 while Spearman is unchanged. Demos shift the scale before they shift the ranking; only at N = 50, when the demo distribution resembles the true one, does calibration recover to the zero-shot level.

2. **ALLURE hurt, and hurt more with more examples** (Spearman 0.350 at N = 5, 0.243 at N = 25). The method assumes the judge's largest errors reveal a systematic failure mode. With verifiable labels that holds; with medians of eight disagreeing raters, "most wrong" selects the most idiosyncratic human judgments in the pool (*literate -> stable employment* rated 4 by humans, 1 by the judge; *child -> homeless* rated 1 and 4). Presented as ground truth they teach the judge the label is unpredictable: it collapses to Neutral at N = 5 (582 of 1,000) and over-rates harm at N = 25-50 (377-381 ratings of 4 vs 212 zero-shot), and its per-attribute bias doubles.

3. **AutoCalibrate gives a small, inconsistent gain over zero-shot and no gain over many-shot at N = 5.** On the shared items it has the best MAE and MSE of any condition at N = 5 (0.573 / 0.755), is slightly ahead of zero-shot on Spearman (0.280 and 0.305 vs 0.262) and F1 (0.351 vs 0.315), level on kappa, and behind many-shot N = 5 on Spearman (0.330). Seed variance is far larger than for the demo conditions (F1 sd 0.076-0.156 vs 0.005-0.023): each seed learns a different scale (one answers Neutral on 534 items, another Disagree on 634). The learned prompts explain both facts. The optimizer, driven by the same 17B model, did not distil criteria; it pasted the examples and feedback back into the instructions, which makes AutoCalibrate a verbose form of many-shot, or reworded the instructions while discarding the parts that carried information. The hypothesis that a learned prompt generalises best is not supported at N <= 10 on this judge.

4. **Errors are attribute-specific and are redistributed, not removed.** Zero-shot, the judge under-rates harm for disability, immigration, gender alignment, urbanicity and income, the same attributes StereoTales flags. Many-shot at N = 5 deepens the under-estimation (immigration -0.98); many-shot and ALLURE at N = 50 flip several attributes to over-estimation; AutoCalibrate leaves disability and immigration where they were or worse. No condition reduces the mean absolute per-attribute bias below the zero-shot value of 0.175.

5. **Practical reading.** With a small free judge and a few dozen labels, the best prompt is the human study's own instructions unchanged. Labels are better spent measuring the judge's per-attribute bias than on prompting, and a stronger judge (GPT-4o zero-shot: Spearman 0.496) buys more than any of the three methods did.

## 5. Limitations

- **Guardrail missingness.** AutoCalibrate's runs lose 5-21% of items each, concentrated on harmful ones. The shared-items comparison removes the confound but on a compressed label distribution (659 items) where every method looks weaker.
- **Grid coverage.** AutoCalibrate was evaluated at N = 5 and 10 only (two seeds at N = 10), so whether a learned prompt improves with more labels is untested above N = 10. Many-shot N = 50 has two seeds and ALLURE N = 50 one.
- **Reflection model = judge.** AutoCalibrate's criterion-induction step depends on the reflection model; a 17B model did not abstract from examples. A stronger reflection model is the obvious next experiment. GEPA also selected candidates on the same N items it learned from, which favours memorising prompts.
- **One judge, one pass, temperature 0.** The paper used three repetitions; single-pass ratings are noisier, and in-context learning may behave differently for a stronger model (the many-shot gains in Song et al. were reported for GPT-4o).
- **Subjective ground truth.** Medians of about eight UK-based raters (89.9% white); agreement means agreement with that panel. Half-integer medians (9%) stay in the test set but not in the demo pool. The realism label was not analysed.
- **ALLURE uses more supervision than N.** Its selection step compares the judge against all 527 demo-pool labels; only N reach the prompt.
- **Platform.** The free model's monthly quota and the guardrail limit reproducibility on this gateway; every run records model id, decoding parameters, failure and blocked-item counts.

**Reproduction.** `pip install -e ".[dev]" && pytest` (26 tests); `bash scripts/run_all.sh` for the grid, `bash scripts/run_missing.sh` to resume; `python -m biasjudge.analyze` for every table and plot.
