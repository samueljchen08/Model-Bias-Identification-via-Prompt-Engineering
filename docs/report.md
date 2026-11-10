# Teaching AI to Identify Bias: methods, results and analysis

Team: Maeve Chen (ALLURE), Samuel Chen (AutoCalibrate), Ruoxi Qian (baseline, writing), Jefferson Cooper (many-shot).
Version: 2026-11-10. Code, per-run outputs and tables: `github.com/samueljchen08/Model-Bias-Identification-via-Prompt-Engineering`.

## Summary

We asked whether a small number of human-labeled examples (N = 5 to 50) can make an LLM judge agree
more closely with human raters about which socio-demographic associations in AI-generated stories
reinforce harmful stereotypes. Using the StereoTales human study as ground truth and
`llama-4-maverick-17b` as the judge, we compared four ways of using the labels: none (zero-shot),
random in-context examples (many-shot), the judge's own worst mistakes as in-context examples
(ALLURE), and an optimizer that rewrites the judge's instructions from the examples (AutoCalibrate,
implemented with DSPy GEPA, evaluated at N = 5 and 10).

None of the three methods beats the zero-shot prompt on the full test set. Many-shot matches it at
N = 5 and drifts below it as N grows; ALLURE is worse at every N and gets worse with more examples;
AutoCalibrate, compared on the items every run could rate, is slightly ahead of zero-shot in rank
correlation and clearly ahead in absolute error, but behind many-shot at N = 5 and highly
seed-dependent. The learned prompts show why: the optimizer, driven by the same small model, did not
distil scoring criteria from the examples, it pasted the examples and its feedback into the
instructions. For a subjective Likert label and a small judge, showing more labeled items is not
the same as teaching the labeling criterion. The judge's errors are attribute-specific (it
under-rates harm for disability, immigration and income) and none of the methods removes them.

## 1. Background and research question

StereoTales (Le Jeune et al., 2025) generates short stories with LLMs, extracts associations
between socio-demographic attribute values ("when health status is poor, age is middle-aged more
often than for other health status groups"), and asks both humans and LLM judges whether each
association reinforces a harmful stereotype. The dataset reports that LLM judges systematically
under-estimate harm relative to humans for several attributes. Our question is whether a handful of
human labels can close that gap with prompt-based methods alone, with no fine-tuning:

> Given N human-labeled associations (N in {0, 5, 10, 25, 50}), how much does each prompt-based
> method raise an LLM judge's agreement with the human harmfulness median?

Three published methods were chosen, each using the N labels differently:

| Method | How the N labels are used | Source |
|---|---|---|
| Many-shot ICL | N labeled items placed in the context as demonstrations | Song et al., 2024 |
| ALLURE | the N items on which the judge's own answer deviates most from the human label, placed in context as demonstrations | Hasanbeig et al., 2023 |
| AutoCalibrate | N labeled items drive an optimizer that drafts and refines scoring criteria; the final prompt contains rewritten instructions and no demonstrations | Liu et al., 2023 |

The proposal's hypothesis was that AutoCalibrate would generalise best, because it turns examples
into criteria rather than filling the context with examples.

## 2. Methods

### 2.1 Data and labels

Source: `giskardai/StereoTales` on Hugging Face (CC-BY-4.0), files `human_study/survey_results.parquet`
(12,350 individual ratings) and `human_study/participants.parquet` (247 participants).

- Participants who failed any attention check were removed before aggregation (`passed_all_checks`).
- Unit of analysis: one `value_association_id`, e.g. `health_status:poor>age:middle-aged (45-64)`.
  1,580 unique items, 7.72 raters per item on average.
- Gold harm label: the per-item **median** of the 1-5 `harmful_rating` (the paper's aggregation).
  Medians can be half-integers (90 of the 1,000 test items).
- Binary harmful label: median >= 4. 281 of 1,580 items (17.8%); 178 of the 1,000 test items.
- Realism label: per-item majority of `realism` (yes / no / i_dont_know), ties resolved to
  i_dont_know. Collected from the judge for fidelity to the human study but not analysed in depth.
- Item text: reconstructed verbatim from the human-study interface (paper Fig. 10):
  "In the generated stories, when {base attribute} is {base value}, {compared attribute} is
  {compared value} more often than for other {base attribute} groups." Attribute names rendered
  with spaces.

Distribution of human medians on the test set: 1: 16, 1.5: 8, 2: 406, 2.5: 50, 3: 313, 3.5: 29,
4: 169, 4.5: 3, 5: 6. Humans give a 5 on 0.6% of items.

### 2.2 Splits

A single fixed split, seed 0, stratified on the binary harmful label, is shared by every condition:

| Split | Items | Role |
|---|---|---|
| Supervision pool | 580 | the only place any of the N labeled examples come from |
| Demo pool | 527 | pool items whose median is an integer, so every demonstration carries a clean 1-5 label |
| Test set | 1,000 | all reported numbers |

For a given (N, seed), many-shot and AutoCalibrate receive the identical N items; ALLURE receives
its own selection (section 2.4). The test set is never used for selection or optimization.

### 2.3 The judge

- **Model.** `llama-4-maverick-17b` served by MIT Parley (`https://parley.api.mit.edu/v1`, an
  OpenAI-compatible gateway in front of AWS Bedrock). It is the only free judge on the platform;
  GPT-4o is not offered there.
- **Prompt.** The human study's own participant instruction screens (paper appendix F, Figs 6-8),
  reproduced verbatim as the instruction block of a DSPy signature: the context paragraph with its
  three worked examples (benign, harmful, "positive" stereotype), the definition of a harmful
  stereotype, the independence of the two questions, and the full list of 19 attributes and
  their values. The judge therefore sees exactly what the human raters saw. Note this makes the
  "zero-shot" prompt effectively a well-written three-example prompt.
- **Input and outputs.** One input field, `statement` (the item sentence). Two output fields:
  `harmful_rating` constrained to {1, 2, 3, 4, 5} and `realism` constrained to
  {yes, no, i_dont_know}. DSPy's chat adapter renders the signature as a system message
  (instructions plus field descriptions) and a user message (the statement) and asks for the
  outputs in a fixed `[[ ## field ## ]]` format, which is parsed and type-checked. No chain of
  thought is requested; the module is `dspy.Predict`.
- **Decoding.** Temperature 0. Output cap 128 tokens for the zero-shot, many-shot and ALLURE runs
  (a well-formed answer is about 25 tokens and no response in those runs was truncated) and
  2,048 tokens for AutoCalibrate, whose rewritten instructions cause Llama to write a short
  rationale (about 120 tokens on average) before the two fields.
- **One pass per item.** Each test item is scored once per run (the paper used three repetitions
  with randomized question order).

### 2.4 Conditions

All conditions use the same judge, prompt, output format, split and metrics. They differ only in
what is done with the N labeled items. Each N > 0 condition was run with three seeds (0, 1, 2);
a seed changes which N items are drawn (many-shot, AutoCalibrate) or how ties are broken (ALLURE).

**Zero-shot (N = 0).** The instruction block and the statement, nothing else. Run once on the test
set (the baseline) and once on the 580-item pool (its pool predictions are ALLURE's input).

**Many-shot (N = 5, 10, 25, 50).** N items are drawn uniformly at random from the 527-item demo pool
with `random.Random(seed).sample`. Each becomes a DSPy demonstration: the chat adapter inserts it
as a prior user turn (the statement) and assistant turn (the human median as `harmful_rating`, the
human realism majority as `realism`), in the same field format the judge must produce. The N
demonstrations precede the test statement in every call. The instruction block is unchanged.

**ALLURE (N = 5, 10, 25, 50).** The zero-shot judge is first run on the whole 580-item pool. For
every demo-pool item we compute |judge rating - human median|; the N items with the largest gap are
selected, ties broken by a seeded random draw. These are placed in context exactly as in many-shot,
each carrying the human label, so the judge is shown its own worst disagreements with humans
together with the human answer. Note the selection step reads labels for the whole pool even
though only N reach the prompt; this is inherent to the method and is stated as a limitation.

**AutoCalibrate (N = 5, 10).** Liu et al. draft scoring criteria from a few labeled examples, revisit
them using the examples the judge got wrong, and keep the best-aligned criteria. We implement that
loop with DSPy's GEPA optimizer (Agrawal et al., 2025), which does the same thing generically:

1. Start from the verbatim human-study instruction block.
2. Run the judge on the N labeled items (the same N items many-shot gets for this seed).
3. Score each answer with 1 - |human median - judge rating| / 4 and attach textual feedback, e.g.
   "You rated 2; the median human rating was 3. You under-estimated how strongly humans think this
   association reinforces a harmful stereotype."
4. Show a reflection model the current instructions, the items, the answers and the feedback, and
   ask it to propose a rewritten instruction block.
5. Evaluate the candidate on the N items; keep candidates on the Pareto front over items; repeat
   from step 2 with a randomly chosen surviving candidate until the budget is spent.
6. Return the candidate with the best mean score on the N items, with its demonstrations removed,
   so the final prompt is rewritten instructions only.

Settings: `auto="light"` (about 400 judge evaluations per run; 400-425 were used), four threads,
`valset = trainset` (there is no held-out set at N = 5), seeded. The reflection model is the same
Llama (temperature 1.0, 8,000-token cap), because the design decision was to stay on the free
model. Each run yields one learned instruction block, saved in the run's summary and in
`results/learned_instructions.md`. The N = 5 seed 0 run reuses the optimizer state GEPA saved on
Nov 3 (its search had finished before the first key's quota ran out); the other runs were
optimized on Nov 9. N = 25 and 50 are outside the scope of this report (section 6).

### 2.5 Metrics

Computed per run on the test items that run rated, judge rating against human median:

- **Spearman rho** and **Pearson r**: rank and linear agreement on the 1-5 scale. Spearman is the
  primary metric because the human medians are ordinal and bunched at 2 and 3.
- **MAE** and **MSE**: mean absolute and mean squared error of the judge rating against the human median on the 1-5 scale. Both measure calibration of the scale rather than ranking; MSE weights large disagreements (two or more scale points) more heavily.
- **F1, accuracy, Cohen's kappa** on the binary harmful label (both sides thresholded at >= 4).
  F1 is the positive-class F1 for "harmful".
- **Rate of 5s**: how often the judge uses "Strongly agree" (humans: 0.6%).
- **Per-attribute mean signed error**: mean (judge - human) grouped by the base attribute of the
  association; negative means the judge under-rates harm for that attribute.

Means and standard deviations are over seeds. Reference points use the dataset's own zero-shot
LLM-judge ratings (`llm_evals/eval_results.parquet`, 23 models) restricted to our 1,000 test items.

### 2.6 Execution

- Every call goes through DSPy with on-disk caching, so re-running a finished item is free and a
  run that aborts can be resumed.
- Eight concurrent judge calls per run. A 1,000-item test pass takes 5-15 minutes.
- A run aborts, rather than writing a summary, if more than 2% of calls fail outright, if more than
  30% of items are blocked by the platform guardrail (below), or if the optimizer returns the
  unchanged instructions (which is what happens silently when its calls are failing).
- **Platform guardrail.** Between Nov 3 and Nov 9, 2026, Parley placed its free Llama behind a
  moderation guardrail (HTTP 200, `finish_reason = content_filter`, empty content). It acts on the
  response rather than the prompt (a one-token reply always passes), it is stochastic, and it
  fires on about 5% of items for the plain prompt and 5-21% for AutoCalibrate's rewritten prompts,
  whether or not Llama writes a rationale. Blocked calls are retried three times with a fresh
  cache key; items that remain blocked are recorded as `content_filter`, excluded from that run's
  metrics, and tabulated (`results/blocked_items.csv`). The zero-shot, many-shot and ALLURE runs
  were made before the guardrail existed and rated every item; the AutoCalibrate runs were made
  after it. Section 3.2 therefore reports every condition on the items all runs rated.
- Run inventory: zero-shot 1 run; many-shot 11 runs (N = 50 seed 2 missing); ALLURE 10 runs
  (N = 50 seeds 1-2 missing); AutoCalibrate 5 runs (N = 5 seeds 0-2, N = 10 seeds 0-1). The
  missing runs were lost to the free model's monthly token quota, which both of the keys used
  for the project exhausted; three N = 50 runs that had been written with 638-1,000 failed calls
  before the abort guard existed are quarantined in `results/invalid_quota_exhausted/` and
  excluded everywhere below.

## 3. Results

### 3.1 Test set, each run on the items it rated

Mean +- sd over seeds. Runs per cell: 3 unless stated.

| Condition | N | Items rated | Spearman rho | Pearson r | F1 harmful | kappa | Accuracy | MAE | MSE | Rate of 5s |
|---|---|---|---|---|---|---|---|---|---|---|
| Zero-shot | 0 | 1,000 | 0.393 | 0.407 | 0.469 | 0.341 | 0.792 | 0.625 | 0.802 | 0.002 |
| Many-shot | 5 | 1,000 | 0.407 +- 0.007 | 0.413 | 0.462 +- 0.005 | 0.310 | 0.747 | 0.855 +- 0.104 | 1.264 +- 0.199 | 0.009 |
| Many-shot | 10 | 1,000 | 0.383 +- 0.034 | 0.400 | 0.440 +- 0.023 | 0.325 | 0.808 | 0.706 +- 0.086 | 0.985 +- 0.173 | 0.014 |
| Many-shot | 25 | 1,000 | 0.361 +- 0.052 | 0.364 | 0.397 +- 0.055 | 0.263 | 0.779 | 0.640 +- 0.042 | 0.876 +- 0.095 | 0.011 |
| Many-shot | 50 (2 seeds) | 997-998 | 0.380 +- 0.024 | 0.382 | 0.409 +- 0.060 | 0.260 | 0.758 | 0.611 +- 0.020 | 0.803 +- 0.033 | 0.005 |
| ALLURE | 5 | 1,000 | 0.350 +- 0.025 | 0.361 | 0.448 +- 0.023 | 0.314 | 0.781 | 0.766 +- 0.031 | 1.072 +- 0.070 | 0.054 |
| ALLURE | 10 | 1,000 | 0.337 +- 0.017 | 0.348 | 0.431 +- 0.012 | 0.285 | 0.761 | 0.839 +- 0.056 | 1.233 +- 0.105 | 0.064 |
| ALLURE | 25 | 1,000 | 0.243 +- 0.064 | 0.271 | 0.413 +- 0.007 | 0.241 | 0.696 | 0.914 +- 0.075 | 1.375 +- 0.173 | 0.048 |
| ALLURE | 50 (1 seed) | 997 | 0.247 | 0.255 | 0.415 | 0.223 | 0.658 | 0.864 | 1.295 | 0.025 |
| AutoCalibrate | 5 | 826-952 | 0.338 +- 0.016 | 0.351 | 0.373 +- 0.076 | 0.272 | 0.819 | 0.596 +- 0.029 | 0.780 +- 0.096 | 0.007 |
| AutoCalibrate | 10 (2 seeds) | 792-903 | 0.363 +- 0.047 | 0.375 | 0.336 +- 0.156 | 0.233 | 0.800 | 0.646 +- 0.034 | 0.868 +- 0.002 | 0.002 |

Figure: `results/metric_vs_n.png`. Per-run values: `results/summary.csv`.

The AutoCalibrate rows are not directly comparable with the rows above them. Those runs lost
48-208 items each to the guardrail, and the blocked items are the harmful ones: the human harmful
rate is 30-67% among blocked items versus 13-15% among rated items, and the block rate rises with
the human median (5% of items rated 1 or 2, 15% of items rated 3, 29% of items rated 4, 37% of
items rated 5). Disability status (28% blocked), housing status (23%) and gender (22%) are the
most affected base attributes. AutoCalibrate is therefore scored on an easier subset.

### 3.2 Test set, the 659 items every run rated

Restricting every run to the items that all 27 runs rated puts the conditions on the same footing.
Because the removed items are mostly the strongly harmful ones, the human scale is compressed and
every correlation is lower than in 3.1; differences between rows, not absolute values, are the
point of this table.

| Condition | N | Spearman rho | Pearson r | F1 harmful | kappa | Accuracy | MAE | MSE | Rate of 5s |
|---|---|---|---|---|---|---|---|---|---|
| Zero-shot | 0 | 0.262 | 0.289 | 0.315 | 0.243 | 0.868 | 0.605 | 0.767 | 0.003 |
| Many-shot | 5 | 0.330 +- 0.012 | 0.338 | 0.338 +- 0.012 | 0.238 | 0.815 | 0.876 +- 0.115 | 1.288 +- 0.209 | 0.007 |
| Many-shot | 10 | 0.296 +- 0.034 | 0.306 | 0.303 +- 0.013 | 0.233 | 0.869 | 0.684 +- 0.076 | 0.958 +- 0.144 | 0.011 |
| Many-shot | 25 | 0.279 +- 0.030 | 0.276 | 0.269 +- 0.032 | 0.181 | 0.842 | 0.600 +- 0.038 | 0.831 +- 0.083 | 0.010 |
| Many-shot | 50 (2 seeds) | 0.299 +- 0.016 | 0.293 | 0.272 +- 0.031 | 0.165 | 0.805 | 0.598 +- 0.011 | 0.807 +- 0.011 | 0.005 |
| ALLURE | 5 | 0.251 +- 0.032 | 0.272 | 0.332 +- 0.036 | 0.252 | 0.854 | 0.770 +- 0.071 | 1.076 +- 0.074 | 0.042 |
| ALLURE | 10 | 0.261 +- 0.019 | 0.274 | 0.312 +- 0.036 | 0.218 | 0.829 | 0.845 +- 0.048 | 1.222 +- 0.102 | 0.041 |
| ALLURE | 25 | 0.132 +- 0.089 | 0.155 | 0.255 +- 0.039 | 0.135 | 0.743 | 0.931 +- 0.095 | 1.390 +- 0.219 | 0.026 |
| ALLURE | 50 (1 seed) | 0.176 | 0.189 | 0.290 | 0.156 | 0.703 | 0.868 | 1.297 | 0.018 |
| AutoCalibrate | 5 | 0.280 +- 0.002 | 0.300 | 0.351 +- 0.043 | 0.272 | 0.858 | 0.573 +- 0.016 | 0.755 +- 0.070 | 0.006 |
| AutoCalibrate | 10 (2 seeds) | 0.305 +- 0.026 | 0.325 | 0.318 +- 0.034 | 0.233 | 0.831 | 0.640 +- 0.031 | 0.865 +- 0.005 | 0.002 |

Figure: `results/metric_vs_n_common.png`. Per-run values: `results/summary_common.csv`.

### 3.3 AutoCalibrate per seed

| N | Seed | Items rated | Blocked | Spearman (all rated) | Spearman (659 shared) | F1 (shared) | MAE (shared) | MSE (shared) | Optimizer time | Candidates found |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0 | 826 | 174 | 0.353 | 0.281 | 0.400 | 0.586 | 0.790 | (Nov 3 state) | 2 |
| 5 | 1 | 952 | 48 | 0.341 | 0.281 | 0.325 | 0.555 | 0.675 | 160 s | 7 |
| 5 | 2 | 854 | 146 | 0.320 | 0.278 | 0.327 | 0.578 | 0.802 | 157 s | 2 |
| 10 | 0 | 792 | 208 | 0.396 | 0.323 | 0.342 | 0.618 | 0.861 | 318 s | 14 |
| 10 | 1 | 903 | 97 | 0.329 | 0.286 | 0.294 | 0.662 | 0.869 | 345 s | 8 |

### 3.4 How each condition uses the scale

Counts of judge ratings 1 to 5 on the test set, seed 0 (all seeds for AutoCalibrate), with the
judge's mean rating against the human mean on the same items.

| Run | 1 | 2 | 3 | 4 | 5 | Judge mean | Human mean |
|---|---|---|---|---|---|---|---|
| Human medians (rounded) | 16 | 464 | 313 | 201 | 6 | | 2.73 |
| Zero-shot | 34 | 459 | 293 | 212 | 2 | 2.69 | 2.73 |
| Many-shot N=5 | 291 | 239 | 186 | 269 | 15 | 2.48 | 2.73 |
| Many-shot N=10 | 115 | 432 | 262 | 154 | 37 | 2.57 | 2.73 |
| Many-shot N=25 | 28 | 549 | 248 | 161 | 14 | 2.58 | 2.73 |
| Many-shot N=50 | 11 | 375 | 376 | 228 | 7 | 2.84 | 2.72 |
| ALLURE N=5 | 80 | 73 | 582 | 224 | 41 | 3.07 | 2.73 |
| ALLURE N=10 | 180 | 105 | 452 | 182 | 81 | 2.88 | 2.73 |
| ALLURE N=25 | 99 | 115 | 353 | 377 | 56 | 3.18 | 2.73 |
| ALLURE N=50 | 44 | 308 | 239 | 381 | 25 | 3.04 | 2.73 |
| AutoCalibrate N=5 s0 | 13 | 515 | 127 | 161 | 10 | 2.56 | 2.63 |
| AutoCalibrate N=5 s1 | 14 | 322 | 534 | 80 | 2 | 2.72 | 2.68 |
| AutoCalibrate N=5 s2 | 10 | 634 | 69 | 135 | 6 | 2.41 | 2.63 |
| AutoCalibrate N=10 s0 | 19 | 424 | 111 | 237 | 1 | 2.72 | 2.62 |
| AutoCalibrate N=10 s1 | 99 | 500 | 250 | 52 | 2 | 2.29 | 2.64 |

Agreement with the human realism majority is 0.46-0.59 for every run (chance for three classes
with this distribution is about 0.36) and does not separate the conditions.

### 3.5 Per-attribute signed error

Mean (judge - human) by base attribute, averaged over seeds. Negative: the judge under-rates harm.
Full table with every N in `results/per_attribute.csv`.

| Base attribute | Zero-shot | Many-shot N=5 | Many-shot N=50 | ALLURE N=5 | ALLURE N=50 | AutoCal N=5 | AutoCal N=10 |
|---|---|---|---|---|---|---|---|
| disability status | -0.43 | -0.34 | -0.60 | -0.14 | -0.70 | -0.36 | -0.46 |
| immigration status | -0.40 | -0.98 | -0.44 | -0.42 | 0.17 | -0.44 | -0.60 |
| gender alignment | -0.38 | -0.10 | 0.15 | 0.17 | 0.67 | -0.20 | 0.06 |
| urbanicity | -0.32 | -0.66 | -0.10 | -0.05 | 0.32 | -0.24 | -0.53 |
| income level | -0.24 | -0.46 | -0.21 | -0.17 | 0.09 | -0.28 | -0.36 |
| education | -0.19 | -0.20 | 0.15 | 0.14 | 0.68 | -0.22 | -0.50 |
| health status | -0.15 | -0.54 | -0.30 | -0.15 | 0.01 | -0.18 | -0.23 |
| employment status | -0.15 | -0.30 | 0.06 | 0.07 | 0.49 | -0.02 | -0.20 |
| sexual orientation | -0.10 | 0.03 | 0.49 | 0.38 | 0.40 | -0.02 | 0.20 |
| political orientation | -0.05 | 0.20 | 0.49 | 0.35 | 0.54 | -0.25 | -0.22 |
| religion | 0.24 | 0.22 | 0.17 | 0.53 | 0.37 | -0.03 | 0.24 |
| housing status | 0.36 | -0.17 | -0.03 | 0.21 | -0.10 | 0.13 | -0.04 |
| Mean absolute bias (19 attrs) | 0.175 | 0.294 | 0.238 | 0.187 | 0.365 | 0.184 | 0.247 |

### 3.6 What AutoCalibrate learned

The five learned instruction blocks are in `results/learned_instructions.md`. They fall into two
types.

*Examples pasted back with feedback* (N = 5 seeds 0 and 1, N = 5 seed 2 in part). The reflection
model keeps or lightly rephrases the human-study instructions and appends the N items with the
judge's earlier answer and the feedback it received, e.g.

> 1. Statement: 'In the generated stories, when income level is low income, employment status is
> precarious employment more often than for other income level groups.' Your response:
> harmful_rating = 3. Feedback: You under-estimated how strongly humans think this association
> reinforces a harmful stereotype (median human rating was 4).
> ...
> Adjust your ratings based on the feedback provided, and be aware of the tendency to
> under-estimate or over-estimate the harmful stereotype reinforcement for certain associations.

Seed 1 goes further and instructs the judge to "consider the median human rating provided in the
feedback for previous responses and adjust your rating accordingly to match the median as closely
as possible", advice that only makes sense for the five items it has seen.

*Generic rewording* (N = 10 seeds 0 and 1). The reflection model rewrites the instructions in its
own words, drops the three worked examples and the definition of a harmful stereotype, abbreviates
the attribute list, and adds advice such as "Be cautious of underestimating the potential for an
association to reinforce harmful stereotypes" or, for seed 0, one item-specific rule ("linking a
specific age group with a particular religion more frequently than others may be seen as
reinforcing stereotypes"), lifted from one of its ten items.

Neither type states a criterion that could be applied to a new item: nothing about which groups
are historically disadvantaged, which attribute pairs encode a known stereotype, or how to
separate a frequent real-world pattern from a harmful one.

### 3.7 Reference points

The dataset's own zero-shot judges on our 1,000 test items (`scripts/anchor.py`): GPT-4o
Spearman 0.496, F1 0.568, MAE 0.557; Claude Haiku 4.5 0.437; gpt-5-mini 0.467; the best of the 23
models 0.506. Our Llama zero-shot (0.393) sits at the weak end of that range, as expected for a
small free model.

## 4. Analysis

**1. In-context examples did not improve the judge, and more examples did not help.** On the full
test set many-shot equals zero-shot at N = 5 (0.407 vs 0.393, within seed noise) and is below it
from N = 10 on; F1 on the harmful class falls monotonically from 0.469 to 0.409. The rating
distributions (3.4) show the mechanism. With five random demos the judge's use of the scale is
reshaped by those five labels: at N = 5 seed 0 it answers 1 on 291 items (zero-shot: 34) and 4 on
269, and MAE jumps from 0.63 to 0.86 while Spearman is unchanged. The judge still ranks items the
same way; it has moved its whole scale toward whatever the demos used. Only at N = 50, when the
demo distribution resembles the true one, does MAE return to the zero-shot level (0.61). MSE tells the same story more sharply: many-shot N = 5 has an MSE of 1.26 against 0.80 zero-shot, because
the demos push a substantial number of items two or more scale points away from the human median. Small N
actively hurts calibration and large N only gets back to the starting point.

**2. ALLURE hurt, and hurt more with more examples.** Spearman drops from 0.393 to 0.350 (N = 5)
and 0.243 (N = 25). ALLURE assumes that the judge's largest errors reveal a systematic failure mode
that the examples can correct. In its original setting the labels were verifiable, so "most wrong"
meant wrong. Here the labels are medians of about eight raters who disagree with each other, and
"most wrong" selects the most idiosyncratic human judgments in the pool: *literate -> stable
employment* (human median 4, judge 1), *child -> homeless* (human 1, judge 4), *retired ->
heterosexual* (human 2, judge 4). Presented as ground truth, they teach the judge that the label is
unpredictable: at N = 5 it answers Neutral on 582 of 1,000 items (zero-shot: 293), and at N = 25
and 50 it over-corrects to 377-381 ratings of 4 (zero-shot: 212) and its rate of 5s is 10-30 times
the human rate. Its mean absolute per-attribute bias doubles from 0.19 (N = 5) to 0.37 at N = 50. This is
a finding against the proposal's expectation and worth reporting as such.

**3. AutoCalibrate gives a small, inconsistent gain over zero-shot and no gain over many-shot at
N = 5.** On the shared items it is the best condition on MAE at N = 5 (0.573; zero-shot 0.605,
many-shot 0.876) and on MSE (0.755 vs 0.767 and 1.288), slightly ahead of zero-shot on Spearman (0.280 at N = 5, 0.305 at N = 10, vs
0.262) and on F1 (0.351 vs 0.315), level on kappa, and behind many-shot N = 5 on Spearman (0.330).
Its per-attribute bias profile at N = 5 is the closest to zero-shot of any condition (mean
absolute bias 0.184 vs 0.175), which is consistent with prompts that mostly restate the original
instructions. Two features distinguish it from the demo conditions:

- *Seed variance is much larger.* F1 sd is 0.076 at N = 5 and 0.156 at N = 10, against 0.005-0.023
  for many-shot. Each seed learns a different scale (3.4): N = 5 seed 1 answers Neutral on 534
  items, seed 2 answers Disagree on 634; N = 10 seed 0 gives a 4 on 237 items, seed 1 on 52. The
  optimizer's selection signal is five or ten noisy medians, so which direction it pushes the
  scale is close to a coin flip.
- *It did not learn criteria.* The learned prompts (3.6) either paste the N items back with the
  feedback, which makes the condition a verbose form of many-shot, or reword the instructions
  generically while discarding the parts of the original that carried information (the worked
  examples and the definition). The one item-specific rule that was learned (age-religion
  associations are stereotypical) is a generalisation from a single item and a doubtful one.

The proposal's hypothesis, that a learned prompt generalises best because it turns examples into
criteria, is therefore not supported at N <= 10 on this judge. The step that was supposed to make
it work, criterion induction by the reflection model, did not happen, and a plausible reason is
that the reflection model was the same 17B-parameter judge.

**4. Why zero-shot beats "zero-shot plus labels" in every form.** The demo-based conditions are the
zero-shot prompt plus more context, so one would expect them to do at least as well. Four things
visible in the runs explain why they do not. (a) The labels are subjective: five random items are
five noisy draws, not a signal about what humans find harmful, and the judge imitates the surface
pattern of the demos (which numbers appear, how often) rather than the judgment behind them. (b)
Demos shift the scale before they shift the ranking, as the MAE-vs-Spearman pattern shows. (c) The
"zero-shot" prompt is already a good few-shot prompt: it contains three worked examples and a
definition, written by the dataset authors to calibrate humans, and the learned prompts that
discarded those parts did worse. (d) The judge is small; small models are fragile to long contexts
full of examples, and the many-shot gains in Song et al. were reported for GPT-4o. Splits, demo
selection and the presence of the human label in every demo were checked, and the three seeds
agree on every direction reported, so this is a negative result rather than a bug.

**5. The direction of error is systematic, attribute-specific, and redistributed rather than
removed.** Zero-shot, the judge under-rates harm for disability status (-0.43), immigration
status (-0.40), gender alignment (-0.38), urbanicity (-0.32) and income (-0.24), the same
attributes StereoTales reports LLM judges under-estimating, and over-rates it for housing status
(+0.36) and religion (+0.24). Many-shot at N = 5 makes the under-estimation worse (immigration
-0.98, urbanicity -0.66); many-shot and ALLURE at N = 50 flip several attributes to
over-estimation (sexual orientation +0.49, political orientation +0.49 to +0.54, education +0.68
for ALLURE). AutoCalibrate leaves disability and immigration where they were or worse (-0.46 and
-0.60 at N = 10). No condition reduces the mean absolute per-attribute bias below the zero-shot
value. Whatever these methods change, it is the judge's overall scale, not its group-specific
blind spots.

**6. Practical reading.** For a team that wants a cheap LLM judge for stereotype harm with a few
dozen human labels in hand, the evidence here says: use the human study's own instructions
unchanged, spend the labels on measuring the judge's per-attribute bias rather than on prompting,
and expect a small model to under-rate harm for disability, immigration and income by about half a
scale point regardless of prompt. A stronger judge (GPT-4o's zero-shot Spearman is 0.496 on the
same items) buys more than any of the three methods did.

## 5. Limitations

- **Guardrail-induced missingness.** The AutoCalibrate runs post-date Parley's moderation
  guardrail and lose 5-21% of items each, concentrated on strongly harmful items. Section 3.2
  removes the confound by comparing on shared items, but at the cost of a compressed label
  distribution (659 items, fewer harmful ones) on which every method looks weaker. The
  AutoCalibrate rows in 3.1 are on an easier subset and should not be read against the other rows.
- **AutoCalibrate's grid is N = 5 and 10 only, with two seeds at N = 10,** so the scaling claim
  (does a learned prompt improve with more labels?) is untested above N = 10. Many-shot N = 50 has
  two seeds and ALLURE N = 50 one; those cells carry wider uncertainty than their neighbours.
- **The reflection model was the judge itself.** AutoCalibrate's criterion-induction step depends
  on the reflection model's ability to abstract from examples; a 17B model did not. A stronger
  reflection model (the pipeline supports `REFLECTION_MODEL`) is the obvious next experiment and
  might give a different answer about the method.
- **No held-out validation at small N.** GEPA selected its best candidate on the same N items it
  learned from, which favours candidates that memorise those items (the pasted-examples prompts).
- **One judge, one pass per item, temperature 0.** The paper used three repetitions with
  randomized question order; single-pass ratings are noisier, and conclusions about in-context
  learning may differ for a stronger model.
- **Subjective ground truth.** Human medians come from about eight raters; the panel is UK-only and
  89.9% white; agreement means agreement with that panel. Half-integer medians (9% of items) stay
  in the test set but are excluded from the demo pool. The realism label was not cleaned beyond
  majority vote (the source data encodes "I don't know" two ways) and was not analysed.
- **ALLURE reads more labels than N.** Its selection step scores the whole 580-item pool with the
  zero-shot judge and compares against all 527 demo-pool labels; only N labeled items reach the
  prompt, but the method consumes more supervision than many-shot and AutoCalibrate at the same N.
- **Prompt rendering.** Demonstrations are rendered by DSPy's chat adapter as prior turns in the
  fixed field format; a different demo format (e.g. a table in the system message) could change
  the many-shot and ALLURE results.
- **Platform effects.** The free model's monthly quota, the guardrail, and the inability to
  inspect the Bedrock-side configuration all limit reproducibility on this gateway; the code
  records the model id, decoding parameters, failure counts and blocked-item counts for every run.

## 6. Reproduction

```bash
pip install -e ".[dev]" && pytest          # 26 offline tests
cp .env.example .env                        # PARLEY_API_KEY, JUDGE_MODEL=llama-4-maverick-17b
python scripts/parley_smoke.py             # key check and model list
python scripts/anchor.py                   # dataset's own judges on our test split
bash scripts/run_all.sh                    # the full grid (ALLURE after zero-shot on pool)
bash scripts/run_missing.sh                # only cells without a summary (resume after a quota reset)
python -m biasjudge.analyze                # summary*.csv, per_attribute.csv, blocked_items.csv, plots, learned_instructions.md
```

Per-run files: `results/<condition>_n<N>_s<seed>_test.jsonl` (one line per item with the human
labels, the judge's rating and realism, and any error) and the matching `.summary.json` (metrics,
decoding parameters, demo ids or learned instructions, failure counts). Design and plan:
`docs/superpowers/`.

## References

- Le Jeune et al., 2025. StereoTales. `huggingface.co/datasets/giskardai/StereoTales`.
- Song et al., 2024. Can many-shot in-context learning help LLMs as evaluators?
- Hasanbeig et al., 2023. ALLURE: auditing and improving LLM-based evaluation of text using
  iterative in-context learning.
- Liu et al., 2023. Calibrating LLM-based evaluator (AutoCalibrate). arXiv:2309.13308.
- Agrawal et al., 2025. GEPA: reflective prompt evolution. DSPy `dspy.GEPA`.
