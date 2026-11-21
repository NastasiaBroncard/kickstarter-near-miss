# Step 3 · Model 4: Gradient boosting + blurb score (deep learning)

Model 4 is done. **It beats Model 3 on both metrics: AUC 0.862 vs 0.857 (both tuned without touching 2024), and the best coaching list so far, 4.9% near-miss hit rate, 2.4× random.**

A pre-trained language model reads every campaign's one-line pitch (the blurb) and turns it into numbers. A small "blurb grader" learns from 2016–18 which pitches tend to get funded and gives every later campaign a `blurb_score`. That one extra column goes into Model 3's gradient boosting. This page records each step and why, in the order we ran them.

## Input and output
- **Input:** `X_train` / `X_val` (191 columns), `slice_text` / `slice_train` / `slice_val` (blurbs and answers), `id` and `blurb` only from `slice_test`, plus `baseline_val`, `model2_val`, `model3_val` for the comparison.
- **Output:**
  - `model4_val.parquet`: 26,739 rows (`id`, `blurb_score`, `score_model4`, `success`, `near_miss`).
  - `blurb_scores.parquet`: `blurb_score` for every 2019–25 campaign, so the final 2025 run needs no re-embedding.
  - `model4.joblib`, `blurb_grader.joblib`: fitted models for the one-time test run.
  - `model4_grid.csv`: the 2023 inner-holdout tuning record for Model 3 and Model 4.

## ⚠️ Team rules followed
- Launch-day inputs only, no `post_` columns.
- The grader learns from **2016–18 only**, years Models 1–3 never see.
- From the 2025 file, **only `id` and `blurb` were loaded**. Its answers were never read.
- **Settings chosen on 2023, not 2024**, as Model 3's README recommended. 2024 influenced no choice, so the Model 4 numbers carry no tuning-luck caveat.

## 1. Turn every blurb into an embedding (deep learning)
`all-MiniLM-L6-v2`, a small pre-trained BERT-family sentence encoder (L11–12), maps each blurb to 384 numbers. Blurbs with similar meaning land close together even without shared words. The encoder is used **frozen**, not fine-tuned: L14's "frozen backbone + small head", with the grader below as the head.

273,153 blurbs embedded in about 4 minutes on an Apple-silicon GPU (text 71s, train 107s, val 27s, test 26s).

## 2. Train the blurb grader on 2016–18
A logistic regression on the 384 numbers, predicting `success`. Its one setting, `C`, was chosen inside the text years: fit on 2016–17, check on 2018.

| C | 0.01 | 0.1 | **1.0** | 10 |
|---|---|---|---|---|
| AUC on 2018 | 0.7351 | 0.7436 | **0.7442** | 0.7438 |

`C = 1.0`, then refit on all 100,240 text-year blurbs. As with Model 2, the choice barely matters.

## 3. Score every 2019–25 blurb

| Slice | Mean `blurb_score` | Actual success |
|---|---|---|
| 2016–18 | 54.2% | 54.2% |
| 2019–23 | 61.1% | 73.5% |
| 2024 | 64.9% | 72.2% |

Scores run low on later years, exactly as Step 2 predicted: the grader learned when only 54% succeeded. **Harmless here**, because boosting only uses the order of the scores and re-learns what each level means from 2019–23.

**The blurb alone reaches AUC 0.749 on 2024**, versus 0.807 for the category lookup. A one-line pitch read by a frozen model gets most of the way to knowing the category.

## 4. Is the blurb just repeating the category?
The real question is whether the pitch ranks campaigns *inside* a category. Model 1 scores 0.500 there by construction.

| Subcategory (2024) | Campaigns | Blurb alone | Model 3 |
|---|---|---|---|
| Games / Tabletop Games | 4,907 | **0.661** | 0.763 |
| Technology / Apps | 606 | 0.501 | 0.811 |
| Film & Video / Documentary | 328 | **0.615** | 0.712 |
| Publishing / Children's Books | 422 | 0.558 | 0.676 |
| other | 3,937 | **0.661** | 0.774 |

**The pitch carries real information beyond the category for games, documentaries and the `other` bucket, and none for apps.** It is weaker than Model 3's goal-and-timing signal inside every category, so the question is whether it adds to Model 3 rather than replaces it. Correlation with Model 3's score: 0.49, so partly overlapping, partly new.

## 5. Clean A/B: Model 3's exact settings, plus one column
Model 3 refit here reproduces the saved file exactly (AUC 0.8603, max score difference 2e-16), so both sides come from identical code.

| | AUC on 2024 |
|---|---|
| Model 3 (`max_depth=None`, lr 0.1, 100 trees) | 0.8603 |
| **Model 4, same settings + `blurb_score`** | **0.8631** |
| gain | **+0.0028** |

## 6. Honest tuning on a 2023 inner holdout (both models)
Fit on 2019–22, choose settings on 2023, refit on 2019–23, score 2024. Grid: `max_depth` ∈ {4, None} × 100/200/400/800 trees at lr 0.1. Both models pick the same settings: **depth 4, lr 0.1, 800 trees**.

| | AUC on 2024 |
|---|---|
| Model 3, tuned on 2023 | 0.8571 (matches Model 3 README §12) |
| **Model 4, tuned on 2023** | **0.8621** |
| gain | **+0.0050** |

**This is the headline comparison**: honest versus honest, identical settings, one extra column.

## 7. Is the gain real? (200 bootstrap resamples of 2024)

| Comparison | Mean | 95% CI | Model 4 better in |
|---|---|---|---|
| Same settings (A/B) | +0.0028 | [+0.0016, +0.0039] | 100% |
| **Both tuned on 2023** | **+0.0050** | **[+0.0035, +0.0062]** | **100%** |
| Model 4 (2023) vs Model 3 shipped (2024-tuned) | +0.0018 | [+0.0004, +0.0034] | 99% |

**Real and small.** Model 4 even edges out Model 3's shipped version, which had the advantage of being tuned on 2024.

## 8. How much does the model lean on the blurb?
Scrambling `blurb_score` costs **0.0175 AUC** (5 shuffles, range 0.0169–0.0180). That puts the blurb on par with campaign length (0.019) and prep time (0.022) in Model 3's ranking. The model uses it heavily, but the net gain is smaller because part of what the blurb says, mostly the category, is already known.

## 9. Coaching metrics (validation 2024, capacity 2,000, random = 2.02%)

| Model | AUC | Closest-to-50%, 2,000 calls | 40–60% bracket | Calibration |
|---|---|---|---|---|
| Model 1: lookup | 0.807 | 4.35% (2.15×) | 3.38% on 7,101 | +5.7 pts |
| Model 2: logistic | 0.853 | 4.25% (2.10×) | 4.21% on 2,542 | +6.4 pts |
| Model 3: shipped | 0.860 | 4.80% (2.37×) | 4.68% on 2,243 | +6.1 pts |
| Model 3: tuned on 2023 | 0.857 | 4.70% (2.32×) | 4.66% on 2,101 | +6.3 pts |
| **Model 4: + blurb** | **0.862** | **4.90% (2.42×)** | **4.97% on 1,972** | +6.7 pts |

"Closest to 50%" = the 2,000 campaigns the model is least sure about, a rule declared before looking. Model 1's figure is inflated by ties: its lookup has only 60 distinct scores.

**Model 4 builds the best coaching list on both rules**, and its 40–60% bracket (1,972) is almost exactly the 2,000-call capacity. **But read the size honestly:** at 2,000 calls, 4.90% vs 4.70% is 98 near-misses versus 94. The direction agrees with the AUC gain; the size is within what one year of data can distinguish, and we did not bootstrap it.

## 10. What a winning pitch looks like
Highest-scored 2024 blurbs are mostly games, miniatures and music, and many signal a **returning creator**: "the 2nd volume," "returns once more with all-new expansions," "a reprint," "30th anniversary." Lowest-scored are all Technology / Apps and read as early-stage ideas asking for money: "I am looking to create a social media app…," "We require start-up capital for application development."

**Caution for the slide:** the extremes partly reflect category (all bottom blurbs are apps, and §4 shows the blurb adds nothing *within* apps). The "returning creator" signal is our reading of the examples, not a tested result.

## Model 4 summary

| Metric (validation 2024) | Model 3 (tuned on 2023) | **Model 4** | |
|---|---|---|---|
| AUC | 0.857 | **0.862** | ✅ +0.005, CI [+0.0035, +0.0062] |
| Blurb grader alone | — | 0.749 | category lookup 0.807 |
| AUC lost if `blurb_score` scrambled | — | 0.018 | on par with duration and prep |
| Closest-to-50%, 2,000 calls | 4.70% | **4.90%** (2.42×) | ✅ small |
| 40–60% bracket | 4.66% (2,101) | **4.97% (1,972)** | ✅ right size |
| Calibration | +6.3 pts | +6.7 pts | ❌ slightly worse |

**Recommendation for Step 5: Model 4 is the winner.** Better AUC with a tight confidence interval, the best coaching list on both rules, and no tuning-on-2024 caveat. Run it once on 2025.

## What to say about it
1. **The pitch matters, a little.** "What you write adds a small, reliable improvement; what you make and how much you ask for matter far more."
2. **Where it helps:** games, documentaries and unusual categories. **Where it doesn't:** apps, where every pitch looks equally doomed.
3. **This is the deep-learning half of the project:** a pre-trained transformer on unstructured text, combined with machine learning on the tabular data.

## Limitations
- Frozen small encoder, not fine-tuned; a larger or fine-tuned model might extract more.
- Grader learned in 2016–18 (54% success), so its raw scores run low on later years. Harmless for trees, stated anyway.
- Blurb text only, not titles, images or videos.
- Coaching-list gains over Model 3 are about 4 near-misses per 2,000 calls and were not bootstrapped.
- Calibration +6.7 points optimistic, the worst of the five; matters for Step 5 if brackets are read as real probabilities.
- Inherits the `X / X` category artifact (Model 1 §6) and the `other` bucket.
- `COACH_CAPACITY = 2,000` is still a placeholder.

## How to reproduce
1. Put `[code] Model 4) Blurb Score + Boosting.ipynb` in a folder with `X_train`, `X_val`, the four `slice_*` files, `baseline_val`, `model2_val` and `model3_val` (all `.parquet`). In Colab, use `/content/split` and a T4 GPU.
2. Run all cells: about 12 minutes on an Apple-silicon Mac or a T4 GPU (embeddings ~4 min, inner tuning ~6 min). On CPU only, the embedding step takes 30–40 minutes. Embeddings are cached, so re-runs are fast.
3. Output: `model4_val.parquet`, `blurb_scores.parquet`, `model4_grid.csv`, `model4.joblib`, `blurb_grader.joblib`.
