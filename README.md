# Finding the campaigns worth saving

Predicting Kickstarter **near misses** on launch day, and turning that prediction into an outreach list a platform team can actually run.

A near miss is a campaign that **failed after raising 50–99% of its goal**. Kickstarter is all-or-nothing, so those creators receive nothing despite finding real backers. There were 410 of them in 2025. The business question is not "which campaigns will fail" but "which failures were close enough that an intervention could have flipped them."

**Headline result:** ranking campaigns by predicted failure is *worse than random* for this task. Our rule contacts the campaigns the model is least certain about, and on a held-out 2025 it found **61 near misses against 31 expected from random contact (1.97×, 95% CI 1.53–2.45×)**.

---

## Data

| | |
|---|---|
| Source | Monthly Kickstarter snapshots from [Web Robots](https://webrobots.io/kickstarter-datasets/) — 87 CSV + 40 JSON files |
| After cleaning | 273,163 campaigns, 2016–2025, 22 columns, no duplicate ids |
| Model inputs | 191 columns, all known on launch day (goal, category, location, timing, name/blurb length) |
| Labels | `success` (reached goal) and `near_miss` (failed at 50–99% of goal) |

Raw and intermediate data are **not committed** (the cleaned file alone is ~75 MB). `notebooks/01_data_cleaning.ipynb` downloads the snapshots and rebuilds everything.

**Hard rule enforced throughout:** columns prefixed `post_` describe what happened after launch and are never model inputs.

## Split

Time-based, never random, so no model learns from its own future. The four slices share zero campaign ids.

| Slice | Years | Rows | Purpose |
|---|---|---|---|
| Text | 2016–2018 | 100,240 | Blurb grader only (Model 4) |
| Train | 2019–2023 | 119,297 | Models 1–4 |
| Validation | 2024 | 26,739 | Model selection, tuning, cutoff |
| Test | 2025 | 26,887 | **Run once**, at the end |

## Models

| | Approach | AUC (2024) |
|---|---|---|
| 1 | Category success-rate lookup (benchmark) | 0.807 |
| 2 | Logistic regression, scaler fit on train only | 0.853 |
| 3 | `HistGradientBoostingClassifier`, tuned on a 2023 inner holdout | 0.857 |
| 4 | **Model 3 + `blurb_score`** | **0.862** |

Model 4 is the deep-learning component: `all-MiniLM-L6-v2` (frozen) embeds every blurb into 384 dims; a logistic "grader" trained **only on 2016–18** maps that to a `blurb_score`; that single column joins the 191 tabular inputs. Final settings `max_depth=4, lr=0.1, 800 trees`.

The gain is real but small: **+0.0050 AUC, 95% CI [+0.0035, +0.0062]**, Model 4 ahead in 100% of 200 bootstrap resamples. Scrambling `blurb_score` costs 0.0175 AUC, on par with campaign duration.

## The decision rule

AUC ranks campaigns by *failure risk*, which is the wrong ordering here — the riskiest campaigns raise ~3% of goal and were never close. Near misses cluster in the **middle** of the score range.

The rule, fixed on 2024 and then run once on 2025: **contact every campaign scoring 40–60%.**

| Rule (same 2,030 contacts, 2025) | Near misses found |
|---|---|
| Riskiest first | 18 |
| Random | 31 |
| **Score 40–60%** | **61** |

At a 20% save rate: **~$174,000** reaches creators who would otherwise get nothing, **$8,718** returns as platform fees, break-even is **$4.29 per contact**. A 15-minute staff call at $30/hr costs $7.50, so the recommendation is an automated email or in-app tip, not a call centre.

## Reproducing

```bash
pip install -r requirements.txt
```

Run `notebooks/` in order. Each notebook writes the inputs the next one reads.

| Notebook | Produces | Runtime |
|---|---|---|
| `01_data_cleaning.ipynb` | `campaigns_clean.parquet` | ~60 min (downloads) |
| `02_split_and_encode.ipynb` | `split.zip` (7 parquet files) | ~5 min |
| `03`–`05` (Models 1–3) | `*_val.parquet`, coefficient/grid CSVs | ~10 min |
| `06_model4_blurb_boosting.ipynb` | `model4_val.parquet`, `blurb_scores.parquet` | ~12 min on GPU, 30–40 min CPU |
| `07_evaluate_coach_impact.ipynb` | Steps 4–6: evaluation, call list, impact | ~2 min |

Built in Google Colab (T4 or Apple silicon for notebook 06). `results/` already holds the per-model validation scores, so **`07` runs on its own** without retraining anything.

## Layout

```
notebooks/   7 notebooks, numbered in execution order
docs/        one write-up per step: method, checks, limitations
results/     validation scores (.parquet) and tuning records (.csv)
```

`docs/` is the detailed record — each file states what was done, what was checked, and what we are unsure of.

## Known limitations

- **The 20% save rate is assumed, not measured.** Every dollar figure scales with it. The first thing to run is a holdout.
- **Scores are a ranking, not a probability.** Model 4 expected 79.0% of 2025 campaigns to be funded; 70.7% were (+8.3 pts optimistic).
- **A few large campaigns dominate the impact.** One is 14% of the $871,770; valuing all saved campaigns at the median goal cuts the fee to ~$3,058.
- **Near-miss counts are thin** (541 in 2024, 410 in 2025), so coaching-list differences between the top models are within noise even where AUC separates them.
- Blurb text only — not titles, images or video. The encoder is frozen, not fine-tuned.
