# Step 3 · Model 3: Gradient boosting (less readable, more accurate)

Model 3 is done. **It beats Model 2: AUC 0.860 vs 0.853**, and it is the first model that is also the best on the business metric.

Many small decision trees, each trained on the mistakes the earlier trees left behind. Tree 100 doesn't look at the campaign data fresh — it looks at where trees 1–99 are still wrong. No single tree is good; the sum of them is. This page records each step and why, in the order we ran them.

## Input and output
- **Input:** the same `X_train` / `X_val` (191 columns) Model 2 used, plus `baseline_val.parquet` and `model2_val.parquet` for the comparison.
- **Output:**
  - `model3_val.parquet` — scores for all 26,739 validation campaigns, for Model 4.
  - `model3_grid.csv` — the full 32-row tuning record, so the choice of settings is auditable.
  - `model3_importance.csv` — the permutation ranking.

## ⚠️ Team rules followed
- Launch-day inputs only, no `post_` columns.
- Grid fitted on training (2019–23), scored on validation (2024).
- `slice_test.parquet` (2025) never opened.

⚠️ **Runtime ~8 minutes**, nearly all of it the grid search.

## 1. No scaling this time, and that isn't an oversight
Model 2 needed `StandardScaler` because logistic regression gives each column one weight per unit. Trees don't care — a tree only asks *"is this value above or below a cut point?"*, which gives the same answer in dollars, log-dollars or thousands of dollars.

The notebook proves it: fitting on `log_goal_usd` and on `log_goal_usd × 1000` gives AUC 0.7290 both times.

Same reason Step 2's note that boosting "isn't hurt" by the log transforms is right. The logs are there for Model 2's benefit.

## 2. The grid search
`max_depth`, `learning_rate`, `max_iter` — the three settings the plan names. 32 combinations.

**The trick that made it affordable:** `warm_start=True` lets a fitted model keep adding trees instead of starting over, so each `(depth, learning_rate)` pair trains once to 800 trees and gets scored at 100, 200, 400 and 800 along the way. 32 fits become 8, and `max_iter` is tuned almost free.

**Chosen: `max_depth=None`, `learning_rate=0.1`, `max_iter=100` → validation AUC 0.8603.**

Spread across the entire grid: **0.0077** — exactly the same size as the gain over Model 2. Tuning matters about as much as the model family change did.

One note on `max_depth=None`: it does **not** mean unlimited trees. `HistGradientBoostingClassifier` caps every tree at `max_leaf_nodes=31` by default, so "no depth limit" still means at most 31 leaves. It lets a tree be lopsided, not unbounded.

## 3. The overfitting curve, in our own numbers
L4 slide 14 draws training error falling forever while unseen-data error turns back up. Our grid is that picture in table form:

| `max_depth=None`, `lr=0.1` | 100 trees | 200 | 400 | 800 |
|---|---|---|---|---|
| train AUC | 0.934 | 0.941 | 0.951 | 0.965 |
| **validation AUC** | **0.860** | 0.859 | 0.856 | 0.853 |

Training AUC climbs with every tree. Validation peaks at 100 and falls away. By 800 trees the model is 0.112 better on data it has seen and *worse* on data it hasn't. **More trees is not more model — past the turn it's memorisation.**

Shallow trees do the opposite: at `max_depth=3`, validation is still improving at 800 trees, because each tree is too simple to memorise much. Same curve, underfitting end. That's why depth and tree count have to be chosen together, not one at a time.

**Edge check:** the winner uses 100 trees, the smallest in the grid — normally a warning that the real optimum lies outside. Testing below settles it: 25 → 0.8571, 50 → 0.8596, 75 → 0.8600, **100 → 0.8603**, 150 → 0.8597. A genuine interior peak.

## 4. Is a +0.008 gain real, or noise?
Model 2 beat the baseline by 0.046 — nobody would doubt that. Model 3 beats Model 2 by 0.008, which deserves a check before it goes in a deck.

- **Stable?** Five random seeds give identical AUC to four decimals. `HistGradientBoostingClassifier` does no subsampling by default, so it's deterministic. Rules out seed luck, says nothing about the data.
- **Bigger than the uncertainty in one year of validation?** Resampling the 26,739 campaigns 200 times and recomputing both AUCs on each resample:

| | mean | 95% CI | better in |
|---|---|---|---|
| Model 2 − Model 1 | +0.0459 | [+0.0430, +0.0486] | 100% of resamples |
| **Model 3 − Model 2** | **+0.0076** | **[+0.0059, +0.0092]** | 100% of resamples |

**The gain is real and it is small.** Both halves of that sentence belong in the write-up.

## 5. What a "less readable" model will still tell you
No coefficient table. But you can ask what it *relies* on: scramble one variable's column into noise, re-score, see how far AUC falls. One-hot columns are scrambled **as a block** (all 61 `subcategory_` columns together), since scrambling them one at a time would understate a variable whose information is spread across many columns.

| Variable | Columns | AUC lost when scrambled |
|---|---|---|
| **subcategory** | 61 | **0.151** |
| log_goal_usd | 1 | 0.036 |
| log_prep_days | 1 | 0.022 |
| duration_days | 1 | 0.019 |
| parent_category | 15 | 0.011 |
| name_len | 1 | 0.004 |
| country | 25 | 0.003 |
| launch_weekday | 7 | 0.002 |
| loc_state | 61 | 0.002 |
| blurb_len / launch_month / loc_type | 18 | ~0.000 |

Blunt, but it's the honest summary of a model you can't read directly — and it tells the same story as Models 1 and 2: **category first, goal size second, everything else a long way back.**

## 6. The surprise: Model 3 ranks *worse* inside most categories
Model 2's whole advantage over the baseline was ranking campaigns *inside* a category. Model 3 is the better model overall and is **worse at that specific job** in four of five categories:

| Subcategory | Campaigns | Model 2 | Model 3 | Difference |
|---|---|---|---|---|
| Games / Tabletop Games | 4,907 | 0.776 | 0.763 | −0.013 |
| Technology / Apps | 606 | 0.818 | 0.811 | −0.007 |
| Film & Video / Documentary | 328 | 0.721 | 0.712 | −0.009 |
| Publishing / Children's Books | 422 | 0.704 | 0.676 | −0.027 |
| **other** | 3,937 | 0.747 | **0.774** | **+0.027** |

## 7. Where the gain actually comes from: the `other` bucket
Splitting validation resolves the contradiction completely:

| Slice of 2024 | Campaigns | Model 2 | Model 3 | Gain |
|---|---|---|---|---|
| **inside `other`** | 3,937 | 0.747 | **0.774** | **+0.027** |
| everything else | 22,802 | 0.851 | 0.857 | +0.006 |

`other` is Step 2's catch-all: **18.9% of training squeezed into one label**, holding every subcategory outside the top 60. Model 2 can give that label exactly **one weight**, applied to all of them equally. Model 3 can split the bucket back apart using the columns still intact — parent category, country, goal size, launch date — effectively rebuilding distinctions the top-60 rule threw away.

**So boosting's advantage here is mostly repair work on a Step 2 decision**, not a deeper insight about Kickstarter. That reframes what to claim for it — and suggests a cheaper alternative worth a line in the write-up: **widening the top-60 list might buy the same gain without boosting at all.**

## 8. The interaction Model 2 couldn't see
Model 2 applies **one goal penalty everywhere**: doubling the goal costs 24% of the odds whether it's a board game or a phone app. It has no way to say otherwise — that's what "linear" means here.

Holding everything else at each category's median and sweeping the goal:

| | $1,000 | $5,000 | $20,000 | $100,000 |
|---|---|---|---|---|
| Tabletop Games — Model 2 | 99.1% | 98.3% | 97.0% | 94.5% |
| **Tabletop Games — Model 3** | 99.6% | 99.6% | 99.5% | **99.4%** |
| Technology / Apps — Model 2 | 31.4% | 19.4% | 12.1% | 6.7% |
| **Technology / Apps — Model 3** | 23.2% | 14.2% | 6.8% | **3.1%** |

Model 3 learned that **a big goal barely dents a tabletop game and is close to fatal for an app**. A 100× goal costs Tabletop Games almost nothing and takes Apps from 23% to 3%. Model 2 is structurally incapable of representing that, so it splits the difference and is wrong in both directions.

**This is the clearest statement of what boosting buys, and it deserves a slide:** the same $50,000 goal means different things in different categories.

## 9. Coaching metrics
The pattern from Model 2 holds and extends — **each model is better than the last at finding failures and none is good at finding near-misses.**

| 2,000-campaign list | Precision | Near-misses | Hit rate | Lift |
|---|---|---|---|---|
| Model 1: lookup | 81.6% | 37 | 1.85% | 0.91× |
| Model 2: logistic | 92.5% | 19 | 0.95% | 0.47× |
| **Model 3: boosting** | **94.1%** | 22 | 1.10% | 0.54× |
| Random | 27.8% | 40 | 2.02% | 1.00× |

Highest precision of the three, still loses to random on the metric that matters.

## 10. With the bracket rule, Model 3 wins on both metrics
Model 2 was better on AUC but awkward on the business metric. Model 3 is better on both — and its best bracket is also the right **size**:

| | Best bracket | Campaigns | Hit rate | Lift |
|---|---|---|---|---|
| Model 1 | 20–40% | 931 | 3.76% | 1.86× |
| Model 2 | 40–60% | 2,542 | 4.21% | 2.08× |
| **Model 3** | **40–60%** | **2,243** | **4.68%** | **2.31×** |

2,243 campaigns against a 2,000-campaign capacity is close enough to use with almost no adjustment. **This is the strongest result in Step 3 so far:** a coaching list that is both the right size and 2.3× better than calling at random.

The shape hasn't changed across any of the three models — near-misses in the middle, thinning at both ends. **Three different model families agreeing makes it a property of the problem, not of any one model.**

## 11. Calibration, and do all 191 columns earn their place?
**Calibration: +6.1 points optimistic**, between Model 2's +6.4 and Model 1's +5.7. Three model families, three different amounts of machinery, same ~6-point drift. It's a property of training on 2019–23 and scoring 2024, and no amount of modelling removes it.

**Columns:** the permutation table made `loc_state`, `loc_type`, `launch_month`, `launch_weekday`, `blurb_len` and `country` look worthless individually — 111 of 191 columns. Dropping all of them costs **0.0039 AUC**, which is half of Model 3's entire gain over Model 2.

Individually negligible, collectively not. **Keep all 191**, and treat "this variable looks useless" as a claim needing a test rather than something read off an importance table.

## 12. How much did tuning on 2024 inflate these numbers? (measured: +0.003)
We chose Model 3's settings by their scores on 2024 — the same year we then report performance on. Picking the best of 32 options by their score on one year means some of the winner's lead is luck rather than skill. Model 1 had 2 options, Model 2 had 6, Model 3 had 32, so **Model 3 had the most chances to look good on 2024**.

Rather than guess at the size of that effect, we measured it: hold 2023 out of training as an **inner tuning year**, repeat the identical 32-combination search there, refit on all of 2019–23, then score 2024 — which now influenced no decision.

| | Settings chosen | Scored on 2024 |
|---|---|---|
| What we shipped | tuned on 2024: `depth=None, lr=0.1, 100 trees` | 0.8603 |
| **Honest estimate** | tuned on 2023: `depth=4, lr=0.1, 800 trees` | **0.8571** |
| | **optimism from tuning on 2024** | **+0.0032** |

**Model 2 came out at exactly +0.0000** — tuning on 2023 picks the same `C=0.001` we shipped. As section 3 predicted, when all six options score within 0.0017 of each other there is nothing to get lucky about. **Model 2's 0.853 needs no caveat at all.**

**Three things this tells us:**
1. **About 40% of Model 3's +0.008 edge over Model 2 was selection luck.** The honest edge is **+0.0045**, not +0.0077. **Model 3 still wins** — the ranking survives.
2. **The settings barely matter.** 2023 picked a completely different configuration (shallow trees, 800 of them) from 2024 (deep trees, 100 of them), and the two land 0.003 apart on 2024. Many configurations reach the same place, which is consistent with the grid's 0.0077 total spread.
3. **The business metric is untouched.** The 40–60% bracket wins either way, at 4.66% (2,101 campaigns) with the 2023 settings versus 4.68% (2,243) with ours — both 2.31× random. **The result that goes in the deck does not change.**

**What we are doing about it:** keeping the shipped model, and reporting **0.860 with the +0.003 caveat stated**. Re-running Step 3 to gain 0.003 is not worth it, and the decision this number supports — Model 3 over Model 2 — is unchanged. For Model 4, use the 2023 inner-holdout from the start; it costs one extra fit and removes the caveat.

⚠️ **Label the 2024 column "validation (used for choosing)", not "performance."** 2024 has now been used to tune Model 3, compare all three models, and will choose the coaching bracket in Step 5. **2025 is the only number nobody has peeked at, and it gets used once.**

## Model 3 summary

| Metric (validation 2024) | Model 1 | Model 2 | **Model 3** |
|---|---|---|---|
| AUC | 0.807 | 0.853 | **0.860** ✅ |
| AUC inside `other` | 0.500 | 0.747 | **0.774** |
| AUC inside Tabletop Games | 0.500 | **0.776** | 0.763 |
| Precision of the 2,000 list | 81.6% | 92.5% | **94.1%** |
| Near-miss hit, riskiest-first | 1.85% | 0.95% | 1.10% ❌ |
| **Near-miss hit, best bracket** | 3.76% (931) | 4.21% (2,542) | **4.68% (2,243)** ✅ |
| Calibration | +5.7 pts | +6.4 pts | +6.1 pts |

**Model 3 is the best model so far: AUC 0.860, best coaching bracket at 2.31× random.**

## What to carry into Model 4
1. **The bar is 0.860, and the realistic gain is small.** Model 2 bought +0.046 over the baseline; Model 3 bought +0.008 over Model 2. If blurb embeddings buy another +0.008, that's a respectable result and should be presented as one — not as a disappointment.
2. **Most of Model 3's gain is repair work on the `other` bucket**, not new insight. Say so plainly, and note that widening Step 2's top-60 list might buy the same thing more cheaply.
3. **The bracket rule is now the established coaching rule.** Three model families agree near-misses sit in the middle. Step 5 sets the bracket on validation and confirms once on test.
4. **Model 4 must beat 0.860 on AUC *and* 4.68% on the bracket.** Reporting one without the other has already been misleading twice in this project.

## Limitations
- The grid is 32 combinations across three settings, not exhaustive; `min_samples_leaf`, `l2_regularization` and `max_leaf_nodes` were left at defaults.
- The +0.008 over Model 2 is real but small, and **measured at +0.0045 once tuning is done without touching 2024** (section 12).
- Tuning on validation **and** later choosing the coaching bracket on the same 2024 data reuses that year twice. Measured cost: **+0.003 AUC of optimism** (section 12). Step 5's test run is the only clean estimate.
- No readable coefficient table; permutation importance is the most that can be said.
- Inherits the `X / X` category artifact (Model 1 §6) and the ~6-point drift.
- `COACH_CAPACITY = 2,000` is still a placeholder, not a staffing estimate.

## How to reproduce
1. Open `[code] Model 3) Gradient Boosting.ipynb` in Colab, upload `split.zip`, unzip to `/content/split`, and put `baseline_val.parquet` and `model2_val.parquet` there too. (Running locally also works — the notebook falls back to the current directory.)
2. Run all cells — about 8 minutes, nearly all of it the grid search.
3. Output: `model3_val.parquet`, `model3_grid.csv`, `model3_importance.csv`.
