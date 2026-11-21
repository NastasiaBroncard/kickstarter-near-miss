# Step 3 · Model 2: Logistic regression (readable)

Model 2 is done. **It beats the baseline: AUC 0.853 vs 0.807.**

Every variable gets a weight, so a campaign's score is built by adding and subtracting points — a higher goal loses points, a longer campaign loses points, a strong category gains points — and the total becomes a 0–100% chance of success. This page records each step and why, in the order we ran them.

## Input and output
- **Input:** `X_train.parquet` / `X_val.parquet` (191 encoded columns) plus `slice_train` / `slice_val` for the answers, and `baseline_val.parquet` from Model 1 for the comparison.
- **Output:**
  - `model2_val.parquet` — 26,739 rows (`id`, `score_model2`, `success`, `near_miss`) for Models 3–4 to compare against.
  - `model2_coefficients.csv` — all 191 weights, sorted. This is the readable artifact and the source for any "what drives success" slide.

## ⚠️ Team rules followed
- Launch-day inputs only, no `post_` columns.
- **Scaler and model both fitted on training (2019–23) only.**
- `slice_test.parquet` (2025) never opened.

## 1. Row alignment, checked not assumed
Step 2 built `X_train` and `slice_train` with matching row orders. The notebook asserts it on three shared columns rather than trusting it — if the rows were offset, every campaign would be graded against someone else's answer and **nothing downstream would warn us**. All four files pass.

## 2. Standardize, fitting the scaler on training only
Logistic regression gives one weight per unit, so columns on different scales get incomparable weights: `log_goal_usd` runs 0.4–18.6, `blurb_len` runs 0–370. Standardizing puts them all at zero mean, one standard deviation.

`StandardScaler().fit(X_train)`, then `.transform()` on each slice. Fitting on validation would feed 2024 into the inputs (L4 slide 17) — the same leak Step 2 avoided by learning the top-60 category list from training alone.

Check: training scales to mean 0.000, sd 1.000. Validation scales to mean −0.0003, **which is correct and worth not "fixing"** — 2024 genuinely differs from 2019–23, and we want the model to see that.

## 3. Regularization: tested, barely matters
| C | 0.001 | 0.01 | 0.1 | 1.0 | 10 | 100 |
|---|---|---|---|---|---|---|
| train AUC | 0.9220 | 0.9228 | 0.9229 | 0.9229 | 0.9229 | 0.9229 |
| **val AUC** | **0.8526** | 0.8514 | 0.8510 | 0.8509 | 0.8509 | 0.8509 |

We take `C = 0.001` because validation says so (L4 slide 24: validation selects). But the spread across four orders of magnitude is **0.0017**, which is inside noise. The honest statement is that this model is insensitive to regularization; HW4's default `C = 1.0` gives 0.8509. **Don't claim tuning as a contribution.**

The **train 0.922 vs validation 0.853 gap** is part genuine overfitting and part year-to-year drift. Model 1 showed a similar gap with no fitted parameters at all, which is the clue that drift is a real component. This notebook does not separate the two.

## 4. The headline: AUC 0.853, beating the baseline by +0.046

| | AUC on 2024 |
|---|---|
| Model 1 (category lookup) | 0.807 |
| **Model 2 (logistic, 191 inputs)** | **0.853** |
| gain | **+0.046** ✅ |

## 5. Where the gain actually comes from
Not "logistic regression is better" — something much more specific.

Model 1 gives every Tabletop Games campaign the same 99.5%, whether the goal is $500 or $500,000. Its AUC *inside* any single category is therefore exactly 0.500 by construction. Model 2 produces 26,739 distinct scores instead of 60:

| Subcategory | Campaigns in 2024 | Model 1 AUC here | Model 2 AUC here |
|---|---|---|---|
| Games / Tabletop Games | 4,907 | 0.500 | **0.776** |
| Technology / Apps | 606 | 0.500 | **0.818** |
| Film & Video / Documentary | 328 | 0.500 | **0.721** |
| Publishing / Children's Books | 422 | 0.500 | **0.704** |
| other | 3,937 | 0.500 | **0.747** |

**That is the entire +0.046.** Worth stating this way in the deck: the improvement is not that the model is cleverer, it is that the baseline could not see goal size, length or prep time, and those matter.

## 6. What the model says, in plain English
Weights are converted from standardized units back to original units, then expressed as odds multipliers.

| Change | Odds of success |
|---|---|
| **Doubling the funding goal** | **−24%** |
| Goal 10× bigger | −60% |
| **Campaign 10 days longer** | **−24%** |
| Campaign 30 days longer | −57% |
| **Doubling preparation time** | **+19%** |
| Name 10 characters longer | +13% |
| Blurb 10 characters longer | +1% (nothing) |

Strongest categories pushing toward success: Games / Tabletop Games, Design / Product Design, Art / Art, Publishing / Children's Books. Strongest toward failure: Technology / Apps, Games / Mobile Games, Technology / Web, Technology / Software.

**The funding goal is the single strongest lever** and the only numeric variable near the top of the ranked weights. Category dominates the rest — which is exactly why Model 1 got to 0.807.

⚠️ **Caveat on reading individual category weights.** Step 2 kept *all* levels of each categorical variable (15 parent categories, not 14), where L2 slide 28 creates `m − 1` dummies with one reference level. With all levels plus an intercept the columns are collinear; the L2 penalty still returns a unique answer, but each category's weight reads as **"versus the average category"**, not "versus a named reference". Arguably easier to explain — but it isn't the textbook setup, and anyone reading the table needs to know which they're looking at.

⚠️ `subcategory_other` is strongly negative (−0.52). That's a warning, not a finding: "other" is 18.9% of training in one cell, so the model is saying "unusual categories do badly on average".

## 7. Reading one real campaign, line by line
This is what the plan asked for — *"you can read how much each variable pushes the prediction up or down."* A real 2024 campaign that ended as a near-miss:

```
campaign 612560855 | other | goal $10,658 | 30 days | 22 days of prep

what the model sees                      points    total   chance
starting point (intercept)               +4.835    4.835    99.2%
log_goal_usd = 9.27                      -3.713    1.122    75.4%
duration_days = 30.42                    -0.848    0.274    56.8%
log_prep_days = 3.13                     +0.791    1.066    74.4%
name_len = 26.00                         +0.315    1.381    79.9%
blurb_len = 98.00                        +0.060    1.441    80.9%
parent_category_Film & Video             -0.032    1.409    80.4%
subcategory_other                        -1.328    0.081    52.0%
loc_state_CA / QC                        -0.075    0.005    50.1%
loc_type_Town                            -0.017   -0.012    49.7%
country_CA                               +0.010   -0.002    50.0%
launch_month_12                          -0.157   -0.159    46.0%
launch_weekday_Friday                    -0.016   -0.176    45.6%

final score: 45.6%   -- actual outcome: failed at 50-99% of goal (near-miss)
```

Campaigns start at 99.2% before anything is known, because most campaigns succeed. The $10,658 goal costs 3.7 points, the 30-day length another 0.8, the 22 days of prep win 0.8 back, and sitting in `other` costs 1.3. It lands at 45.6% — squarely in the coaching bracket — and it did in fact end as a near-miss.

**This table is the Model 2 slide.** It is the one output in the whole project that a non-technical stakeholder can read unaided.

## 8. ⚠️ The better model builds a worse coaching list
This sharpens the Model 1 finding rather than contradicting it.

| Coaching list, 2,000 campaigns | Precision (did fail) | Near-miss hit rate | Lift |
|---|---|---|---|
| Model 1: category lookup | 81.6% | 1.85% | 0.91× |
| **Model 2: logistic regression** | **92.5%** | **0.95%** | **0.47×** |
| Random | 27.8% | 2.02% | 1.00× |

Model 2 is **better** at finding failures and **worse** at finding near-misses. Those are the same fact. Model 2 can see goal size, so it identifies campaigns that are *hopeless* — $90,000 goal, 60 days, no prep, weak category. Those fail at 2% of goal. They are not near-misses and a coaching call will not save them. Model 1 was too blunt to find them and accidentally scraped up more borderline campaigns on the way.

**For the deck: improving AUC made this particular business rule worse.** That is why the Step 4 table needs both columns, and why Step 5's rule cannot be "call the riskiest".

## 9. With the plan's bracket rule, Model 2 wins properly
The five brackets written into Step 5 — declared in advance, not searched, so these aren't the best of many tried windows.

| Bracket | Model 1 | | Model 2 | |
|---|---|---|---|---|
| | campaigns | hit rate | campaigns | hit rate |
| 0–20% | 1,226 | 1.22% | 1,714 | 0.88% |
| 20–40% | 931 | **3.76%** | 1,977 | 2.63% |
| **40–60%** | 7,101 | 3.38% | **2,542** | **4.21%** |
| 60–80% | 1,189 | 3.28% | 3,095 | 3.97% |
| 80–100% | 16,292 | 1.30% | 17,411 | 1.40% |

**Model 2's 40–60% bracket: 2,542 campaigns at 4.21%, 2.08× random.** Two wins over Model 1's best (3.76% on 931):
1. **Higher hit rate.**
2. **Roughly the right size** — 2,542 against a 2,000-campaign capacity. Model 1's best bracket held 931 and there was no way to enlarge it, because a 60-value lookup table cannot produce a list of any chosen size.

The hit rate still falls off at both ends. A sharper model moved the peak higher but did not move *where* the peak is.

## 10. Calibration: still ~6 points optimistic
| | Mean predicted | Actual 2024 | Optimism |
|---|---|---|---|
| Model 1 | 77.9% | 72.2% | +5.7 |
| Model 2 | 78.5% | 72.2% | +6.4 |

Fitting 191 parameters did not fix this, and that's the point. Drift between training years and the scored year isn't something a model can learn its way out of when it never sees the later year. Doesn't touch AUC; does matter for Step 5, where brackets are read as real probabilities.

## 11. `blurb_len`: keep it, don't present it
Smallest weight of the five numeric inputs (+1% odds per 10 characters). Dropping it moves validation AUC by 0.0001 (0.8526 → 0.8525).

**Keep it** — one column out of 191, and Model 4 will want blurb features nearby for comparison. But it should not appear on any "what drives success" slide. Blurb *length* doesn't matter; Model 4 tests whether blurb *content* does.

## Model 2 summary

| Metric (validation 2024) | Model 1 | **Model 2** | |
|---|---|---|---|
| AUC | 0.807 | **0.853** | ✅ +0.046 |
| AUC inside Tabletop Games | 0.500 | **0.776** | ✅ source of the gain |
| Distinct scores | 60 | 26,739 | ✅ capacity-sized lists possible |
| Precision of the 2,000 list | 81.6% | **92.5%** | ✅ |
| Near-miss hit rate, riskiest-first | 1.85% | 0.95% | ❌ worse than random |
| Near-miss hit rate, best bracket | 3.76% (931) | **4.21% (2,542)** | ✅ 2.08× random |
| Calibration | +5.7 pts | +6.4 pts | ❌ marginally worse |

**The new bar for Models 3–4: AUC 0.853.**

## What to carry into Model 3
1. **The gain came from ranking inside categories**, goal size above all. Boosting should extend this by finding interactions a linear model cannot — a large goal may matter more in Technology than in Comics.
2. **Report both metrics, always.** Model 2 improved AUC and made the naive coaching rule worse. A single-number comparison would have hidden that completely.
3. **The 40–60% bracket is the working coaching rule** until Step 5 sets it properly.
4. **Drift is unfixed and may not be fixable.** Both models run ~6 points optimistic. That's a limitations line, not another modelling attempt.

## Limitations
- Regularization barely matters (0.0017 across the sweep), so the "tuned" `C` is cosmetic.
- All one-hot levels kept, so category weights read against the average category, not a named reference level.
- `subcategory_other` carries 18.9% of training in a single weight.
- The 0.069 train–validation AUC gap mixes overfitting with year drift; not separated here.
- Inherits the `X / X` category artifact flagged in Model 1 section 6.
- `COACH_CAPACITY = 2,000` is still a placeholder, not a staffing estimate.

## How to reproduce
1. Open `[code] Model 2) Logistic Regression.ipynb` in Colab, upload `split.zip`, unzip to `/content/split`, and put `baseline_val.parquet` there too. (Running locally also works — the notebook falls back to the current directory.)
2. Run all cells — about 1 minute, most of it the `C` sweep.
3. Output: `model2_val.parquet` and `model2_coefficients.csv`.
