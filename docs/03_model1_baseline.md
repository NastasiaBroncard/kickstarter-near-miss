# Step 3 · Model 1: Baseline (category lookup table)

Model 1 is done. Every 2024 campaign now has a score, and there is a number for Models 2–4 to beat.

**The bar: AUC 0.807 on validation (2024).**

The model is one average per category, nothing else:

> `score(campaign) = average of the success column, over the training campaigns in that campaign's category`

No fitting, no settings, no inputs other than the category label. This page records each step and why, in the order we ran them.

## Input and output
- **Input:** `slice_train.parquet` (2019–23) and `slice_val.parquet` (2024) from Step 2.
- **Output:**
  - `baseline_val.parquet` — 26,739 rows × 4 columns (`id`, `score_model1`, `success`, `near_miss`). Models 2–4 load this to compare on identical rows.
  - `baseline_lookup.csv` — the 61-row lookup table. This *is* the model, and it is the slide-ready version of it.

## ⚠️ Team rules followed
- No `post_` columns anywhere.
- Every rate counted on **training (2019–23) only**.
- `slice_test.parquet` (2025) never opened.

## 1. Pick the category column
**Goal:** decide what "category" means before computing anything.

Step 2 built two candidates. We scored both and took the stronger one, so Models 2–4 have to beat the best simple thing rather than the weakest.

| Lookup column | Categories | Distinct scores | AUC on 2024 |
|---|---|---|---|
| none — everyone gets 73.5% | 1 | 1 | 0.500 |
| `parent_category` | 15 | 15 | 0.737 |
| **`subcategory`** | **61** | **60** | **0.807** |

**Decision: Model 1 is the `subcategory` lookup.**

1. **Why the finer column wins by 0.070:** inside Technology, "Apps" funds 9% of the time and the "Technology" catch-all nearly always. The 15-label column averages that distinction away.
2. **Why 60 distinct scores, not 61:** `Comics / Comics` and `Technology / Technology` both sit at exactly 100% in training.
3. **Coverage check:** 0 of 2024's category labels were unseen in training, so the global-rate fallback never fires. Step 2 learned the top-60 list from training and applied it to every slice, which is what makes this true.

## 2. What the lookup table says
**Goal:** read the model, since it is small enough to read.

The spread is 8.8% to 100%, which is why so simple a rule reaches 0.807.

| Lowest 5 (training rate) | | Highest 5 (training rate) | |
|---|---|---|---|
| Technology / Apps | 8.8% | Technology / Technology | 100.0% |
| Technology / Web | 10.7% | Comics / Comics | 100.0% |
| Games / Mobile Games | 11.5% | Art / Art | 99.9% |
| Technology / Software | 13.8% | Design / Design | 99.9% |
| Food / Food Trucks | 16.1% | Publishing / Publishing | 99.7% |

1. **The bottom of the table holds up:** Technology / Apps predicted 8.8%, actual 9.2% in 2024.
2. **The top does not:** several categories predicted above 99% land at 74–92% in 2024. See section 6.
3. **Smallest training cell: 495 campaigns.** Step 2's top-60 rule left no thin cell, which is why section 5 could rule out smoothing.

## 3. Grading the model (validation 2024)
**Goal:** the three metrics from Step 4 of the plan, on 2024 only.

| Metric | Model 1 | Reference |
|---|---|---|
| AUC | **0.807** | 0.500 = coin flip |
| Precision of the 2,000-campaign coaching list | 81.6% | 27.8% if called at random |
| Share of 2024 failures reached | 21.9% | 26.9% is the ceiling at this capacity |
| Near-miss hit rate, riskiest-first | 1.85% | 2.02% random — **loses** |
| Near-miss hit rate, 20–40% score bracket | 3.76% | 2.02% random — 1.9× |
| Calibration | +5.7 points optimistic | — |

### Why we cut by capacity, not at 0.50
A 0.50 cutoff flags 3,240 campaigns. Nobody decided Kickstarter can coach 3,240 campaigns — that number is an accident of where the category averages fall, and it moves with the base rate rather than with staffing. So `coaching_list()` fills a fixed number of slots riskiest-first, the same shape as HW4's `replacement_plan()`.

**`COACH_CAPACITY = 2,000` is a placeholder and needs a real staffing number.** It is roughly 7.5% of 2024 campaigns, about 8 calls per working day for one coach. Every metric above is reported as a function of it, so changing it re-runs one line.

### The tie problem
With 60 distinct scores, the capacity cut lands *inside* a block of tied campaigns: at capacity 2,000, 1,876 campaigns score strictly below the cut and 281 sit exactly at it, so 124 of that block are picked arbitrarily (ties broken by campaign id, so the list is reproducible).

This is a real weakness, not a bug. The lookup cannot rank two Tabletop Games campaigns against each other even though their goals, lengths and prep times differ enormously. **That gap is what Models 2–4 exist to close.**

## 4. ⚠️ The finding that changes Step 5
**Near-misses are not the riskiest campaigns.** This is the most important result here, and it is not about Model 1 — it applies to every model we build.

Calling the 2,000 campaigns the model thinks are riskiest produces a **1.85% near-miss hit rate against a 2.02% random rate**. It is worse than picking 2,000 campaigns at random.

The reason: a campaign scored under 20% does fail, but it fails at 3% of goal, not 80%. It was never close. Near-misses cluster where the model is *uncertain*.

| Score band | Campaigns | Near-miss rate | Lift vs random |
|---|---|---|---|
| 0–20% | 1,226 | 1.22% | 0.60× |
| 20–40% | 931 | 3.76% | **1.86×** |
| 40–60% | 7,101 | 3.38% | 1.67× |
| 60–80% | 1,189 | 3.28% | 1.62× |
| 80–100% | 16,292 | 1.30% | 0.64× |

**Three consequences:**
1. **Riskiest-first is the wrong coaching rule for every model.** Step 5's bracket approach was the right instinct; this is the evidence for it.
2. **AUC on `success` is still right for *choosing* a model, but it is not the business metric.** A model can improve AUC and still build a worse coaching list. Both numbers belong in the Step 4 comparison table.
3. **Worth a team decision:** the models predict `success`; the business cares about `near_miss`. Keeping `success` is defensible — near-misses are only 2% of campaigns, so a direct model trains on very few positives — but the write-up should say *why* rather than leave it implicit.

Model 1 also cannot produce a bracket of any chosen size: the best rule available from it is a list of 931, not 2,000. Matching brackets to real coaching capacity is another job for Models 2–4.

## 5. Smoothing: tested, rejected
**Goal:** check whether small categories need shrinking toward the global rate.

Categories sitting at exactly 0% or 100% are usually overconfident, and the standard fix shrinks each toward the global rate by `k` imaginary average campaigns.

| k | 0 | 25 | 50 | 100 | 250 |
|---|---|---|---|---|---|
| AUC | 0.8067 | 0.8073 | 0.8076 | 0.8063 | 0.8053 |

Best case is +0.001 at `k = 50`. **Model 1 stays unsmoothed** — Step 2's top-60 rule already did the work, and a baseline should stay the simplest defensible thing.

## 6. Data-quality flag for Models 2–4
Seven categories repeat their parent's name — `Art / Art`, `Music / Music`, `Technology / Technology`, `Comics / Comics`, `Publishing / Publishing`, `Design / Design`, `Crafts / Crafts`. These are campaigns whose JSON had no sub-level, so Step 2's label-building fell back to the parent name.

**8,915 training campaigns (7.5% of training) at a 99.8% average success rate, dropping to 89.4% in 2024.** `Music / Music` falls from 99.5% to 74.1%. A near-perfect training rate that does not hold up is the signature of a cell that is not a real category.

Not blocking for Model 1 — these campaigns score at the top of the range where few near-misses live. Flagged because **Models 2–4 use the same 61 one-hot columns and will inherit it**, and because it belongs in the final limitations.

## 7. Calibration
Separate from ranking. The lookup predicts 77.9% of 2024 campaigns will succeed; 72.2% did — **5.7 points optimistic**, because it learned on 2019–23 when success averaged 73.5% and 2024 was slightly harder.

This does not affect AUC (adding a constant to every score changes no ranking), but it does affect Step 5, where brackets are read as real probabilities. Models 2–4 should be checked the same way.

## What Models 2–4 have to beat
**AUC 0.807 on 2024.** Two notes on reading that number:

1. **If logistic regression lands near 0.82, that is a finding, not a failure.** It would mean category explains most of what is knowable on launch day, and that is a more honest headline than an unexplained number. Report it either way.
2. **Beat the baseline on both metrics.** AUC says the model ranks success better; the coaching-list hit rate says it builds a better call list. Model 1 shows these can move in opposite directions.

## Limitations
- 60 distinct scores and heavy ties — capacity-sized coaching lists cannot be cut cleanly.
- +5.7 points optimistic on 2024.
- Inherits the `X / X` category artifact (section 6).
- `other` lumps 18.9% of training into one cell; in 2024 it holds 3,937 campaigns at a 2.9% near-miss rate, so real signal is hidden inside it.
- `COACH_CAPACITY = 2,000` is a placeholder, not a staffing estimate.

## How to reproduce
1. Open `[code] Model 1) Baseline.ipynb` in Colab, upload `split.zip` and unzip to `/content/split`. (Running locally also works — the notebook falls back to the current directory.)
2. Run all cells — about 10 seconds.
3. Output: `baseline_val.parquet` and `baseline_lookup.csv`.
