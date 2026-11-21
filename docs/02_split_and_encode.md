# Step 2: Split & Encode — Kickstarter campaigns, 2016–2025

Step 2 is done: four year-based slices with no shared campaigns:
- **2016–2018:** Deep learning blurb model (Model 4)
- **2019–2023:** Training (Models 1–3)
- **2024:** Validation
- **2025:** Test

There are model inputs of 191 numeric columns (there are so many because of the one-hot encodings for categorical variables like location and category), identical across training, validation and test. This page records each step we took and why, in the order we ran them.

## Input and output
- **Input:** `campaigns_clean.parquet` from Step 1 (273,163 campaigns, 2016–2025, 22 columns). Located in 1) Data Cleaning drive folder.
- **Output:** `split.zip`, seven files with the cleaned, split data that every later step should load.

## ⚠️ Team rules (Model 1 onward)
- **Columns starting with `post_` are never model inputs.**
- Category success rates for the baseline come from training (2019–23) only.
- Don't look at test results until the final run.

## 1. Load and check the clean file
**Goal:** confirm the file arrived intact before building on it.

1. Upload `campaigns_clean.parquet` to Colab and load it.
2. Check rows, duplicates and per-year counts against the Step 1 notebook.
   - Result: 273,163 rows, 0 duplicate ids, and per-year near-misses matching Step 1 (541 in 2024, 410 in 2025).

## 2. Unpack category and location
**Goal:** turn two columns of JSON text for the category and location columns into clean labels a model can use.

1. **Why:** each `category` and `location` cell is one long string, such as `{"id":34,"name":"Tabletop Games","slug":"games/tabletop games",...}`. The old JSON snapshots (2016–18) and the CSV snapshots (2019+) store these fields in different layouts.
2. **New columns** (the originals are kept for checking):
   - `subcategory` = the category's name, such as "Tabletop Games" (161 values).
   - `parent_category` = the part of the slug before "/", such as "Games" (15 values).
   - `loc_state` = the location's state or region code, such as "CA".
   - `loc_type` = the kind of place, such as "Town".
3. **Check:** 0% blank in all four columns; a random sample matched its JSON on every row.
4. **Fix for repeated names:** about 10 subcategory names appear under more than one parent (171 parent + subcategory pairs vs. 161 names). Each subcategory is now labeled with its parent, such as "Games / Tabletop Games", so names like Anthologies under Comics and under Publishing stay separate.
   - The label is rebuilt from the raw JSON, so the cell is safe to run more than once.

## 3. Split by year
**Goal:** decide which campaigns the models study and which they are tested on, using time, not random shuffling.

| Slice | Years | Job | Campaigns | Success rate | Near-misses |
|---|---|---|---|---|---|
| Text training (DL) | 2016–2018 | Blurb model only | 100,240 | 54% | 1,673 |
| Training | 2019–2023 | Main models study here | 119,297 | 73% | 1,923 |
| Validation | 2024 | Choose settings, best model, cutoff | 26,739 | 72% | 541 |
| Test | 2025 | Final results, run once | 26,887 | 71% | 410 |

1. **Total check:** the four slices add up to all 273,163 campaigns.
2. **Overlap check:** all six pairs of slices share 0 campaign ids.
3. **Note for the README:** success jumps from 54% (text years) to 73% (training years). The blurb model learns from a period with more failures. It should still rank blurbs well, but its raw scores may run low on later years (our interpretation, untested).

## 4. Group rare labels
**Goal:** keep the label columns small enough to encode, while keeping most of the information.

Rule: every "most common" list is counted on **training (2019–23) only**. Counting 2024–25 too would let the future shape the inputs (leakage).

| Column | Decision | Training coverage | Why |
|---|---|---|---|
| `parent_category` | Keep all 15 | 100% | Few enough to keep; also covers campaigns whose subcategory became "other" |
| `subcategory` | Top 60 + "other" | 81.1% (val/test about 85%) | Top 40 covered only 71.4%; the 20 added labels average about 580 campaigns each |
| `loc_state` | Country + state, top 60 + "other" | 81.2% (val/test about 77%) | Prefix stops code clashes, such as "US / WA" vs. "AU / WA"; top 20 without prefix covered only 58% |
| `loc_type` | Town, County, Suburb, LocalAdmin + "other" | 99.7% | Town alone is 94% of training; "other" holds 315 campaigns |
| `country` | Keep all 25 | 100% | No unseen countries in validation or test |
| `currency` | **Dropped** | — | Repeats country (e.g., US and USD both 68,284) |

## 5. Add launch-day number columns
**Goal:** add facts Kickstarter knows on launch day, and check each one against the training success rate before keeping it.

| Column | How it's built | Patterns: What % of campaigns got funded | Kept as |
|---|---|---|---|
| `duration_days` | (deadline − launch) ÷ 86,400 seconds | Longer campaign = lower % funded: 88% (≤20 days) → 41% (50–60 days) | Number |
| `log_prep_days` | Log of (launch − page creation) in days | Launched within a week: 59%; 1–3 months of prep: 81%; flat after that | Number (log scale) |
| `launch_month` | Month of launch | 72–76% most months; December 65% | Label (12 columns) |
| `launch_weekday` | Weekday of launch (UTC) | Tuesday 80%; weekends 66–68% | Label (7 columns) |
| `name_len` | Characters in the name | Longer = higher: 64% (≤20 chars) → 79% (52–61) | Number |
| `blurb_len` | Characters in the blurb | Flat, 71–76% | Number (kept; costs one column) |

1. **Why logs:** a log measures "how many times bigger," not "how many more." Doubling a goal from $1,000 to $2,000 counts as much as $50,000 to $100,000. Logistic regression gives each column one fixed weight per unit, so it needs this. Boosting doesn't, but it isn't hurt. `log_goal_usd` comes from Step 1.
2. **Originals stay in the files** (`goal_usd`, `prep_days`) so people can read plain dollars and days. They are not model inputs: two versions of the same fact make logistic regression weights unstable.
3. **Facts vs. interpretation:** these are links, not causes. Long campaigns may not cause failure; creators expecting a slow campaign may choose longer ones.
4. **Open data notes:**
   - 2,083 training campaigns run over 60 days (max 120.6). We expected 60 to be Kickstarter's limit; kept, flag in the README.
   - Weekdays use UTC time, so some US evening launches count as the next day.

## 6. One-hot encode and line up the columns
**Goal:** turn every input into numbers, with identical columns in training, validation and test.

| Input type | Columns | Total input columns |
|---|---|---|
| Numerical inputs before one-hot encoding | `log_goal_usd`, `duration_days`, `log_prep_days`, `name_len`, `blurb_len` | 5 |
| Yes/no numerical inputs after one-hot encoding | `parent_category` (15), `subcategory` (61, including other), `loc_state` (61, including other), `loc_type` (5, including other), `country` (25), `launch_month` (12), `launch_weekday` (7) | 186 |
| **Total** | | **191** |

1. **One-hot encoding:** each label column becomes a group of yes/no columns, with exactly one 1 per group. Example: a Fashion / Apparel campaign from New York launched on a Wednesday in January has 1s in `parent_category_Fashion`, `subcategory_Fashion / Apparel`, `loc_state_US / NY`, `launch_month_1` and `launch_weekday_Wednesday`.
2. **Blanks:** filled with the **training** median in every slice. There were 0 blanks, so this is a safeguard only.
3. **`.reindex(columns=X_train.columns, fill_value=0)`** gives validation and test exactly the training columns, in the same order.
4. **Checks:** 191 columns and 0 blanks in all three slices; columns identical across them.
5. **Left out on purpose:** `currency`, `goal_usd` and `prep_days` (log versions used instead), raw `category`, `location`, name and blurb text, raw dates, and every `post_` column. The text slice isn't encoded; only the blurb model uses it.

## 7. Save the files
**Goal:** save once, so every later step loads the same split. Colab deletes its files when the session ends, so download `split.zip`.

| File | Rows × columns | What it holds | Used by |
|---|---|---|---|
| `X_train` | 119,297 × 191 | Model inputs for training set (2019–2023) – no answers | Models 1–4 learn from this set |
| `X_val` | 26,739 × 191 | Model inputs for validation set (2024) | Used to compare models, tune settings, pick cutoff |
| `X_test` | 26,887 × 191 | Model inputs for test set (2025) | Run once only on the winning model, at the end. |
| `slice_text` | 100,240 × 33 | Full spreadsheet of data for deep learning blurb model (2016–2018) including columns that are not model inputs | Model 4's blurb grader only. It learns which blurb meanings succeed from the blurb text, so it needs no 191-column inputs and there is no `X_text`. Models 1–3 never see these years, which keeps the blurb scores on 2019–2025 fair. |
| `slice_train` | 119,297 × 33 | Full spreadsheet (2019–2023), same rows as `X_train` but includes all columns including those that aren't model inputs | Used by Models 1–4: This spreadsheet includes the answer key (`success` column) to teach the models. Model 4's blurb grader also scores these blurbs. |
| `slice_val` | 26,739 × 33 | Full spreadsheet (2024), same rows as `X_val` but includes all columns including those that aren't model inputs | Used to grade Models 1–4 on 2024: the answer key (`success` and `near_miss` columns) for comparing models, tuning settings and picking the cutoff. Model 4's blurb grader also scores these blurbs. |
| `slice_test` | 26,887 × 33 | Full spreadsheet (2025), same rows as `X_test` but includes all columns including those that aren't model inputs | Final grading of the winning model, run once (`success` and `near_miss` columns). Model 4's blurb grader also scores these blurbs. |

## Open decisions
- **Near-miss counts are below target.** Ideally we would have about 1,000 "near-misses" each in validation and test; we have 541 (2024) and 410 (2025).
  - Split is **locked as is** so Model 1 can start.
  - If hit rates look noisy later, we can consider widening the near-miss range (e.g., 40–99%). That re-runs one line and rebuilds no model, because models predict `success`, not `near_miss`.

## Limitations (README notes)
Campaigns over 60 days, UTC weekdays, the success-rate gap between text and training years, and the choice of top 60 for subcategory and location.
