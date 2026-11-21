# Data: Kickstarter campaigns, 2016–2025

## Source
Monthly Kickstarter snapshots published by Web Robots (https://webrobots.io/kickstarter-datasets/).
- **CSV snapshots:** June 2019 – October 2026 (87 files)
- **JSON snapshots:** January 2016 – May 2019 (40 files; CSV files don't exist before May 2019)

Raw files are not stored in this repo (too large). The cleaning notebook downloads them directly.

## How to reproduce
1. Open `01_clean_data.ipynb` in Google Colab.
2. Run all cells (about 1 hour in total: ~35 min for CSVs, ~20 min for JSONs).
3. Output: `campaigns_clean.parquet` (~74 MB). Load it with:
   ```python
   import pandas as pd
   df = pd.read_parquet("campaigns_clean.parquet")
   ```

## What the cleaning does
1. **Combines all snapshots.** Each snapshot is a partial sample of Kickstarter, so one file alone misses most campaigns.
2. **Keeps one row per campaign** — the version from the latest snapshot it appears in.
3. **Keeps finished campaigns only** (`successful` or `failed`). Live, canceled, suspended and never-launched campaigns are dropped because they have no final outcome.
4. **Keeps launches from 2016–2025.** 2026 launches are excluded; 2026 snapshots are still used because they contain final outcomes for late-2025 campaigns.
5. **Creates the answer columns** (`success`, `near_miss`).
6. **Converts goals to USD** using `static_usd_rate`, the same rate Kickstarter uses for its own USD figures (verified: reproduces `usd_pledged` for 100% of rows).
7. **Separates spoilers.** Anything not known on launch day is either dropped or renamed with a `post_` prefix.

**Result:** 273,163 campaigns, 22 columns, no duplicate IDs.

## ⚠️ Team rule
**Columns starting with `post_` must never be used as model inputs.** They describe what happened after launch. They are kept only for building answers and analysis.

## Column guide

| Column | Meaning | Model input? |
|---|---|---|
| `id` | Kickstarter campaign ID | No (identifier) |
| `year` | Launch year | Used for the train/validation/test split |
| `snapshot` | Date of the snapshot this row came from | No (bookkeeping) |
| `success` | 1 = reached its goal, 0 = did not | **Answer** |
| `near_miss` | 1 = failed but raised 50–99% of goal | **Answer** (for the coaching metric) |
| `log_goal_usd` | log(1 + goal in USD) | Yes |
| `goal_usd` | Funding goal in USD | Yes (prefer the log version) |
| `category` | Category info, stored as JSON text | Yes (needs unpacking) |
| `country` | Creator's country | Yes |
| `currency` | Campaign currency | Yes |
| `launched_at` | Launch time (Unix seconds) | Yes (e.g., month, day of week) |
| `deadline` | End time (Unix seconds) | Yes (campaign length = deadline − launched_at) |
| `created_at` | When the page was created (Unix seconds) | Yes (prep time = launched_at − created_at) |
| `name` | Campaign title | Yes (text) |
| `blurb` | Short description | Yes (text; input to the blurb model) |
| `location` | Location info, stored as JSON text | Optional |
| `post_state` | Final outcome (`successful` / `failed`) | **Never** |
| `post_usd_pledged` | Amount raised, USD | **Never** |
| `post_pct_funded` | Pledged ÷ goal × 100 | **Never** |
| `post_backers` | Number of backers | **Never** |
| `post_staff_pick` | Picked by Kickstarter staff (may happen mid-campaign) | **Never** |
| `post_state_changed_at` | When the outcome was recorded | **Never** |

**Dropped entirely:** `spotlight` (perfectly reveals the outcome: 100% of spotlighted campaigns succeeded), `video` (missing in JSON files), `usd_exchange_rate` and `fx_rate` (mostly missing), plus raw-currency versions of goal and pledged.

## Key decisions
- **Near-miss = 50–99% funded.** Originally 70–99%, widened because 70–99% gave only ~150 near-misses per recent year. With 50–99%: 541 in 2024, 410 in 2025.
- **Percent funded is calculated as pledged ÷ goal**, because JSON files lack Kickstarter's `percent_funded`. The calculation matches Kickstarter's own value for 100% of CSV rows.

## Campaigns by launch year

| Year | Campaigns | Success rate | Near-misses (50–99%) |
|---|---|---|---|
| 2016 | 36,257 | 45% | 585 |
| 2017 | 34,099 | 56% | 567 |
| 2018 | 29,884 | 63% | 521 |
| 2019 | 28,240 | 69% | 386 |
| 2020 | 25,836 | 73% | 336 |
| 2021 | 23,277 | 78% | 266 |
| 2022 | 20,645 | 75% | 432 |
| 2023 | 21,299 | 74% | 503 |
| 2024 | 26,739 | 72% | 541 |
| 2025 | 26,887 | 71% | 410 |

## Limitations
- **Failed campaigns drop out of later snapshots over time.** Older years rely on the JSON files to recover them (~10,500 failures from 2016–2018 recovered). Some failures are likely still missing.
- **Partial scrapes, mid-2021 to mid-2023.** Several Web Robots snapshots in this period are very small. Later full snapshots recovered most campaigns.
- **Success rate rises over time** (45% in 2016 → 71–78% in 2021–2025). This may be a real trend or a data-collection effect; it matters when training on older years and testing on recent ones.
- **4 campaigns are marked successful but under 100% funded.** Kept, since `state` is Kickstarter's official outcome.
- **One CSV snapshot (2019-05-16) failed to download**; its JSON version was used instead.
- **Very large goals** (up to ~$152M) are kept; the log scale limits their influence.
