# CrisisForge Data Dictionary

Version: Phase 0 v0.3.0  
Default panel: `research_core`  
Model timezone: `America/New_York`  
Decision timestamp: `after_close`  
Requested sample: 2000-01-03 onward

## 1. Scope and claim boundary

### 1.1 Public `research_core`

The reproducible public track contains:

- 12 value-weighted U.S. industry research portfolios from the Kenneth French
  Data Library; and
- three transparent duration-convexity Treasury return proxies derived from
  official U.S. Treasury par yields.

This is a retrospective research panel for developing and falsifying the
CrisisForge architecture. It is not an investable ETF panel, a point-in-time
security database, or evidence of live-trading implementability. Every table and
figure must label the French portfolios as research portfolios and the Treasury
series as synthetic return proxies.

“Public” means that the current source can be reached without a paid data
subscription. It does not grant permission to redistribute every raw file or make
commercial use of it. The repository versions downloaders, configurations,
metadata, and hashes; it does not treat source availability as an open-data
license.

### 1.2 Licensed extensions

Cboe VIX, SKEW, VVIX, option surfaces, and other option-implied distributions are
**not part of the public core**. They may enter only through a licensed extension,
for example Cboe DataShop or OptionMetrics, with a separately reviewed catalog,
timestamp convention, redistribution policy, and experiment family.

Before making investability, live-risk, security-selection, or implementability
claims, replace the public targets with licensed point-in-time total returns such
as CRSP/WRDS, Bloomberg, LSEG, FactSet, or another suitable archive.

## 2. Audited Phase 0 output

The current `artifacts/phase0/quality_report.json` records the following snapshot:

| Object | Rows | Columns | First date | Last date |
|---|---:|---:|---|---|
| `target_returns.parquet` | 6,593 | 15 | 2000-01-03 | 2026-05-29 |
| `target_interval_audit.parquet` | 6,593 | 7 | 2000-01-03 | 2026-05-29 |
| `macro_features.parquet` | 6,593 | 10 | 2000-01-03 | 2026-05-29 |
| `validation_labels.parquet` | 6,593 | 4 | 2000-01-03 | 2026-05-29 |
| `optional_diagnostics.parquet` | 6,593 | 0 | 2000-01-03 | 2026-05-29 |
| `availability_alignment_audit.parquet` | 6,593 | 48 | 2000-01-03 | 2026-05-29 |
| `model_matrix.parquet` | 6,465 | 25 | 2000-07-05 | 2026-05-29 |
| Train split | 3,367 | 25 | 2000-07-05 | 2013-12-31 |
| Validation split | 1,499 | 25 | 2014-01-02 | 2019-12-31 |
| Test split | 1,599 | 25 | 2020-01-02 | 2026-05-29 |

The first target row is undefined because there is no preceding common endpoint
from which to compute a holding-period return. Each target therefore has 6,592
observations and coverage of `0.9998483239799788`. The clean model matrix is finite,
contains no imputed target return, and includes 15 asset columns plus 10 model
context columns.

The audited run status is `passed_with_warnings`: all 23 computed gates passed and
no return exceeded the configured 40% absolute-return review threshold. The only
warning was that the French snapshot ended on 2026-05-29, 60 calendar days before
the requested run endpoint. Counts and freshness are snapshot-specific and must be
read from the receipt for each new run.

The machine-readable authorities are:

- `configs/data_catalog.yaml` for sources, fields, units, roles, lags, and
  staleness; and
- `configs/pipeline.yaml` for clock, sample, splits, thresholds, seed, and paths.

## 3. Asset-level target returns

### 3.1 Kenneth French 12 Industry Portfolios

| Field | Definition |
|---|---|
| Source | Kenneth R. French Data Library |
| Landing page | <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html> |
| Exact file | <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_daily_CSV.zip> |
| Section parsed | `Average Value Weighted Returns -- Daily` |
| Source frequency | Daily |
| Source unit | Percent simple return |
| Normalized unit | Decimal simple return |
| Normalization | Parse `YYYYMMDD`; replace `-99.99` and `-999` with null; divide valid values by 100 |
| Role | Asset-level target return |
| Point-in-time status | No; this is a current retrospective research file that may be reconstructed or revised |
| Investable status | No; these are constructed portfolios, not securities or funds |
| Repository policy | Do not commit or redistribute the downloaded raw source; retain URL, retrieval time, payload hash, normalized-file hash, schema, and date range |

| Source column | Canonical column | Economic label |
|---|---|---|
| `NoDur` | `asset__industry_nondurables` | Consumer Nondurables |
| `Durbl` | `asset__industry_durables` | Consumer Durables |
| `Manuf` | `asset__industry_manufacturing` | Manufacturing |
| `Enrgy` | `asset__industry_energy` | Energy |
| `Chems` | `asset__industry_chemicals` | Chemicals |
| `BusEq` | `asset__industry_business_equipment` | Business Equipment |
| `Telcm` | `asset__industry_telecom` | Telecommunications |
| `Utils` | `asset__industry_utilities` | Utilities |
| `Shops` | `asset__industry_shops` | Wholesale, Retail, and Services |
| `Hlth` | `asset__industry_health` | Healthcare |
| `Money` | `asset__industry_finance` | Finance |
| `Other` | `asset__industry_other` | Other |

The pipeline uses the value-weighted daily section only. Equal-weighted and monthly
sections are not substituted. Optional French factor files may later be used as
benchmarks, but not silently relabeled as investable assets.

### 3.2 Synthetic Treasury return proxies

| Field | Definition |
|---|---|
| Underlying source | U.S. Treasury Daily Par Yield Curve Rates |
| Landing page | <https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView> |
| Exact yearly endpoint | `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={year}` |
| Source fields | `BC_2YEAR`, `BC_5YEAR`, `BC_10YEAR` |
| Source unit | Percent per year, par yield |
| Target unit | Approximate decimal simple return |
| Role | Asset-level public research target |
| Investable status | No; these are approximate synthetic research proxies, not observed total returns |

| Yield input | Canonical target | Label |
|---|---|---|
| `BC_2YEAR` | `asset__treasury_proxy_2y` | 2-Year Treasury Return Proxy |
| `BC_5YEAR` | `asset__treasury_proxy_5y` | 5-Year Treasury Return Proxy |
| `BC_10YEAR` | `asset__treasury_proxy_10y` | 10-Year Treasury Return Proxy |

For maturity \(M\), yields are converted from percent to decimal. On consecutive
common endpoints, the implemented approximation is:

\[
\widehat r_t^{(M)}
=
y_{t-1}\Delta\tau_t
-
\frac{M}{1+y_{t-1}}\Delta y_t
+
\frac{1}{2}
\frac{M(M+1)}{(1+y_{t-1})^2}
(\Delta y_t)^2,
\]

where:

\[
\Delta\tau_t
=
\frac{\text{calendar time between endpoint }t-1\text{ and }t}{365.25\text{ years}}.
\]

The elapsed-time carry is therefore not assumed to be \(y_{t-1}/252\) when the
endpoint gap spans multiple calendar days. The code retains \(1/252\) only as a
fallback for a non-datetime index, which the production Phase 0 panel does not use.

This approximation does not model coupons, exact cash-flow duration, roll-down,
rebalancing, bid/ask costs, or reinvestment. It is not a constant-duration index,
individual Treasury security, ETF, or substitute for a licensed bond total-return
series.

## 4. Common-endpoint holding intervals

Industry returns and Treasury yields have different holiday calendars. Simply
joining same-date rows would risk assigning different holding periods to the two
asset groups. Phase 0 instead defines:

\[
\mathcal E
=
\mathcal C_{\text{industry}}
\cap
\mathcal C_{\text{Treasury}},
\]

where \(\mathcal E\) is the ordered set of common endpoints.

For every interval \((e_{t-1},e_t]\), French simple returns are compounded across
all source observations in that interval:

\[
R_{e_t}
=
\exp\left(
\sum_{s\in(e_{t-1},e_t]}
\log(1+r_s)
\right)-1.
\]

Treasury proxy returns use yields at the same two endpoints and the actual
elapsed-calendar-time carry described above. Consequently, one target row is a
**common-endpoint holding interval**; it is not a claim that every row represents
one identical exchange session.

`data/interim/target_interval_audit.parquet` records:

| Column | Meaning |
|---|---|
| Index `interval_end` | Common target endpoint |
| `interval_start` | Previous common endpoint |
| `calendar_days` | Elapsed calendar days in the interval |
| `endpoint_present__industry_returns` | Whether the industry source contains the endpoint |
| `observations__industry_returns` | Number of industry observations in the interval |
| `endpoint_present__treasury_yields` | Whether the Treasury source contains the endpoint |
| `observations__treasury_yields` | Number of Treasury observations in the interval |
| `all_source_endpoints_present` | Conjunction of source endpoint checks |

The first interval has no start, duration, counts, or target return. The
`common_interval_endpoints_aligned` gate checks every subsequent row for positive
duration and complete source endpoints.

## 5. Public model context

The public model matrix contains 10 `macro__*` columns: five official/derived rate
and policy features plus five return-derived market-state features. It contains no
Cboe option index.

### 5.1 U.S. Treasury par-yield context

The same Treasury source used for the proxy targets supplies:

| Source / derivation | Canonical feature | Unit | Model-session lag | Maximum staleness |
|---|---|---|---:|---:|
| `BC_2YEAR` | `macro__treasury_2y_level` | percent | 1 | 5 calendar days |
| `BC_5YEAR` | `macro__treasury_5y_level` | percent | 1 | 5 calendar days |
| `BC_10YEAR` | `macro__treasury_10y_level` | percent | 1 | 5 calendar days |
| 10Y minus 2Y | `macro__yield_curve_10y_2y` | percentage points | inherited from aligned inputs | inherited from aligned inputs |

The slope is computed only after the 10-year and 2-year levels have passed their
availability and staleness rules.

### 5.2 Effective Federal Funds Rate

| Field | Definition |
|---|---|
| Publisher | Federal Reserve Bank of New York |
| Landing page | <https://www.newyorkfed.org/markets/reference-rates/effr> |
| API | <https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json> |
| Source field | `percentRate` |
| Canonical feature | `macro__effective_fed_funds` |
| Unit | Percent |
| Model-session lag | 1 |
| Maximum staleness | 5 calendar days |
| Role | Model context: policy/overnight funding |

The current normalized adapter retains the effective date and `percentRate`.
Research using revision-specific claims should extend the schema to preserve every
revision field exposed by the API. Attribution and non-endorsement requirements
from the New York Fed terms remain applicable.

### 5.3 Return-derived market features

These five features are computed exclusively from target returns observed by the
after-close decision timestamp on endpoint \(t\). Their source and availability
dates are therefore both \(t\), but they may be used only to forecast targets
beginning at \(t+1\).

Let \(r_{i,t}\) be the 12 industry returns,
\(\bar r_t=12^{-1}\sum_i r_{i,t}\),
\(g_t=\log(1+\bar r_t)\), and
\(\Delta\tau_t=\text{calendar days in interval }t/365.25\).

| Canonical feature | Implemented definition | Unit |
|---|---|---|
| `macro__industry_market_return` | \(\bar r_t\), the cross-sectional mean of the 12 industry returns | decimal simple return |
| `macro__realized_volatility_20d` | \(\sqrt{\sum g_s^2/\sum\Delta\tau_s}\) over the latest 20 common-endpoint observations | elapsed-calendar-time annualized decimal volatility |
| `macro__downside_semivolatility_20d` | \(\sqrt{\sum \min(g_s,0)^2/\sum\Delta\tau_s}\) over the latest 20 common-endpoint observations | elapsed-calendar-time annualized decimal downside semivolatility |
| `macro__industry_dispersion_20d` | Daily cross-sectional sample standard deviation (`ddof=1`) across 12 industries, averaged over 20 observations | decimal return dispersion |
| `macro__market_drawdown_60d` | Wealth \(W_t=\prod_{s\le t}(1+\bar r_s)\), divided by its 60-observation rolling maximum, minus 1 | decimal drawdown |

The rolling windows count common-endpoint observations rather than asserting a
fixed number of exchange sessions or calendar days. Volatility and downside
semivolatility divide accumulated squared log returns by the actual elapsed
calendar years in each window, so a multi-day interval is not treated as one
ordinary daily observation.

## 6. Validation-only labels

OFR FSI is intentionally excluded from the model matrix because its market-based
components overlap the target and context variables. It is retained for external
labeling, diagnostic agreement, and robustness checks.

| Field | Definition |
|---|---|
| Publisher | Office of Financial Research |
| Landing page | <https://www.financialresearch.gov/financial-stress-index/> |
| CSV | <https://www.financialresearch.gov/financial-stress-index/data/fsi.csv> |
| Model-session lag | 2 |
| Maximum staleness | 6 calendar days |
| Model inclusion | No |

| Source field | Canonical label | Unit |
|---|---|---|
| `OFR FSI` | `validation__ofr_financial_stress` | standard deviations |
| `Credit` | `validation__ofr_credit_stress` | contribution |
| `Funding` | `validation__ofr_funding_stress` | contribution |
| `Volatility` | `validation__ofr_volatility_stress` | contribution |

The `validation_only_features_excluded` quality gate asserts that no
`validation__*` column enters `model_matrix.parquet`.

## 7. Optional and excluded sources

### 7.1 Licensed option-data extension only

VIX, SKEW, VVIX, option surfaces, implied moments, and related option-derived tail
features may be added only from a licensed source such as Cboe DataShop or
OptionMetrics. They require:

- a separate catalog and source adapter;
- the licensed observation/publication timestamp rather than an assumed website
  date;
- explicit permission and redistribution constraints;
- no raw licensed values in public fixtures or artifacts; and
- a registered comparison proving that the shorter or different history does not
  make results incomparable with the public core.

`data/interim/optional_diagnostics.parquet` is currently a zero-column,
6,593-row schema-stable placeholder. No Cboe raw file is cached by the public
pipeline.

### 7.2 FRED and ALFRED

The public pipeline does not request FRED or ALFRED. As reviewed on 2026-07-28,
FRED's current Terms of Use prohibit using FRED services or content to develop or
train machine-learning, deep-learning, or generative-AI software. Routing an
originating institution's series through FRED does not change that restriction.

Accordingly, CrisisForge obtains permitted public inputs directly from their
originating institutions. A future terms change requires a dated governance review
and catalog revision; it must not be activated silently.

## 8. Availability, timing, missingness, and leakage

1. **Decision clock.** The Phase 0 origin is after the U.S. close in
   `America/New_York`. It is not an intraday or pre-close design.
2. **Actual model-session lags.** `availability_lag_model_sessions` is measured on
   the ordered target/model calendar, not with pandas weekday offsets. For a source
   date, the pipeline finds its insertion position in actual target dates and adds
   the configured number of model sessions.
3. **Backward as-of only.** A source value is admitted only when
   `available_date <= model_date`. The join never uses a future observation and
   never backward-fills from the future.
4. **Calendar-day staleness.** Age is measured from `source_date` to `model_date` in
   calendar days. Values exceeding the catalog limit are set to null.
5. **Targets are never imputed.** All 15 target returns must be jointly observed.
6. **Context is not imputed.** Missing or stale context rows are dropped when the
   clean model matrix is formed.
7. **Return-derived timing.** Features computed at close \(t\) may condition only
   targets beginning at \(t+1\).
8. **No Phase 0 fitting.** Phase 0 fits no scaler, imputer, PCA/factor model, regime
   model, EVT threshold, or calibration map.
9. **Chronological split.** Train ends 2013-12-31, validation ends 2019-12-31, and
   test starts 2020-01-02 in the current clean calendar.
10. **Later horizon experiments.** Any \(H\)-step forecast must add the applicable
    purge/embargo and fit every transform inside the training fold.
11. **Regime information.** Later forecasting may use filtered state probabilities
    and predictive state paths only. Smoothed full-sample states remain
    retrospective diagnostics.
12. **Current-vintage limitation.** Public histories can be revised; a current
    snapshot is not a historical point-in-time vintage.

`data/interim/availability_alignment_audit.parquet` records aligned values and,
where applicable, `source_date__*`, `available_date__*`, and `stale__*` fields.
Return-derived features receive source and availability date \(t\) and `stale=False`
under the after-close \(t\)-to-\(t+1\) contract.

## 9. Snapshot and refresh integrity

### 9.1 Raw snapshot metadata

Each normalized raw CSV has a sibling `.meta.json` containing:

- `source_id` and provider;
- exact source URL or URLs;
- UTC retrieval timestamp;
- remote payload SHA-256;
- normalized local CSV SHA-256;
- row count and exact ordered columns; and
- first and last dates.

Before a cached snapshot is used, the loader verifies:

- required metadata fields are present;
- `source_id`, provider, landing-page identity, and catalog schema match the active
  catalog;
- local CSV SHA-256 matches metadata;
- exact schema and column order match;
- row count matches;
- dates parse successfully;
- dates are unique and monotonically increasing; and
- first and last dates exactly match metadata;
- metadata prove that the cache covers the requested sample; and
- the returned frame is filtered to the requested start and end on every read.

The complete raw cache must also match the catalog allowlist exactly. Missing
configured files and uncatalogued leftovers, including a stale restricted-data
snapshot, are hard errors.

### 9.2 Atomic refresh

`--refresh` and `--offline` are mutually exclusive. A refresh downloads every
configured source into a temporary `.raw-staging-*` directory. Each CSV and
metadata file is first written to a temporary file and promoted with `os.replace`.
The staged raw candidate is then read, filtered, transformed, and tested against
all 23 quality gates before it can replace `data/raw`.

After the candidate passes, Phase 0 treats the raw cache and all three publication
trees as one transaction:

- `data/raw`;
- `data/interim`;
- `data/processed`; and
- `artifacts/phase0`.

The prior versions are retained as sibling backups. Any exception while writing a
Parquet file, quality report, manifest, or receipt restores all four previous trees
exactly. Backups are deleted only after the complete publication succeeds.
Adversarial tests inject failures during Parquet and manifest writes and require
every protected file hash, as well as the prior manifest verification, to remain
unchanged.

The Git ignore policy covers `.raw-staging-*`, raw failed/backup trees, and
interim/processed publication backups. This prevents a process killed before
cleanup from exposing raw or licensed observations as accidental Git add
candidates.

### 9.3 Freshness

Freshness is evaluated against the requested end date, or the run date when
`sample.end_date` is null:

- more than 45 calendar days old: warning;
- more than 180 calendar days old: failed gate.

In the audited run, Treasury was 1 day old, NY Fed EFFR 4 days, OFR 5 days, and the
French file 60 days. These values are not permanent source guarantees.

## 10. Computed quality gates

Phase 0 computes and records these 23 gates:

1. `raw_cache_matches_catalog_allowlist`
2. `minimum_asset_coverage`
3. `target_index_unique`
4. `target_index_increasing`
5. `common_interval_endpoints_aligned`
6. `target_source_internal_gaps_within_limit`
7. `target_source_start_delays_within_limit`
8. `relative_target_calendar_density`
9. `model_context_source_coverage`
10. `model_context_source_internal_gaps_within_limit`
11. `validation_source_coverage`
12. `validation_source_internal_gaps_within_limit`
13. `model_index_relative_density`
14. `model_index_internal_gap_within_limit`
15. `common_interval_calendar_gap_within_limit`
16. `interval_source_observation_count_gap_within_limit`
17. `model_matrix_finite`
18. `model_index_unique`
19. `source_dates_not_after_model_date`
20. `availability_dates_not_after_model_date`
21. `strict_chronological_splits`
22. `validation_only_features_excluded`
23. `source_freshness_within_failure_limit`

The context gates measure usable aligned coverage and maximum internal gaps
separately for model inputs and validation-only labels. The model-index gates then
test the density and maximum internal gap of the final complete-case matrix. This
prevents an internally missing year of EFFR or OFR data from disappearing silently
through the final `dropna`.

Status is `failed` if any gate fails, otherwise `passed_with_warnings` if warnings
exist, and `passed` otherwise. The pipeline writes the quality report, manifest,
and receipt, then raises `DataQualityError` when computed gates failed.

Outlier returns above the configured absolute threshold are retained and surfaced
as warnings; they are not silently winsorized or deleted.

## 11. Canonical files and schemas

Phase 0 wide files use a `DatetimeIndex` and these prefixes:

- `asset__*`: target simple returns;
- `macro__*`: model context;
- `validation__*`: external labels excluded from training;
- `diagnostic__*`: optional diagnostics excluded from training;
- `source_date__*`: originating observation date;
- `available_date__*`: first admissible model date; and
- `stale__*`: whether the source exceeded its calendar-day cutoff.

| Path | Grain | Contents |
|---|---|---|
| `data/raw/returns/french_12_industries.{csv,meta.json}` | Source observation | Normalized French snapshot and provenance |
| `data/raw/macro/us_treasury_yield_curve.{csv,meta.json}` | Source observation | Official par-yield snapshot and provenance |
| `data/raw/macro/nyfed_effr.{csv,meta.json}` | Source observation | EFFR snapshot and provenance |
| `data/raw/macro/ofr_fsi.{csv,meta.json}` | Source observation | OFR FSI snapshot and provenance |
| `data/interim/target_returns.parquet` | Common-endpoint interval | 15 target returns |
| `data/interim/target_interval_audit.parquet` | Common-endpoint interval | Interval duration, source endpoints, and observation counts |
| `data/interim/macro_features.parquet` | Target endpoint | 10 public model features |
| `data/interim/validation_labels.parquet` | Target endpoint | Four OFR validation labels |
| `data/interim/optional_diagnostics.parquet` | Target endpoint | Empty public-core placeholder |
| `data/interim/availability_alignment_audit.parquet` | Target endpoint | Values plus source, availability, and staleness evidence |
| `data/processed/model_matrix.parquet` | Joint clean endpoint | 15 targets plus 10 model features |
| `data/processed/model_matrix.parquet` plus `configs/pipeline.yaml` | Joint clean endpoint | Logical train, validation, and sealed-test views; splits are not materialized as separate files |

## 12. Manifest and run receipt

`artifacts/phase0/manifest.json` is deterministic for the same inputs, governed
files, and serialization. It has no creation timestamp. It stores:

- manifest schema version `1.0`;
- pipeline version `phase0-v0.3.0`;
- catalog and pipeline config paths; and
- sorted repository-relative file records with byte size and SHA-256.

The manifest covers raw CSV/metadata, interim and processed outputs, quality report
JSON/Markdown, governed configuration, documentation, source code, tests, and the
selected repository control files. It does not use the timestamped run receipt or
the mutable `experiments/registry.csv` as an input. The registry refers to Phase 0
hashes and is therefore excluded to avoid a circular hash dependency; every
completed experiment stores the exact Phase 0 manifest hash in its own local run
receipt.

`artifacts/phase0/run_receipt.json` is deliberately timestamped separately. It
records run mode (`refresh`, `online-cache`, or `offline`), status, row/column
counts, split counts, warnings, failed gates, Python/platform/package versions, Git
commit when available, and the manifest path, SHA-256, and file count.

The completed experiments are tied to the content-addressed Phase 0 snapshot
`f051ec35236a481858b67c5b1e7136f1698036832f427efba521dbf3fcd36d70`.
That manifest is preserved locally and must not be overwritten after downstream
runs. A future data refresh creates a new Phase 0 lineage and requires new
dependent experiments rather than relabeling the existing evidence.

## 13. Licensed applied-panel substitutions

| Component | Preferred licensed source | Evidential improvement |
|---|---|---|
| Security, ETF, index, and Treasury total returns | CRSP via WRDS | Adjusted total returns, delistings, stable identifiers, Treasury files |
| Cross-asset histories | Bloomberg, LSEG, or FactSet | Vendor-maintained adjusted histories and broader coverage |
| Credit OAS and Treasury volatility | ICE Data Services | Long credit and MOVE histories |
| Option-implied distributions | OptionMetrics or Cboe DataShop | Licensed point-in-time surfaces, VIX/SKEW/VVIX, and implied tail measures |
| Futures and roll-aware returns | CME DataMine and relevant exchange/vendor feeds | Tradable settlements and explicit roll methodology |
| Vintage macro releases | Haver or another licensed point-in-time archive | Historical vintages and publication timestamps |

Paid data do not eliminate leakage. Vendor timestamps, revisions, survivorship,
corporate actions, stale quotes, and universe construction still require explicit
tests and experiment metadata.

## 14. Data-release checklist

Before a result is cited:

- the catalog names the original publisher and exact endpoint;
- source terms and redistribution status have a dated review;
- raw metadata pass hash, schema, row-count, order, and date-range validation;
- the common-interval audit and all 23 quality gates pass;
- source and availability dates do not exceed model dates;
- target units and return types are explicit;
- French and Treasury proxy labels appear in every relevant caption;
- train/validation/test boundaries and any horizon purge/embargo are frozen;
- the experiment references its exact content-addressed manifest and run receipt;
  and
- no public-core result is described as licensed-panel or investable evidence.
