# Research Log 0002 — Phase 0 Independent Audit and Remediation

**Date:** 2026-07-28  
**Status:** Resolved in Phase 0 v0.3.0 with one freshness warning

## Why this entry exists

An independent review was run after the first end-to-end Phase 0 build. The
review deliberately attempted to falsify the pipeline rather than confirm it.
It found material defects. They are retained here because the failed first
version is part of the research record.

## Material failures found

1. **Mixed holding intervals.** An inner join on dates combined French-industry
   and Treasury proxy returns whose preceding observations did not always share
   the same date. Fifty-eight retained rows in the first build represented
   different holding intervals, with a maximum observed discrepancy of 9.82
   percentage points in the audit calculation.
2. **Unauthenticated offline cache.** Editing a cached CSV without changing its
   metadata did not stop the pipeline.
3. **Literal pass flags.** Several quality outcomes were hard-coded rather than
   computed from configured gates.
4. **Treasury null observation and carry.** The official feed contained an
   all-null row on 2010-10-11, and the initial proxy assumed one-day carry even
   across longer gaps.
5. **Weak run governance.** The first manifest omitted code/configuration
   content, refresh was not atomic, and `--refresh --offline` was ambiguous.
6. **Publication risk.** Cboe index histories had been included before the
   public-release rights boundary was settled.
7. **Cached-range and identity drift.** An offline cache could ignore a changed
   requested sample, and metadata identity was not bound to the active catalog.
8. **Calendar-quality blind spots.** Removing a full year of target or context
   observations could still leave a finite matrix and a nominally passing run.
9. **Uncatalogued raw leftovers.** An extra raw snapshot could enter freshness and
   manifest evidence even when it was not in the public-core catalog.
10. **Incomplete publication rollback.** Raw rollback alone could leave partially
    rewritten interim, processed, or artifact files after a persistence exception.
11. **Interrupted-process exposure.** Raw staging and publication backup siblings
    under `data/` were not initially ignored by Git.

## Remediation

- Returns are now compounded over `(previous common endpoint, current common
  endpoint]` for every source. The separate interval audit records start, end,
  calendar duration, endpoint presence, and source observation counts.
- Every offline snapshot now requires metadata and validates SHA-256, row count,
  exact column order, date range, uniqueness, and monotonicity. Source ID,
  provider, landing page, catalog schema, and requested-range coverage are bound
  to the active configuration, and every read filters to the requested range.
- The raw cache must match the public-core catalog allowlist exactly.
- Treasury rows with all requested tenors missing are removed before
  differencing. Carry uses actual elapsed calendar time.
- Realized volatility and downside semivolatility divide squared log returns by
  actual elapsed calendar years rather than assigning every irregular endpoint
  one fixed trading-day scale.
- Quality status is derived from 23 named gates. They cover target-source
  coverage, start delay, internal gaps and density; common-interval duration and
  source-count disparity; model and validation context coverage and internal
  gaps; final model-index density and gap; index and finite-value invariants;
  chronological splits; feature-role separation; availability timing; raw
  allowlisting; and source freshness.
- A refresh is fully transformed and gated in staging before activation. Raw,
  interim, processed, and `artifacts/phase0` trees then publish as one rollback
  transaction. Any Parquet, quality, manifest, or receipt exception restores the
  complete prior release.
- Raw staging, failed, and backup siblings and interim/processed publication
  backups are Git-ignored so kill-9 leftovers cannot expose raw data for an
  accidental add.
- The deterministic content manifest covers raw snapshots and metadata,
  generated datasets, configuration, documentation, experiment/failure
  scaffolds, scripts, source code, tests, lockfile, Git-ignore policy, and project
  metadata. The v0.3.0 audit checkpoint covered 73 files. A separate timestamped
  receipt records execution mode and runtime.
- Cboe VIX, SKEW, and VVIX were removed from the public core. The replacement is
  a set of backward-looking features derived from the public target return
  panel. Licensed Cboe DataShop or OptionMetrics data remain optional applied
  extensions.

## Verification result

- Unit tests: 59 passed at the final Phase 0 v0.3.0 audit checkpoint.
- Ruff lint checks: passed.
- The integrated test count may increase as later stages are merged; the final
  release must replace this checkpoint count with the fresh integrated total.
- Refresh run: passed with warnings.
- Offline rerun: passed with the same content-manifest SHA-256 as the refresh
  run.
- Treasury all-null requested-tenor rows after normalization: 0.
- Common interval endpoints present across target sources: 100%.
- All 23 required quality gates passed.
- Remaining warning: the current Kenneth French daily file ends on 2026-05-29,
  60 calendar days before the requested end. The panel is therefore a
  retrospective research dataset, not a live-trading feed.

The independent audit also reran destructive and adversarial cases:

- a 2010–2012 requested sample stayed entirely inside that range;
- deleting and rehashing all 2008 French observations failed three target-calendar
  gates;
- deleting and rehashing all 2008 EFFR observations failed model-context coverage,
  context-gap, model-density, and model-gap gates;
- deleting and rehashing all 2008 OFR observations failed validation coverage and
  validation-gap gates;
- forged source ID, provider, landing page, and catalog schema were rejected;
- an uncatalogued `cboe_vix` raw pair was rejected;
- a refresh candidate containing a return below -100% never activated;
- injected Parquet and manifest persistence exceptions preserved every prior raw,
  interim, processed, and Phase 0 artifact hash and left the prior manifest valid;
- missing source dates were marked stale rather than treated as current;
- elapsed-time volatility and Treasury carry matched independent calculations;
  and
- refresh followed by two offline reruns produced the same deterministic manifest.

## Claim boundary

Different source session counts can still occur inside the same calendar
holding interval, for example when bond and equity markets observe different
holidays. This is not hidden. The target row is interpreted as the return over
the same common-endpoint interval, and the source-specific counts remain in the
audit table.
