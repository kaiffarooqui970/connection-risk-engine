# DATA CONTRACTS

Contracts live in `src/cre/contracts/schemas.py` (pandera, polars backend) and are enforced by
`validate_or_quarantine()` at every boundary where data enters the warehouse or the model.
Tests that feed deliberately broken data: `tests/test_contracts.py`, `tests/test_db_api.py`, `tests/test_weather.py`.

## Behaviour policy

| Situation | Behaviour | Why |
|---|---|---|
| Missing column / wrong dtype (*schema-level*) | **Raise `ContractViolation`**, load nothing | The source changed shape. Guessing would silently corrupt every downstream number |
| Invalid rows (*row-level*) | Valid rows load; invalid rows go to `quarantine.<table>` with a `failed_checks` reason | One bad row shouldn't block a month; quarantined rows stay inspectable |
| Invalid share > threshold (1% historical, 5% live, 2% weather) | **Raise**, load nothing for that batch | Systemic breakage: training/serving on "the rest" would be biased in unknown ways |
| Data **stale** (older than max age) | `check_freshness()` → `STALE`; live scoring uses the **fallback model** and flags reduced confidence (Phase 9) | A prediction from 3-hour-old delays is worse than an honest historical prior |
| Data **missing** | `MISSING` → same fallback; the monitoring job alerts | |
| Live credentials missing | Collector logs `collect_skipped` and exits 0 | A permanently red scheduled job gets ignored; a logged skip is visible but not noise |
| One station/hour fails at fetch | Logged to `live.fetch_log`; the rest of the cycle continues. Job fails only if >50% of calls fail | Partial data > no data; widespread failure is an incident |

Dtypes are **normalized at the boundary** (`hf_loader.normalize`: ns→µs timestamps, int32→int64) *before* validation,
so the contract checks one canonical form and any new upstream drift fails loudly. (This bit us on the first run;
see POSTMORTEM.)

## Tables

### `raw.stops` (HF `monthly_processed_data` subset), contract `RawStops`

| Column | Type | Null | Rule |
|---|---|---|---|
| id | str | no | unique; matches `^-?\d+-\d{10}-\d+$` (ride-start-stop#) |
| eva | str | no | 8 digits |
| train_line_station_num | int | no | 1…500 |
| arrival/departure_planned_time | datetime | yes | at least one non-null; planned dep ≥ planned arr |
| arrival/departure_change_time | datetime | yes | delay ∈ [−30, 1440] min **unless the stop is cancelled** (cancelled stops carry stale times) |
| arrival/departure_is_canceled | bool | no | |
| train_type, train_number, line_number, final_destination_station | str | yes | train_type null 0.07%: tolerated, mapped to `unknown` downstream |

Freshness: the HF dataset is published monthly. A training run requires every configured month to be present
(row count > 0); the `monitor` job warns if the newest month is more than 45 days old.

**First load result (2026-08…10):** 4,715,863 valid, 3,291 quarantined (0.07%):
3,138 with no planned time, 145 delay out of range, 8 planned dep < arr.

### `raw.weather` (Bright Sky), contract `Weather`

Key `(station_key, ts)` unique, `ts` in **UTC** (DST-safe). Ranges: temperature −40…50 °C,
precipitation 0…150 mm/h, wind 0…250 km/h, gust 0…300 km/h. Max age for live use: 3 h.
First load: 4 × 2,257 rows, 0 quarantined.

### `live.plan` / `live.changes` (DB Timetables v1), contracts `LivePlan` / `LiveChange`

- `live.plan`: PK `id`, upsert (latest plan wins).
- `live.changes`: PK `(id, fetched_at)`, append-only snapshot history (monitoring needs the history).
- `live.fetch_log`: one row per HTTP call, with status. This is the audit trail for freshness and coverage.
- Max age for live scoring: **2 h** since the last successful `fchg` for the station.

### Derived tables

`connections` (Phase 4) and `features` (Phase 5) have their own contracts, documented there. They are
enforced inside the pipeline rather than at ingestion.

## Live ingestion secrets (`.github/workflows/collect.yml`)

1. Register an app on the DB API Marketplace and subscribe to *Timetables*.
2. In the GitHub repo: *Settings → Secrets and variables → Actions → New repository secret*:
   `DB_CLIENT_ID`, `DB_API_KEY`.
3. Locally: `cp .env.example .env` and fill in. `.env` is git-ignored.

Without the secrets the hourly job runs, logs `collect_skipped`, and exits green.
The live store persists between runs via the GitHub Actions cache. **That is a demo-grade store:** caches unused
for 7 days are evicted. Production would write to object storage (S3/GCS) instead (see OPERATIONS).
