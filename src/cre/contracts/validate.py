"""Apply a contract: keep valid rows, quarantine invalid ones, fail loudly when too many are bad.

Policy (see docs/DATA_CONTRACTS.md):
- *Schema-level* errors (missing column, wrong dtype) -> raise. The source changed; a human must look.
- *Row-level* errors -> those rows go to `quarantine` with the failed check; the rest proceed.
- If the invalid share exceeds `max_invalid_frac` -> raise: something systemic is wrong, and
  silently training/serving on the remaining rows would be worse than stopping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandera.errors as pe
import pandera.polars as pa
import polars as pl

from cre.logging_utils import log_event

log = logging.getLogger("cre.contracts")


class ContractViolation(RuntimeError):
    pass


@dataclass
class ValidationResult:
    valid: pl.DataFrame
    quarantine: pl.DataFrame  # invalid rows + `failed_checks` column
    n_in: int

    @property
    def invalid_frac(self) -> float:
        return 0.0 if self.n_in == 0 else self.quarantine.height / self.n_in


def validate_or_quarantine(
    df: pl.DataFrame, schema: type[pa.DataFrameModel], table: str, max_invalid_frac: float = 0.01
) -> ValidationResult:
    n = df.height
    df = df.with_row_index("_row")
    try:
        schema.validate(df, lazy=True)
        res = ValidationResult(df.drop("_row"), df.head(0).drop("_row").with_columns(failed_checks=pl.lit("")), n)
    except pe.SchemaErrors as e:
        fc = e.failure_cases
        row_level = fc.filter(pl.col("index").is_not_null())
        schema_level = fc.filter(pl.col("index").is_null())
        if schema_level.height:
            raise ContractViolation(f"{table}: schema-level contract failure:\n{schema_level}") from e
        reasons = (
            row_level.with_columns(_row=pl.col("index").cast(pl.UInt32))
            .group_by("_row")
            .agg(failed_checks=pl.col("check").cast(pl.Utf8).unique().sort().str.join("; "))
        )
        bad = df.join(reasons, on="_row", how="inner").drop("_row")
        good = df.join(reasons, on="_row", how="anti").drop("_row")
        res = ValidationResult(good, bad, n)
    except pe.SchemaError as e:  # non-lazy path (e.g. column missing entirely)
        raise ContractViolation(f"{table}: {e}") from e

    log_event(
        log,
        "contract_checked",
        table=table,
        rows=n,
        invalid=res.quarantine.height,
        invalid_frac=round(res.invalid_frac, 5),
    )
    if res.invalid_frac > max_invalid_frac:
        raise ContractViolation(
            f"{table}: {res.invalid_frac:.2%} invalid rows > {max_invalid_frac:.2%} threshold; "
            f"top reasons: {res.quarantine['failed_checks'].value_counts(sort=True).head(3).rows()}"
        )
    return res


@dataclass
class Freshness:
    source: str
    latest: datetime | None
    checked_at: datetime
    max_age: timedelta

    @property
    def age(self) -> timedelta | None:
        return None if self.latest is None else self.checked_at - self.latest

    @property
    def status(self) -> str:
        """FRESH / STALE / MISSING; consumers decide (live scoring falls back when not FRESH)."""
        if self.latest is None:
            return "MISSING"
        return "FRESH" if self.age <= self.max_age else "STALE"


def check_freshness(source: str, latest: datetime | None, now: datetime, max_age: timedelta) -> Freshness:
    f = Freshness(source, latest, now, max_age)
    log_event(
        log,
        "freshness_checked",
        level=logging.INFO if f.status == "FRESH" else logging.WARNING,
        source=source,
        status=f.status,
        age_min=None if f.age is None else round(f.age.total_seconds() / 60, 1),
    )
    return f
