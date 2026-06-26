"""Normalise raw source rows into the unified property feature schema.

See :mod:`comp_ranker.data.schema` for the full unified schema. This module
provides two entry points:

- :func:`normalize_unified_df` -- vectorized normalization applied once at
  corpus-build time (coerces dtypes, harmonises the occupancy scale across
  sources, and pre-computes normalized text columns for fuzzy lookup).
- :func:`from_row` -- normalise a single raw row (dict or ``pd.Series``) into
  the unified feature dict used by the rule-based scorer.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from comp_ranker.scoring.config import PROPERTY_TYPE_MAP


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def normalise_type(raw: Any) -> str | None:
    """Map a raw property type string to the canonical type."""
    if raw is None:
        return None
    return PROPERTY_TYPE_MAP.get(str(raw).strip().lower(), str(raw).strip())


def normalise_class(raw: Any) -> str | None:
    """Extract building class letter (A, B, C) from a raw string such as
    'Class A', 'class-b', 'A', 'Office - Class B', etc.

    Returns the raw value unchanged if no letter is found.
    """
    if raw is None:
        return None
    m = re.search(r"\b([ABC])\b", str(raw).upper())
    return m.group(1) if m else str(raw).strip() or None


def _clean(value: Any, cast=None) -> Any:
    """Return None for NaN/None/empty, otherwise cast if provided."""
    import math
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and not value.strip():
        return None
    return cast(value) if cast else value


# Numeric columns coerced at corpus-build time. Any column that is absent in a
# given corpus is simply skipped (its derived pairwise feature(s) become NaN,
# which the tree models handle natively).
_NUMERIC_COLS = (
    "lat", "lon", "sqft",
    "num_units", "year_built", "year_renovated",
    "num_stories", "occupancy", "land_area_ac", "parking_ratio",
    "avg_unit_size",
)


def normalize_unified_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ``from_row``-equivalent normalization to an entire DataFrame at once.

    Called once at corpus-build time so that per-query calls never need to
    re-normalize millions of rows. The result is what gets cached.
    """
    df = df.copy()

    # Numeric columns -- coerce bad values to NaN
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Occupancy is stored on two different scales across sources: some sources
    # use 0-100 (percent leased), others use 0-1 (fractional). Normalise to a
    # single 0-1 scale so the pairwise occupancy_diff is meaningful for
    # cross-source pairs. The 1.5 threshold separates the two scales (real
    # fractional occupancies cap at 1.0; percentages below 1.5% do not occur in
    # operational CRE data).
    if "occupancy" in df.columns:
        pct_mask = df["occupancy"] > 1.5
        df.loc[pct_mask, "occupancy"] = df.loc[pct_mask, "occupancy"] / 100.0
        df["occupancy"] = df["occupancy"].clip(lower=0.0, upper=1.0)

    # String column: property_subtype -- strip whitespace, normalize empties
    if "property_subtype" in df.columns:
        mask = df["property_subtype"].notna()
        stripped = df.loc[mask, "property_subtype"].astype(str).str.strip()
        bad = stripped.isin(("", "nan", "None", "NaN"))
        df.loc[mask, "property_subtype"] = stripped.where(~bad)

    # Date columns -- parse once so the scorer doesn't re-parse per call
    if "last_sale_date" in df.columns:
        df["last_sale_date"] = pd.to_datetime(df["last_sale_date"], errors="coerce")

    # String columns -- strip whitespace, convert empty / literal "nan" to NaN
    for col in ("property_id", "property_name", "full_address", "submarket", "metro"):
        if col not in df.columns:
            continue
        mask = df[col].notna()
        stripped = df.loc[mask, col].astype(str).str.strip()
        bad = stripped.isin(("", "nan", "None", "NaN"))
        df.loc[mask, col] = stripped.where(~bad)

    # Normalize bldg_class and main_type (small regex / dict lookup per value)
    if "bldg_class" in df.columns:
        df["bldg_class"] = df["bldg_class"].apply(normalise_class)
    if "main_type" in df.columns:
        df["main_type"] = df["main_type"].apply(normalise_type)

    # Pre-compute normalised text columns for fuzzy lookup so that
    # lookup_property() never has to re-normalise millions of rows at query time
    from comp_ranker.features.entity_resolution import normalize_address, _extract_state
    df["_norm_name"] = df["property_name"].apply(normalize_address)
    df["_norm_addr"] = df["full_address"].apply(normalize_address)
    df["_state"] = df["full_address"].apply(_extract_state)

    return df


def from_row(row: dict | Any, source: str | list[str]) -> dict:
    """Convert a single raw source row (dict or ``pd.Series``) into the
    unified feature schema. The row must already contain the standard unified
    column names.
    """
    if not isinstance(row, dict):
        row = row.to_dict()

    return {
        "property_id":    _clean(row.get("property_id"), str),
        "property_name":  _clean(row.get("property_name"), str),
        "full_address":   _clean(row.get("full_address"), str),
        "lat":            _clean(row.get("lat"), float),
        "lon":            _clean(row.get("lon"), float),
        "sqft":           _clean(row.get("sqft"), float),
        "num_units":      _clean(row.get("num_units"), lambda x: int(float(x))),
        "year_built":     _clean(row.get("year_built"), lambda x: int(float(x))),
        "year_renovated": _clean(row.get("year_renovated"), lambda x: int(float(x))),
        "bldg_class":     normalise_class(row.get("bldg_class")),
        "main_type":      normalise_type(row.get("main_type")),
        "submarket":      _clean(row.get("submarket"), str),
        "metro":          _clean(row.get("metro"), str),
        "last_sale_date": row.get("last_sale_date"),  # already date or None
        "source":         source,
    }
