"""Type-tuple similarity plumbing.

Stores a symmetric distance between every pair of ``(main_type, subtype)``
tuples in the corpus. Distance is an integer ordinal: ``0`` means identical,
higher = further, ``100`` = definitely no match.

Storage
-------
A long-form CSV at ``data/type_similarity.csv`` with columns::

    subject_main_type, subject_subtype, comp_main_type, comp_subtype, distance

One row per explicitly-set pair. Empty-string subtypes represent the
null-subtype sentinel (properties whose ``property_subtype`` is missing).

The CSV is compact: only the diagonal (identical tuples, distance=0) and
any curated non-default values are emitted. Missing rows get defaults
from ``type_distance`` -- cross-main pairs default to ``100`` (different
main_type = no match), same-main pairs fall back to the diagonal = ``0``.

Symmetric by construction -- stored once, mirrored on load.

Usage
-----
    from comp_ranker.scoring.type_similarity import type_distance, allowed_by_distance

    type_distance("Office", "High-Rise", "Office", "High-Rise")   # 0.0 (identity)
    type_distance("Office", "High-Rise", "Office", "Low-Rise")    # 0.0 (same-main default)
    type_distance("Office", "High-Rise", "Multifamily", "Garden") # 100.0 (cross-main default)
    type_distance("UnknownType", "", "Office", "")                # None (type not in CSV)

    # As a filter:
    allowed_by_distance("Office", "High-Rise", "Multifamily", "Garden",
                        max_distance=50)  # False (100 > 50)
"""
from __future__ import annotations

import functools
import os
from typing import Optional

import pandas as pd

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
_CSV_PATH = os.path.abspath(os.path.join(_ROOT, "data", "type_similarity.csv"))

_EXPECTED_COLS = [
    "subject_main_type",
    "subject_subtype",
    "comp_main_type",
    "comp_subtype",
    "distance",
]


@functools.lru_cache(maxsize=1)
def load_type_similarity(
    path: str | None = None,
) -> dict[tuple[str, str, str, str], float]:
    """Load the similarity CSV into a flat ``{(subj_m, subj_s, comp_m, comp_s): distance}`` dict.

    Missing / blank subtypes are coerced to the empty string ``""`` (the
    null-subtype sentinel). Rows with a blank distance are dropped.

    **Symmetry:** every stored entry ``(A, a, B, b) -> d`` with
    ``(A, a) != (B, b)`` is mirrored as ``(B, b, A, a) -> d`` so downstream
    lookups work in either direction. If both directions are present in the
    CSV with different values, a ``ValueError`` is raised naming the pair.

    Cached per session -- call ``load_type_similarity.cache_clear()`` after
    regenerating the CSV to pick up new values in the same process.
    """
    csv_path = path or _CSV_PATH
    if not os.path.exists(csv_path):
        return {}

    # ``keep_default_na=False, na_values=[""]`` keeps literal string values
    # like "nan" and "NA" as-is in the type-label positions (they can be real
    # ``main_type`` values) while still treating blank distance cells as NaN so
    # we can drop them below.
    df = pd.read_csv(
        csv_path,
        dtype={
            "subject_main_type": str,
            "subject_subtype": str,
            "comp_main_type": str,
            "comp_subtype": str,
        },
        keep_default_na=False,
        na_values=[""],
    )

    missing = [c for c in _EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path}: missing columns {missing!r}; expected {_EXPECTED_COLS!r}"
        )

    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df = df.dropna(subset=["distance"])

    # Coerce NaN subtypes back to "" -- pandas strips them via na_values.
    for col in ("subject_subtype", "comp_subtype"):
        df[col] = df[col].fillna("").astype(str)

    table: dict[tuple[str, str, str, str], float] = {}
    for row in df.itertuples(index=False):
        key = (
            str(row.subject_main_type),
            str(row.subject_subtype),
            str(row.comp_main_type),
            str(row.comp_subtype),
        )
        dist = float(row.distance)
        if key in table and table[key] != dist:
            raise ValueError(
                f"{csv_path}: duplicate row for {key!r} with conflicting "
                f"distances {table[key]} vs {dist}"
            )
        table[key] = dist

    # Mirror on load: every entry (A, a, B, b) also inserted as (B, b, A, a).
    mirrored: dict[tuple[str, str, str, str], float] = {}
    for (sm, ss, cm, cs), dist in table.items():
        mirrored[(sm, ss, cm, cs)] = dist
        if (sm, ss) == (cm, cs):
            continue
        rev = (cm, cs, sm, ss)
        if rev in table and table[rev] != dist:
            raise ValueError(
                f"{csv_path}: asymmetric entries for {(sm, ss)!r} <-> "
                f"{(cm, cs)!r}: forward {dist} vs reverse {table[rev]}. "
                f"This module requires symmetric distances -- edit one side "
                f"to match, or delete the duplicate."
            )
        mirrored[rev] = dist

    return mirrored


def _norm(v: Optional[str]) -> Optional[str]:
    """Coerce missing/NaN subtypes to '' (the sentinel). None stays None."""
    if v is None:
        return None
    if isinstance(v, float):  # pandas NaN leaks through as float("nan")
        return None
    if not isinstance(v, str):
        return None
    return v


def type_distance(
    subject_main: Optional[str],
    subject_subtype: Optional[str],
    comp_main: Optional[str],
    comp_subtype: Optional[str],
) -> Optional[float]:
    """Return the distance for two ``(main_type, subtype)`` tuples.

    Resolution order:
      1. Either ``main_type`` None/NaN -> None (can't place on the type grid).
      2. ``subtype`` None/NaN coerced to ``""`` (null-subtype sentinel).
      3. Identity: ``(subj_main, subj_sub) == (comp_main, comp_sub)`` -> 0.0.
      4. Exact lookup in the CSV-backed dict.
      5. Fallback: main-type-only row ``(subj_main, "", comp_main, "")``.
      6. Default: if both main_types are known (their diagonal row was
         emitted) and ``subj_main != comp_main``, returns 100.0 -- the
         cross-main "no match" default. Same-main unspecified pairs
         always resolve via step 5 (the diagonal row of the null-sub
         sentinel is 0).
      7. If the main_type is unknown (no diagonal row in the CSV),
         returns None. This prevents silently scoring unknown types.

    Symmetry is handled by load-time mirroring, so
    ``type_distance(A, a, B, b) == type_distance(B, b, A, a)`` by construction.
    """
    sm = _norm(subject_main)
    cm = _norm(comp_main)
    if sm is None or cm is None:
        return None

    ss = _norm(subject_subtype) or ""
    cs = _norm(comp_subtype) or ""

    if (sm, ss) == (cm, cs):
        return 0.0

    table = load_type_similarity()
    dist = table.get((sm, ss, cm, cs))
    if dist is not None:
        return dist

    # Fallback: main-type-only row (subtype=="" on both sides).
    dist = table.get((sm, "", cm, ""))
    if dist is not None:
        return dist

    # Default for cross-main missing pairs: 100 ("no match"). Only applies
    # when both main_types are known (their diagonal row was regenerated);
    # otherwise we don't know the type, so we return None.
    if sm != cm:
        sm_known = (sm, "", sm, "") in table
        cm_known = (cm, "", cm, "") in table
        if sm_known and cm_known:
            return 100.0

    return None


def allowed_by_distance(
    subject_main: Optional[str],
    subject_subtype: Optional[str],
    comp_main: Optional[str],
    comp_subtype: Optional[str],
    max_distance: Optional[float],
) -> bool:
    """Filter helper. Returns True when the pair should be kept.

    - ``max_distance is None`` -> True (no filter, default behaviour).
    - Else: True iff ``type_distance(...) <= max_distance``. Missing pairs
      (lookup returns None) fail the filter -- conservative default.
    """
    if max_distance is None:
        return True
    dist = type_distance(subject_main, subject_subtype, comp_main, comp_subtype)
    if dist is None:
        return False
    return dist <= max_distance


def known_tuples() -> list[tuple[str, str]]:
    """Return the sorted list of ``(main_type, subtype)`` tuples the CSV knows about."""
    table = load_type_similarity()
    seen: set[tuple[str, str]] = set()
    for (sm, ss, cm, cs) in table.keys():
        seen.add((sm, ss))
        seen.add((cm, cs))
    return sorted(seen)
