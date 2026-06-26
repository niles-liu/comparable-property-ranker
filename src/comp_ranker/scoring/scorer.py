"""Rule-based property similarity scoring engine.

This is the interpretable baseline comparator. Each scoring factor starts at
100 and loses points as the subject and candidate diverge; the composite is a
weighted average over the factors that have data on both sides.

Scoring summary
---------------
Each factor starts at 100 and loses points:
  location       : -10 pts per metro-specific distance step (default 0.25 mi)
  size           : -1 pt per 1% symmetric size variance
  year_built     : -2 pts per year difference
  year_renovated : -2 pts per year difference  (skipped if either is None)
  bldg_class     : -20 pts if classes differ    (skipped if either is None)
  submarket      : 100 if same submarket, 0 if different (skipped if either is None)
  sale_recency   : 100 within 1 yr of as_of_date, -10 pts/yr after (skipped if None)

Null handling: factors with None inputs are excluded and their weights are
redistributed to the remaining factors.

Size metric: multifamily uses ``num_units`` as the size metric when present,
otherwise ``sqft``; all other types use ``sqft``.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..accel import xp, to_cpu, to_gpu, gpu_available

from .config import (
    DEFAULT_DISTANCE_STEP,
    MAX_RADIUS_MILES,
    SIZE_TOLERANCE,
    WEIGHTS,
    SIZE_PENALTY_PER_PCT,
    YEAR_PENALTY_PER_YEAR,
    CLASS_PENALTY_MISMATCH,
    RECENCY_PENALTY_PER_YEAR,
    RECENCY_GRACE_YEARS,
    load_metro_steps,
)


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in miles."""
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Individual factor scores (each returns float 0-100 or None)
# ---------------------------------------------------------------------------

def _location_score(distance_miles: float, step: float) -> float:
    """10-point penalty per distance step; floored to 0."""
    return max(0.0, 100.0 - math.floor(distance_miles / step) * 10.0)


def _size_score(size1: float | None, size2: float | None) -> float | None:
    """Symmetric variance: |a-b| / avg(a,b) * 100; 1 pt penalty per 1%."""
    if not size1 or not size2:
        return None
    variance_pct = abs(size1 - size2) / ((size1 + size2) / 2.0) * 100.0
    return max(0.0, 100.0 - variance_pct * SIZE_PENALTY_PER_PCT)


def _year_score(year1: int | None, year2: int | None) -> float | None:
    """2-point penalty per year of difference."""
    if year1 is None or year2 is None:
        return None
    return max(0.0, 100.0 - abs(year1 - year2) * YEAR_PENALTY_PER_YEAR)


def _class_score(class1: str | None, class2: str | None) -> float | None:
    """Full credit for exact match; CLASS_PENALTY_MISMATCH deducted otherwise."""
    if not class1 or not class2:
        return None
    match = class1.strip().upper() == class2.strip().upper()
    return 100.0 if match else max(0.0, 100.0 - CLASS_PENALTY_MISMATCH)


def _submarket_score(sub1: str | None, sub2: str | None) -> float | None:
    """Binary: 100 if same submarket, 0 if different."""
    if not sub1 or not sub2:
        return None
    return 100.0 if sub1.strip().upper() == sub2.strip().upper() else 0.0


def _recency_score(last_sale: date | None, as_of: date) -> float | None:
    """Full credit within grace window, RECENCY_PENALTY_PER_YEAR per year after."""
    if last_sale is None:
        return None
    years_ago = (as_of - last_sale).days / 365.25
    return max(0.0, 100.0 - max(0.0, years_ago - RECENCY_GRACE_YEARS) * RECENCY_PENALTY_PER_YEAR)


# ---------------------------------------------------------------------------
# Composite score (dynamic weight redistribution for None factors)
# ---------------------------------------------------------------------------

def _composite(factor_scores: dict[str, float | None], weights: dict[str, float]) -> float:
    """Weighted average of available (non-None) factor scores.

    Weights of None factors are redistributed proportionally to the rest.
    """
    available = {k: v for k, v in factor_scores.items() if v is not None}
    total_w = sum(weights.get(k, 0.0) for k in available)
    if total_w == 0:
        return 0.0
    return sum(score * weights.get(k, 0.0) / total_w for k, score in available.items())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_score(
    prop1: dict[str, Any],
    prop2: dict[str, Any],
    metro_steps: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
    as_of_date: date | None = None,
) -> dict[str, Any]:
    """Compute a pairwise similarity score between two property feature dicts.

    Parameters
    ----------
    prop1, prop2 : dict
        Unified feature dicts produced by ``features.normalize.from_row()``.
    metro_steps : dict, optional
        metro_name -> distance_step_miles.  Loaded from CSV if not supplied.
    weights : dict, optional
        Factor weights.  Uses ``scoring.config.WEIGHTS`` if not supplied.
    as_of_date : date, optional
        Reference date for recency scoring.  Defaults to today.

    Returns
    -------
    dict with composite_score, per-factor scores, the raw diffs, and a
    human-readable ``score_explanation``.
    """
    if metro_steps is None:
        metro_steps = load_metro_steps()
    if weights is None:
        weights = WEIGHTS
    if as_of_date is None:
        as_of_date = date.today()

    # Require lat/lon on both sides
    if any(prop.get(c) is None for prop in (prop1, prop2) for c in ("lat", "lon")):
        return {}

    # ---- distance ----------------------------------------------------------
    distance = haversine_miles(prop1["lat"], prop1["lon"], prop2["lat"], prop2["lon"])
    metro = (prop1.get("metro") or "").strip()
    step = metro_steps.get(metro, DEFAULT_DISTANCE_STEP)
    loc_score = _location_score(distance, step)

    # ---- size (prefer num_units for multifamily) ----------------------------
    def _pick_size(p: dict) -> float | None:
        t = (p.get("main_type") or "").lower()
        if "multifamily" in t or "apartment" in t:
            return float(p["num_units"]) if p.get("num_units") else p.get("sqft")
        return p.get("sqft")

    s1, s2 = _pick_size(prop1), _pick_size(prop2)
    sz_score = _size_score(s1, s2)
    size_variance_pct = (
        abs(s1 - s2) / ((s1 + s2) / 2.0) * 100.0 if s1 and s2 else None
    )

    # ---- year built / renovated --------------------------------------------
    yb_score = _year_score(prop1.get("year_built"), prop2.get("year_built"))
    yr_score = _year_score(prop1.get("year_renovated"), prop2.get("year_renovated"))

    # ---- building class ----------------------------------------------------
    cls_score = _class_score(prop1.get("bldg_class"), prop2.get("bldg_class"))
    bldg_match: bool | None = None
    if prop1.get("bldg_class") and prop2.get("bldg_class"):
        bldg_match = prop1["bldg_class"].strip().upper() == prop2["bldg_class"].strip().upper()

    # ---- submarket ---------------------------------------------------------
    sub_score = _submarket_score(prop1.get("submarket"), prop2.get("submarket"))

    # ---- sale recency (use comp's last sale date) --------------------------
    rec_score = _recency_score(prop2.get("last_sale_date"), as_of_date)

    # ---- composite ---------------------------------------------------------
    factor_scores = {
        "location":       loc_score,
        "size":           sz_score,
        "year_built":     yb_score,
        "year_renovated": yr_score,
        "bldg_class":     cls_score,
        "submarket":      sub_score,
        "sale_recency":   rec_score,
    }
    composite = _composite(factor_scores, weights)

    # ---- explanation -------------------------------------------------------
    explanation = (
        f"Distance {distance:.2f} mi (step {step} mi) -> location {loc_score:.0f}. "
    )
    if sz_score is not None:
        explanation += f"Size variance {size_variance_pct:.1f}% -> size {sz_score:.0f}. "
    else:
        explanation += "Size not compared (missing data). "
    if yb_score is not None:
        diff = abs((prop1.get("year_built") or 0) - (prop2.get("year_built") or 0))
        explanation += f"Year built diff {diff} yrs -> {yb_score:.0f}. "
    if yr_score is not None:
        diff = abs((prop1.get("year_renovated") or 0) - (prop2.get("year_renovated") or 0))
        explanation += f"Year renovated diff {diff} yrs -> {yr_score:.0f}. "
    if cls_score is not None:
        explanation += f"Class {'match' if bldg_match else 'mismatch'} -> {cls_score:.0f}. "
    if sub_score is not None:
        explanation += f"Submarket {'match' if sub_score == 100 else 'mismatch'} -> {sub_score:.0f}. "
    if rec_score is not None:
        explanation += f"Recency -> {rec_score:.0f}. "
    explanation += f"Composite: {composite:.1f}."

    return {
        "property_id_1":       prop1.get("property_id"),
        "property_id_2":       prop2.get("property_id"),
        "property_name_1":     prop1.get("property_name"),
        "property_name_2":     prop2.get("property_name"),
        "composite_score":     round(composite, 2),
        "location_score":      round(loc_score, 2),
        "size_score":          round(sz_score, 2) if sz_score is not None else None,
        "year_built_score":    round(yb_score, 2) if yb_score is not None else None,
        "year_renovated_score": round(yr_score, 2) if yr_score is not None else None,
        "bldg_class_score":    round(cls_score, 2) if cls_score is not None else None,
        "submarket_score":     round(sub_score, 2) if sub_score is not None else None,
        "sale_recency_score":  round(rec_score, 2) if rec_score is not None else None,
        "distance_miles":      round(distance, 2),
        "size_variance_pct":   round(size_variance_pct, 2) if size_variance_pct is not None else None,
        "year_built_diff":     abs(prop1["year_built"] - prop2["year_built"]) if prop1.get("year_built") and prop2.get("year_built") else None,
        "year_renovated_diff": abs(prop1["year_renovated"] - prop2["year_renovated"]) if prop1.get("year_renovated") and prop2.get("year_renovated") else None,
        "bldg_class_match":    bldg_match,
        "same_main_type":      prop1.get("main_type") == prop2.get("main_type"),
        "score_explanation":   explanation,
    }


def get_best_set(
    subject: dict[str, Any],
    candidates: pd.DataFrame,
    n: int = 10,
    metro_steps: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
    as_of_date: date | None = None,
) -> pd.DataFrame:
    """Return the top-n most similar properties for a subject from a candidates DataFrame.

    Hard filters applied before scoring:
      1. Same normalised main_type
      2. Within MAX_RADIUS_MILES
      3. Symmetric size variance <= SIZE_TOLERANCE  (or no size data on either side)
      4. Not the subject itself

    Scoring is fully vectorized (and GPU-accelerated when CuPy + a CUDA device
    are available; see :mod:`comp_ranker.accel`) so a single subject can be
    scored against a large candidate pool efficiently.

    Returns
    -------
    pd.DataFrame ranked by composite_score descending, with a rank_position column.
    """
    if metro_steps is None:
        metro_steps = load_metro_steps()
    if weights is None:
        weights = WEIGHTS
    if as_of_date is None:
        as_of_date = date.today()

    subject_id = subject.get("property_id")
    subject_type = subject.get("main_type")
    subject_lat = subject.get("lat")
    subject_lon = subject.get("lon")

    if subject_lat is None or subject_lon is None:
        raise ValueError("Subject property must have lat/lon for scoring.")

    # ==== Vectorized hard filters =========================================
    pool = candidates[
        (candidates["property_id"] != subject_id)
        & (candidates["main_type"] == subject_type)
        & candidates["lat"].notna()
        & candidates["lon"].notna()
    ]

    if pool.empty:
        return pd.DataFrame()

    # ---- vectorized haversine (GPU-accelerated when available) ------------
    lat1 = math.radians(subject_lat)
    lon1 = math.radians(subject_lon)
    lat2 = to_gpu(np.radians(pool["lat"].values.astype(np.float64)))
    lon2 = to_gpu(np.radians(pool["lon"].values.astype(np.float64)))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (xp.sin(dlat / 2.0) ** 2
         + math.cos(lat1) * xp.cos(lat2) * xp.sin(dlon / 2.0) ** 2)
    dist_arr = 3958.8 * 2.0 * xp.arctan2(xp.sqrt(a), xp.sqrt(1.0 - a))

    # Radius filter -- move mask to CPU for pandas boolean indexing
    within = to_cpu(dist_arr <= MAX_RADIUS_MILES)
    pool = pool[within].copy()
    dist_arr = dist_arr[within]

    if pool.empty:
        return pd.DataFrame()

    # ---- size filter (num_units for multifamily, else sqft) ---------------
    s_subj = subject.get("num_units") or subject.get("sqft")

    cand_units = to_gpu(pool["num_units"].values.astype(np.float64))
    cand_sqft = to_gpu(pool["sqft"].values.astype(np.float64))
    s_cand_filter = xp.where(~xp.isnan(cand_units) & (cand_units != 0),
                             cand_units, cand_sqft)

    if s_subj:
        avg_f = (s_subj + s_cand_filter) / 2.0
        variance_f = xp.abs(s_subj - s_cand_filter) / avg_f
        # Keep if variance OK *or* candidate has no size data
        size_ok = to_cpu(xp.isnan(s_cand_filter) | (variance_f <= SIZE_TOLERANCE))
        pool = pool[size_ok].copy()
        dist_arr = dist_arr[to_gpu(size_ok) if gpu_available() else size_ok]

    if pool.empty:
        return pd.DataFrame()

    n_pool = len(pool)

    # ==== Vectorized scoring (GPU-accelerated when available) =============
    cand_units = to_gpu(pool["num_units"].values.astype(np.float64))
    cand_sqft = to_gpu(pool["sqft"].values.astype(np.float64))

    # ---- size metric (type-aware, as in calculate_score) -----------------
    def _pick_size_subj() -> float | None:
        t = (subject.get("main_type") or "").lower()
        if "multifamily" in t or "apartment" in t:
            return float(subject["num_units"]) if subject.get("num_units") else subject.get("sqft")
        return subject.get("sqft")

    is_mf = pool["main_type"].str.lower().str.contains(
        "multifamily|apartment", na=False).values
    is_mf = to_gpu(is_mf)
    s_cand_score = xp.where(is_mf & ~xp.isnan(cand_units) & (cand_units != 0),
                            cand_units, cand_sqft)
    s_subj_score = _pick_size_subj()

    # ---- location score --------------------------------------------------
    metro_name = (subject.get("metro") or "").strip()
    step = metro_steps.get(metro_name, DEFAULT_DISTANCE_STEP)
    loc_scores = xp.maximum(0.0, 100.0 - xp.floor(dist_arr / step) * 10.0)

    # ---- size score ------------------------------------------------------
    if weights.get("size", 0) and s_subj_score is not None:
        avg_s = (s_subj_score + s_cand_score) / 2.0
        var_pct = xp.abs(s_subj_score - s_cand_score) / avg_s * 100.0
        sz_scores = xp.maximum(0.0, 100.0 - var_pct * SIZE_PENALTY_PER_PCT)
        no_size = xp.isnan(s_cand_score) | (s_cand_score == 0)
        sz_scores = xp.where(no_size, xp.nan, sz_scores)
        var_pct = xp.where(no_size, xp.nan, var_pct)
    else:
        sz_scores = xp.full(n_pool, xp.nan)
        var_pct = xp.full(n_pool, xp.nan)

    # ---- year built score ------------------------------------------------
    subj_yb = subject.get("year_built")
    if weights.get("year_built", 0) and subj_yb is not None:
        cand_yb = to_gpu(pd.to_numeric(pool["year_built"], errors="coerce").values)
        yb_diff = xp.abs(float(subj_yb) - cand_yb)
        yb_scores = xp.where(
            xp.isnan(cand_yb), xp.nan,
            xp.maximum(0.0, 100.0 - yb_diff * YEAR_PENALTY_PER_YEAR))
    else:
        yb_diff = xp.full(n_pool, xp.nan)
        yb_scores = xp.full(n_pool, xp.nan)

    # ---- year renovated score --------------------------------------------
    subj_yr = subject.get("year_renovated")
    if weights.get("year_renovated", 0) and subj_yr is not None:
        cand_yr = to_gpu(pd.to_numeric(pool["year_renovated"], errors="coerce").values)
        yr_diff = xp.abs(float(subj_yr) - cand_yr)
        yr_scores = xp.where(
            xp.isnan(cand_yr), xp.nan,
            xp.maximum(0.0, 100.0 - yr_diff * YEAR_PENALTY_PER_YEAR))
    else:
        yr_diff = xp.full(n_pool, xp.nan)
        yr_scores = xp.full(n_pool, xp.nan)

    # ---- building class score (string ops stay on CPU) -------------------
    subj_cls = subject.get("bldg_class")
    if weights.get("bldg_class", 0) and subj_cls:
        subj_cls_u = subj_cls.strip().upper()
        cand_cls_raw = pool["bldg_class"].fillna("").to_numpy(dtype=str, na_value="")
        cand_cls_u = np.char.upper(np.char.strip(cand_cls_raw))
        has_cls = cand_cls_u != ""
        cls_match_arr = cand_cls_u == subj_cls_u  # stays on CPU (string comparison)
        cls_scores = to_gpu(np.where(
            ~has_cls, np.nan,
            np.where(cls_match_arr, 100.0, max(0.0, 100.0 - CLASS_PENALTY_MISMATCH))))
    else:
        cls_match_arr = np.full(n_pool, False)
        cls_scores = xp.full(n_pool, xp.nan)

    # ---- submarket score (string ops stay on CPU) ------------------------
    subj_sub = subject.get("submarket")
    if weights.get("submarket", 0) and subj_sub:
        subj_sub_u = subj_sub.strip().upper()
        cand_sub_raw = pool["submarket"].fillna("").to_numpy(dtype=str, na_value="")
        cand_sub_u = np.char.upper(np.char.strip(cand_sub_raw))
        has_sub = cand_sub_u != ""
        sub_match_arr = cand_sub_u == subj_sub_u
        sub_scores = to_gpu(np.where(~has_sub, np.nan,
                              np.where(sub_match_arr, 100.0, 0.0)))
    else:
        sub_scores = xp.full(n_pool, xp.nan)

    # ---- sale recency score (dates already parsed at normalization time) --
    if weights.get("sale_recency", 0):
        cand_sale = pool["last_sale_date"]  # already datetime64 from normalize_unified_df
        as_of_ts = pd.Timestamp(as_of_date)
        years_ago = to_gpu((as_of_ts - cand_sale).dt.days.values / 365.25)
        rec_scores = xp.maximum(
            0.0, 100.0 - xp.maximum(0.0, years_ago - RECENCY_GRACE_YEARS) * RECENCY_PENALTY_PER_YEAR)
        rec_scores = xp.where(to_gpu(cand_sale.isna().values), xp.nan, rec_scores)
    else:
        rec_scores = xp.full(n_pool, xp.nan)

    # ==== Composite (dynamic weight redistribution) =======================
    factor_keys = [
        "location", "size", "year_built", "year_renovated",
        "bldg_class", "submarket", "sale_recency",
    ]
    all_scores = xp.column_stack([
        loc_scores, sz_scores, yb_scores, yr_scores,
        cls_scores, sub_scores, rec_scores,
    ])
    w_arr = xp.array([weights.get(k, 0.0) for k in factor_keys])

    valid = ~xp.isnan(all_scores)
    weighted = xp.where(valid, all_scores * w_arr, 0.0)
    total_w = (valid * w_arr).sum(axis=1)
    composite = xp.where(total_w > 0, weighted.sum(axis=1) / total_w, 0.0)

    # ==== Move results back to CPU for top-n selection & DataFrame build ==
    composite = to_cpu(composite)
    dist_arr = to_cpu(dist_arr)
    loc_scores = to_cpu(loc_scores)
    sz_scores = to_cpu(sz_scores)
    yb_scores = to_cpu(yb_scores)
    yr_scores = to_cpu(yr_scores)
    cls_scores = to_cpu(cls_scores)
    sub_scores = to_cpu(sub_scores)
    rec_scores = to_cpu(rec_scores)
    var_pct = to_cpu(var_pct)
    yb_diff = to_cpu(yb_diff)
    yr_diff = to_cpu(yr_diff)

    # ==== Select top-n (partial sort -- O(n) instead of O(n log n)) ========
    if n and n < len(composite):
        top_idx = np.argpartition(-composite, n)[:n]
        top_idx = top_idx[np.argsort(-composite[top_idx])]
    else:
        top_idx = np.argsort(-composite)

    # ==== Build result rows (explanations only for the top-n) =============
    pid_arr = pool["property_id"].values
    pname_arr = pool["property_name"].values

    def _safe_int(val) -> int | None:
        """Convert numpy scalar to int, returning None for NaN."""
        v = float(val)
        return None if math.isnan(v) else int(v)

    def _safe_round(val, ndigits=2) -> float | None:
        """Round a numpy scalar, returning None for NaN."""
        v = float(val)
        return None if math.isnan(v) else round(v, ndigits)

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        r_loc = float(loc_scores[idx])
        r_sz = float(sz_scores[idx])
        r_yb = float(yb_scores[idx])
        r_yr = float(yr_scores[idx])
        r_cls = float(cls_scores[idx])
        r_sub = float(sub_scores[idx])
        r_rec = float(rec_scores[idx])
        r_dist = float(dist_arr[idx])
        r_var = float(var_pct[idx])
        r_comp = float(composite[idx])

        # Explanation (identical format to calculate_score)
        expl = f"Distance {r_dist:.2f} mi (step {step} mi) -> location {r_loc:.0f}. "
        if not math.isnan(r_sz):
            expl += f"Size variance {r_var:.1f}% -> size {r_sz:.0f}. "
        else:
            expl += "Size not compared (missing data). "
        if not math.isnan(r_yb):
            expl += f"Year built diff {_safe_int(yb_diff[idx]) or 0} yrs -> {r_yb:.0f}. "
        if not math.isnan(r_yr):
            expl += f"Year renovated diff {_safe_int(yr_diff[idx]) or 0} yrs -> {r_yr:.0f}. "
        if not math.isnan(r_cls):
            expl += f"Class {'match' if cls_match_arr[idx] else 'mismatch'} -> {r_cls:.0f}. "
        if not math.isnan(r_sub):
            expl += f"Submarket {'match' if r_sub == 100 else 'mismatch'} -> {r_sub:.0f}. "
        if not math.isnan(r_rec):
            expl += f"Recency -> {r_rec:.0f}. "
        expl += f"Composite: {r_comp:.1f}."

        # bldg_class_match
        bldg_match_val = None
        if subj_cls and not math.isnan(r_cls):
            bldg_match_val = bool(cls_match_arr[idx])

        results.append({
            "rank_position":        rank,
            "property_id_1":        subject.get("property_id"),
            "property_id_2":        pid_arr[idx],
            "property_name_1":      subject.get("property_name"),
            "property_name_2":      pname_arr[idx],
            "composite_score":      round(r_comp, 2),
            "location_score":       round(r_loc, 2),
            "size_score":           _safe_round(r_sz),
            "year_built_score":     _safe_round(r_yb),
            "year_renovated_score": _safe_round(r_yr),
            "bldg_class_score":     _safe_round(r_cls),
            "submarket_score":      _safe_round(r_sub),
            "sale_recency_score":   _safe_round(r_rec),
            "distance_miles":       round(r_dist, 2),
            "size_variance_pct":    _safe_round(r_var),
            "year_built_diff":      _safe_int(yb_diff[idx]),
            "year_renovated_diff":  _safe_int(yr_diff[idx]),
            "bldg_class_match":     bldg_match_val,
            "same_main_type":       True,  # guaranteed by the type filter above
            "score_explanation":    expl,
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)
