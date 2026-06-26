"""Transform (subject, comp) pairs into pairwise feature vectors for ranking.

Given a labeled_pairs DataFrame and the unified property table, this module
produces a feature matrix where each row represents a (subject, comp) pair with
relative/differential features suitable for a learning-to-rank model or a binary
classifier.

Feature sets
------------
The production model uses the **18-feature** set (``FEATURE_COLS_PLUS_OCC``):
the 16 base pairwise features plus signed/absolute occupancy difference. A
cross-type variant (``FEATURE_COLS_EXT``) adds three type-awareness features so
a model can learn which cross-type pairings (if any) experts accept, and a
further variant registers the optional curated ``type_distance`` feature.

Features produced
-----------------
Continuous:
    distance_miles          Haversine distance between subject and comp
    log_distance            log1p(distance_miles) -- compresses the long tail
    size_ratio              comp_size / subject_size (>1 = comp is bigger)
    log_size_ratio          log(size_ratio) -- symmetric around 0
    size_variance_pct       |s1-s2| / avg(s1,s2) * 100
    year_built_diff         comp_year_built - subject_year_built (signed)
    abs_year_built_diff     |year_built_diff|
    year_renovated_diff     comp_year_renovated - subject_year_renovated (signed)
    abs_year_renovated_diff |year_renovated_diff|
    sale_recency_years      years since comp's last sale (from as_of_date)
    stories_diff            comp_stories - subject_stories (signed)
    abs_stories_diff        |stories_diff|
    occupancy_diff          comp_occupancy - subject_occupancy (signed, fraction)
    abs_occupancy_diff      |occupancy_diff|
    land_area_ratio         comp_land_ac / subject_land_ac, clipped to [1e-3, 1e3]
    log_land_area_ratio     log(land_area_ratio) -- symmetric around 0
    parking_ratio_diff      comp_parking - subject_parking (signed)
    abs_parking_ratio_diff  |parking_ratio_diff|

Binary / ordinal:
    same_bldg_class         1 if building class matches, 0 otherwise
    same_submarket          1 if submarket matches, 0 otherwise
    bldg_class_ord_diff     ordinal distance (A=0, B=1, C=2); NaN if missing
    same_subtype            1 if property_subtype matches, 0 otherwise

Type-awareness (cross-type set only):
    same_main_type          1 if subject and comp share main_type, else 0
    subj_type_ord           ordinal encoding of subject main_type
    comp_type_ord           ordinal encoding of comp main_type
    type_distance           scalar from the curated type-similarity matrix

Categorical (for tree models that handle them natively):
    subject_main_type       canonical property type of the subject
    subject_metro           metro area of the subject
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from comp_ranker.scoring.type_similarity import type_distance


# Ordinal encoding for building class (A > B > C)
_CLASS_ORD = {"A": 0, "B": 1, "C": 2}

# Ordinal encoding for main_type (arbitrary stable order; tree models only need
# consistency, not a meaningful ordering). Used by the extended (cross-type)
# feature set so the model can learn specific subject/comp type pairings.
_MAIN_TYPE_ORD: dict[str, int] = {
    "Multifamily": 0, "Office": 1, "Retail": 2, "Industrial": 3, "Lodging": 4,
    "Land": 5, "Healthcare": 6, "Specialty": 7, "Mixed Use": 8,
    "Sports & Entertainment": 9, "Mobile Home": 10, "Dev Site": 11,
    "Self Storage": 12, "Other": 13,
}

# Base pairwise feature columns (22 columns) in stable order.
FEATURE_COLS: list[str] = [
    "distance_miles",
    "log_distance",
    "size_ratio",
    "log_size_ratio",
    "size_variance_pct",
    "year_built_diff",
    "abs_year_built_diff",
    "year_renovated_diff",
    "abs_year_renovated_diff",
    "sale_recency_years",
    "same_bldg_class",
    "same_submarket",
    "bldg_class_ord_diff",
    "stories_diff",
    "abs_stories_diff",
    "occupancy_diff",
    "abs_occupancy_diff",
    "land_area_ratio",
    "log_land_area_ratio",
    "same_subtype",
    "parking_ratio_diff",
    "abs_parking_ratio_diff",
]

# Extended (cross-type) feature set: FEATURE_COLS plus type-awareness features.
FEATURE_COLS_EXT: list[str] = FEATURE_COLS + [
    "same_main_type",        # 1 if subject and comp share main_type, else 0
    "subj_type_ord",         # ordinal encoding of subject main_type
    "comp_type_ord",         # ordinal encoding of comp main_type
]

# Optional curated pair-distance feature. Adds the scalar type_distance from the
# type-similarity matrix (data/type_similarity.csv). With the default 0/100
# scheme this is equivalent to same_main_type on an inverted scale -- useful
# only once cross-subtype / cross-main cells are curated. NaN when the lookup is
# missing; XGBoost handles NaN natively.
_TYPE_DISTANCE = ["type_distance"]

# 16-feature baseline: FEATURE_COLS_EXT minus the 9 sparser pairwise features.
_NEW_9 = [
    "stories_diff", "abs_stories_diff",
    "occupancy_diff", "abs_occupancy_diff",
    "land_area_ratio", "log_land_area_ratio",
    "same_subtype",
    "parking_ratio_diff", "abs_parking_ratio_diff",
]
FEATURE_COLS_16F: list[str] = [c for c in FEATURE_COLS_EXT if c not in _NEW_9]

# 18-feature production set: 16f + occupancy difference. Occupancy cleared all
# multi-seed significance tests (sign test 9/9, Wilcoxon p<0.01) -- the
# production feature width.
FEATURE_COLS_PLUS_OCC: list[str] = FEATURE_COLS_16F + ["occupancy_diff", "abs_occupancy_diff"]

# 19-feature variant: production 18f + curated type_distance.
FEATURE_COLS_PLUS_OCC_TSIM: list[str] = FEATURE_COLS_PLUS_OCC + _TYPE_DISTANCE

# Extended + type_distance.
FEATURE_COLS_EXT_TSIM: list[str] = FEATURE_COLS_EXT + _TYPE_DISTANCE

# Dispatch table for auto-selecting feature columns from a model's input width.
# predict_comps() uses this when feature_cols is not passed explicitly.
_FEATS_BY_WIDTH: dict[int, list[str]] = {
    16: FEATURE_COLS_16F,          # cross-type 16f
    18: FEATURE_COLS_PLUS_OCC,     # cross-type 18f (production)
    19: FEATURE_COLS_PLUS_OCC_TSIM,  # 18f + type_distance
    22: FEATURE_COLS,              # same-type base
    25: FEATURE_COLS_EXT,          # cross-type extended
    26: FEATURE_COLS_EXT_TSIM,     # 25f + type_distance
}

# Categorical columns (encoded separately or used by tree models directly)
CAT_COLS: list[str] = ["subject_main_type", "subject_metro"]


def build_pairwise_features(
    pairs: pd.DataFrame,
    unified: pd.DataFrame,
    as_of_date: date | None = None,
) -> pd.DataFrame:
    """Build pairwise feature vectors from labeled pairs.

    Parameters
    ----------
    pairs : DataFrame
        Must have columns: subject_property_id, comp_property_id.
        Optionally: comp_rank (used as the relevance label),
        subject_match_score, comp_match_score (fuzzy-match confidence).
    unified : DataFrame
        The unified property table (see :mod:`comp_ranker.data.schema`).
    as_of_date : date, optional
        Reference date for sale recency. Defaults to today.

    Returns
    -------
    DataFrame with one row per pair, containing the feature columns plus
    subject_property_id / comp_property_id (for joins), any label/match-score
    columns carried through, and an integer ``qid`` per subject for ranking
    models.

    Notes
    -----
    Vectorized via a two-sided merge of ``pairs`` against ``unified``: every
    feature is computed as a column operation on the merged frame. The only
    Python loop is over unique ``type_distance`` 4-tuples (cached), which is
    O(unique tuples) rather than O(rows) -- so a single subject scored against
    a large candidate pool resolves only a handful of tuples.
    """
    if as_of_date is None:
        as_of_date = date.today()

    if pairs.empty:
        return pd.DataFrame()

    keep_cols = [
        "property_id", "lat", "lon",
        "main_type", "num_units", "sqft",
        "year_built", "year_renovated", "last_sale_date",
        "bldg_class", "submarket",
        "num_stories", "occupancy", "land_area_ac", "parking_ratio",
        "property_subtype", "metro",
        "avg_unit_size",
    ]
    # Tolerate corpora that don't carry every optional column. Any absent column
    # makes its derived pairwise feature(s) all-NaN -- XGBoost handles missing
    # natively.
    keep_cols = [c for c in keep_cols if c in unified.columns]

    # When `pairs` covers <~30% of the unified universe (the labeled-pairs
    # case), pre-filter uni to just the needed IDs so the merge + dedup operate
    # on a small frame. For the inference case (predict_comps passes ~all of
    # unified) the isin scan is wasted work, so we skip it.
    subj_series = pairs["subject_property_id"]
    comp_series = pairs["comp_property_id"]
    n_pairs = len(pairs)
    if n_pairs * 3 < len(unified):
        needed = pd.unique(pd.concat([subj_series, comp_series], ignore_index=True))
        uni = unified.loc[unified["property_id"].isin(needed), keep_cols]
    else:
        uni = unified[keep_cols]
    uni = uni.drop_duplicates(subset=["property_id"], keep="first")

    # Merge both sides of `pairs` against uni. String normalization happens on
    # the merged frame (size <= n_pairs) rather than on the full unified table.
    s_side = uni.add_prefix("s_").rename(columns={"s_property_id": "subject_property_id"})
    c_side = uni.add_prefix("c_").rename(columns={"c_property_id": "comp_property_id"})

    df = pairs.merge(s_side, on="subject_property_id", how="left")
    df = df.merge(c_side, on="comp_property_id", how="left")

    # Normalize the case-folded compared strings on the merged frame.
    # `astype(str)` on NaN yields "nan"; collapse that + "" + "NONE" back to ""
    # so the downstream truthiness check (`!= ""`) reads them as null.
    def _upper_col(frame: pd.DataFrame, raw_col: str, out_col: str) -> None:
        if raw_col in frame.columns:
            frame[out_col] = (
                frame[raw_col].astype(str).str.strip().str.upper()
                .replace({"NAN": "", "NONE": ""}, regex=False)
            )
        else:
            frame[out_col] = ""

    _upper_col(df, "s_bldg_class", "s__bldg_u")
    _upper_col(df, "c_bldg_class", "c__bldg_u")
    _upper_col(df, "s_submarket", "s__sub_u")
    _upper_col(df, "c_submarket", "c__sub_u")
    _upper_col(df, "s_property_subtype", "s__subtype_u")
    _upper_col(df, "c_property_subtype", "c__subtype_u")

    # Drop rows where subject/comp missing from unified or lat/lon missing.
    valid = (
        df["s_lat"].notna() & df["s_lon"].notna()
        & df["c_lat"].notna() & df["c_lon"].notna()
    )
    df = df.loc[valid].reset_index(drop=True)

    if df.empty:
        return pd.DataFrame()

    n = len(df)

    # --- Distance ----------------------------------------------------------
    R = 3958.8
    s_lat = np.radians(df["s_lat"].to_numpy(dtype=float))
    c_lat = np.radians(df["c_lat"].to_numpy(dtype=float))
    dphi = c_lat - s_lat
    dlam = np.radians(df["c_lon"].to_numpy(dtype=float) - df["s_lon"].to_numpy(dtype=float))
    a = np.sin(dphi / 2) ** 2 + np.cos(s_lat) * np.cos(c_lat) * np.sin(dlam / 2) ** 2
    dist_raw = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = np.round(dist_raw, 4)
    log_dist = np.log1p(dist_raw)

    # --- Size (num_units for Multifamily, else sqft) ------------------------
    def _pick_size_vec(main_col: pd.Series, units_col: pd.Series, sqft_col: pd.Series) -> np.ndarray:
        lower = main_col.fillna("").astype(str).str.lower()
        is_mf = lower.str.contains("multifamily|apartment", regex=True, na=False).to_numpy()
        units = units_col.to_numpy(dtype=float) if units_col is not None else np.full(n, np.nan)
        sqft = sqft_col.to_numpy(dtype=float) if sqft_col is not None else np.full(n, np.nan)
        use_units = is_mf & ~np.isnan(units) & (units > 0)
        sqft_ok = ~np.isnan(sqft) & (sqft > 0)
        return np.where(use_units, units, np.where(sqft_ok, sqft, np.nan))

    s_size = _pick_size_vec(df["s_main_type"], df.get("s_num_units"), df.get("s_sqft"))
    c_size = _pick_size_vec(df["c_main_type"], df.get("c_num_units"), df.get("c_sqft"))
    with np.errstate(invalid="ignore", divide="ignore"):
        size_ok = ~np.isnan(s_size) & ~np.isnan(c_size) & (s_size > 0)
        size_ratio = np.where(size_ok, c_size / np.where(s_size == 0, np.nan, s_size), np.nan)
        log_size_ratio = np.where(
            size_ok & (size_ratio > 0), np.log(np.where(size_ratio > 0, size_ratio, 1.0)), np.nan
        )
        avg_size = (s_size + c_size) / 2.0
        size_var = np.where(
            size_ok & (avg_size > 0),
            np.abs(s_size - c_size) / np.where(avg_size > 0, avg_size, 1.0) * 100.0,
            np.nan,
        )

    # --- Signed/absolute diffs (each "both sides present or NaN") -----------
    def _signed_diff(s_col: str, c_col: str) -> tuple[np.ndarray, np.ndarray]:
        if s_col not in df.columns or c_col not in df.columns:
            return np.full(n, np.nan), np.full(n, np.nan)
        sv = df[s_col].to_numpy(dtype=float)
        cv = df[c_col].to_numpy(dtype=float)
        ok = ~np.isnan(sv) & ~np.isnan(cv)
        diff = np.where(ok, cv - sv, np.nan)
        return diff, np.where(ok, np.abs(diff), np.nan)

    yb_diff, abs_yb_diff = _signed_diff("s_year_built", "c_year_built")
    yr_diff, abs_yr_diff = _signed_diff("s_year_renovated", "c_year_renovated")
    stories_diff, abs_stories_diff = _signed_diff("s_num_stories", "c_num_stories")
    occ_diff, abs_occ_diff = _signed_diff("s_occupancy", "c_occupancy")
    park_diff, abs_park_diff = _signed_diff("s_parking_ratio", "c_parking_ratio")

    # --- Sale recency (years since comp's last sale) -----------------------
    c_sale = pd.to_datetime(df.get("c_last_sale_date"), errors="coerce")
    as_of_ts = pd.Timestamp(as_of_date)
    recency = (as_of_ts - c_sale).dt.days.to_numpy(dtype=float) / 365.25

    # --- Building class ----------------------------------------------------
    s_bldg = df["s__bldg_u"].to_numpy(dtype=object) if "s__bldg_u" in df else np.full(n, "", dtype=object)
    c_bldg = df["c__bldg_u"].to_numpy(dtype=object) if "c__bldg_u" in df else np.full(n, "", dtype=object)
    bldg_valid = (s_bldg != "") & (c_bldg != "")
    same_cls = np.where(bldg_valid, (s_bldg == c_bldg).astype(float), np.nan)

    s_ord = pd.Series(s_bldg).map(_CLASS_ORD).to_numpy(dtype=float)
    c_ord = pd.Series(c_bldg).map(_CLASS_ORD).to_numpy(dtype=float)
    ord_ok = bldg_valid & ~np.isnan(s_ord) & ~np.isnan(c_ord)
    cls_ord_diff = np.where(ord_ok, np.abs(c_ord - s_ord), np.nan)

    # --- Submarket ---------------------------------------------------------
    s_sub = df["s__sub_u"].to_numpy(dtype=object) if "s__sub_u" in df else np.full(n, "", dtype=object)
    c_sub = df["c__sub_u"].to_numpy(dtype=object) if "c__sub_u" in df else np.full(n, "", dtype=object)
    sub_valid = (s_sub != "") & (c_sub != "")
    same_sub = np.where(sub_valid, (s_sub == c_sub).astype(float), np.nan)

    # --- Land area ratio (clipped to [1e-3, 1e3]) --------------------------
    s_land = df["s_land_area_ac"].to_numpy(dtype=float) if "s_land_area_ac" in df else np.full(n, np.nan)
    c_land = df["c_land_area_ac"].to_numpy(dtype=float) if "c_land_area_ac" in df else np.full(n, np.nan)
    land_ok = ~np.isnan(s_land) & ~np.isnan(c_land) & (s_land > 0) & (c_land > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_ratio = np.where(land_ok, c_land / np.where(s_land == 0, np.nan, s_land), np.nan)
        clipped = np.clip(raw_ratio, 1e-3, 1e3)
        land_ratio = np.where(land_ok, clipped, np.nan)
        log_land_ratio = np.where(land_ok, np.log(np.where(land_ok, clipped, 1.0)), np.nan)

    # --- Property subtype (gated on same main_type + both non-null) --------
    s_main = df["s_main_type"].to_numpy(dtype=object)
    c_main = df["c_main_type"].to_numpy(dtype=object)
    s_subt_u = df["s__subtype_u"].to_numpy(dtype=object) if "s__subtype_u" in df else np.full(n, "", dtype=object)
    c_subt_u = df["c__subtype_u"].to_numpy(dtype=object) if "c__subtype_u" in df else np.full(n, "", dtype=object)
    subtype_gate = (
        (s_subt_u != "") & (c_subt_u != "")
        & pd.Series(s_main).notna().to_numpy() & pd.Series(c_main).notna().to_numpy()
        & (s_main == c_main)
    )
    same_subtype = np.where(subtype_gate, (s_subt_u == c_subt_u).astype(float), np.nan)

    # --- Type-awareness (same_main_type, ordinal encodings) ----------------
    s_main_valid = pd.Series(s_main).notna() & (pd.Series(s_main).astype(str) != "")
    c_main_valid = pd.Series(c_main).notna() & (pd.Series(c_main).astype(str) != "")
    type_valid = (s_main_valid & c_main_valid).to_numpy()
    same_main = np.where(type_valid, (s_main == c_main).astype(float), np.nan)

    s_type_ord = pd.Series(s_main).map(_MAIN_TYPE_ORD).fillna(-1).astype(int).to_numpy()
    c_type_ord = pd.Series(c_main).map(_MAIN_TYPE_ORD).fillna(-1).astype(int).to_numpy()

    # --- type_distance (tuple-keyed curated similarity) --------------------
    # Enumerate unique 4-tuples of (s_main, s_subtype, c_main, c_subtype) and
    # resolve type_distance() once per tuple. Typical inference case: one unique
    # subject across many candidates -> only a handful of unique tuples.
    def _sub_as_str(series: pd.Series | None) -> np.ndarray:
        if series is None:
            return np.full(n, "", dtype=object)
        return series.fillna("").astype(str).replace({"nan": "", "None": ""}, regex=False).to_numpy(dtype=object)

    s_sub_raw = _sub_as_str(df.get("s_property_subtype"))
    c_sub_raw = _sub_as_str(df.get("c_property_subtype"))

    td_index = pd.MultiIndex.from_arrays(
        [s_main, s_sub_raw, c_main, c_sub_raw],
        names=("sm", "ss", "cm", "cs"),
    )
    unique_keys = td_index.unique()
    td_lookup: dict[tuple, float] = {}
    for key in unique_keys:
        sm, ss, cm, cs = key
        try:
            td_val = type_distance(sm, ss, cm, cs)
        except Exception:
            td_val = None
        td_lookup[tuple(key)] = np.nan if td_val is None else float(td_val)
    td_feature = np.fromiter(
        (td_lookup[tuple(k)] for k in td_index),
        dtype=float,
        count=n,
    )

    # --- Assemble output ----------------------------------------------------
    out = pd.DataFrame({
        "subject_property_id": df["subject_property_id"].to_numpy(),
        "comp_property_id": df["comp_property_id"].to_numpy(),
        "distance_miles": dist,
        "log_distance": log_dist,
        "size_ratio": size_ratio,
        "log_size_ratio": log_size_ratio,
        "size_variance_pct": size_var,
        "year_built_diff": yb_diff,
        "abs_year_built_diff": abs_yb_diff,
        "year_renovated_diff": yr_diff,
        "abs_year_renovated_diff": abs_yr_diff,
        "sale_recency_years": recency,
        "same_bldg_class": same_cls,
        "same_submarket": same_sub,
        "bldg_class_ord_diff": cls_ord_diff,
        "stories_diff": stories_diff,
        "abs_stories_diff": abs_stories_diff,
        "occupancy_diff": occ_diff,
        "abs_occupancy_diff": abs_occ_diff,
        "land_area_ratio": land_ratio,
        "log_land_area_ratio": log_land_ratio,
        "same_subtype": same_subtype,
        "parking_ratio_diff": park_diff,
        "abs_parking_ratio_diff": abs_park_diff,
        "same_main_type": same_main,
        "subj_type_ord": s_type_ord,
        "comp_type_ord": c_type_ord,
        "type_distance": td_feature,
        "subject_main_type": df["s_main_type"].to_numpy(),
        "subject_metro": df["s_metro"].to_numpy() if "s_metro" in df else np.full(n, None, dtype=object),
    })

    # Carry label/match-score columns through when present.
    for col in ("comp_rank", "subject_match_score", "comp_match_score"):
        if col in pairs.columns:
            out[col] = df[col].to_numpy()

    # Assign integer query IDs (one per unique subject) in first-seen order,
    # matching the group split used downstream by XGBoost.
    qid_map = {sid: i for i, sid in enumerate(out["subject_property_id"].unique())}
    out["qid"] = out["subject_property_id"].map(qid_map)

    return out


def add_negatives(
    pairs: pd.DataFrame,
    unified: pd.DataFrame,
    neg_per_subject: int = 20,
    seed: int = 42,
    radius_miles: float | None = None,
    same_type_only: bool = True,
    max_type_distance: float | None = None,
    same_metro: bool = False,
    same_submarket: bool = False,
) -> pd.DataFrame:
    """Augment labeled pairs with negative (non-comp) examples for each subject.

    For each subject, samples random properties that were NOT in the subject's
    labeled comp set. These get ``comp_rank = 0`` (irrelevant).

    Parameters
    ----------
    pairs : DataFrame
        Labeled pairs with subject_property_id, comp_property_id, comp_rank.
    unified : DataFrame
        The unified property table.
    neg_per_subject : int
        Number of negative samples per subject.
    seed : int
        Random seed for reproducibility.
    radius_miles : float, optional
        If set, restrict the negative pool to properties within this haversine
        distance of the subject. Use this to align training negatives with the
        inference-time candidate pool (``predict_comps`` applies the same radius
        filter). Default ``None`` samples from the entire table.
    same_type_only : bool
        When True (default), restricts negatives to the subject's own
        ``main_type``. When False, draws negatives from all property types so
        the model can learn which cross-type pairings experts accept (requires
        ``FEATURE_COLS_EXT`` during training).
    max_type_distance : float, optional
        When set (and ``same_type_only=False``), restricts negatives to comps
        whose ``(main_type, property_subtype)`` tuple is within this distance of
        the subject's tuple per ``data/type_similarity.csv``. ``None`` (default)
        applies no type-similarity filter. Ignored when ``same_type_only=True``.
    same_metro, same_submarket : bool
        When True, restrict the negative pool to candidates sharing the
        subject's metro / submarket (a null subject- or candidate-value never
        matches). They compose (AND) and stack with the type / radius filters.

    Returns
    -------
    DataFrame with original pairs + negative pairs (comp_rank = 0).
    """
    import math

    rng = np.random.default_rng(seed)
    neg_rows: list[dict] = []

    valid = unified.loc[unified["lat"].notna() & unified["lon"].notna()].reset_index(drop=True)
    id_to_pos = pd.Series(np.arange(len(valid)), index=valid["property_id"].values)
    valid_ids = valid["property_id"].values
    valid_types = valid["main_type"].values
    valid_subtypes = (
        valid["property_subtype"].fillna("").astype(str).values
        if "property_subtype" in valid.columns
        else np.array([""] * len(valid), dtype=object)
    )
    valid_lat = valid["lat"].to_numpy(dtype=float)
    valid_lon = valid["lon"].to_numpy(dtype=float)
    valid_metro = (
        valid["metro"].values if "metro" in valid.columns
        else np.array([None] * len(valid), dtype=object)
    )
    valid_submarket = (
        valid["submarket"].values if "submarket" in valid.columns
        else np.array([None] * len(valid), dtype=object)
    )

    # Pre-compute per-main_type position arrays for the same_type_only branch.
    type_positions: dict[str, np.ndarray] = {}
    if same_type_only:
        for t, positions in pd.Series(np.arange(len(valid))).groupby(valid_types).indices.items():
            type_positions[str(t)] = np.asarray(positions, dtype=np.int64)

    # Pre-compute per-(main_type, subtype) position arrays for max_type_distance.
    tuple_positions: dict[tuple[str, str], np.ndarray] = {}
    if max_type_distance is not None:
        grouped = pd.Series(np.arange(len(valid))).groupby(
            [valid_types, valid_subtypes]
        ).indices
        for (t_main, t_sub), positions in grouped.items():
            tuple_positions[(str(t_main), str(t_sub))] = np.asarray(
                positions, dtype=np.int64
            )

    for sid, group in pairs.groupby("subject_property_id"):
        if sid not in id_to_pos.index:
            continue
        subj_pos = int(id_to_pos[sid])
        subj_type = valid_types[subj_pos]
        subj_sub = str(valid_subtypes[subj_pos])
        subj_lat = float(valid_lat[subj_pos])
        subj_lon = float(valid_lon[subj_pos])

        labeled_comps = set(group["comp_property_id"])
        labeled_positions = {
            int(id_to_pos[c]) for c in labeled_comps if c in id_to_pos.index
        }
        exclude = labeled_positions | {subj_pos}

        # Build candidate position array
        if same_type_only:
            pool_positions = type_positions.get(str(subj_type), np.empty(0, dtype=np.int64))
        elif max_type_distance is not None:
            allowed_tuple_keys = [
                k for k in tuple_positions
                if (type_distance(subj_type, subj_sub, k[0], k[1]) is not None)
                and (type_distance(subj_type, subj_sub, k[0], k[1]) <= max_type_distance)
            ]
            if not allowed_tuple_keys:
                continue
            pool_positions = np.concatenate([tuple_positions[k] for k in allowed_tuple_keys])
        else:
            pool_positions = np.arange(len(valid), dtype=np.int64)

        if pool_positions.size == 0:
            continue

        # Drop subject + labeled comps from the candidate pool
        if exclude:
            keep = ~np.isin(pool_positions, np.fromiter(exclude, dtype=np.int64, count=len(exclude)))
            pool_positions = pool_positions[keep]
            if pool_positions.size == 0:
                continue

        # Optional radius filter -- vectorized haversine on the candidate slice
        if radius_miles is not None:
            lat1 = math.radians(subj_lat)
            lon1 = math.radians(subj_lon)
            lat2 = np.radians(valid_lat[pool_positions])
            lon2 = np.radians(valid_lon[pool_positions])
            dphi = lat2 - lat1
            dlam = lon2 - lon1
            a = np.sin(dphi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlam / 2) ** 2
            dist_miles = 3958.8 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            pool_positions = pool_positions[dist_miles <= radius_miles]
            if pool_positions.size == 0:
                continue

        # Optional location-membership filter (same metro / same submarket).
        if same_metro or same_submarket:
            keep_loc = np.ones(pool_positions.size, dtype=bool)
            for use, arr in ((same_metro, valid_metro), (same_submarket, valid_submarket)):
                if not use:
                    continue
                subj_val = arr[subj_pos]
                if pd.isna(subj_val):
                    keep_loc[:] = False
                    break
                keep_loc &= (arr[pool_positions] == subj_val)
            pool_positions = pool_positions[keep_loc]
            if pool_positions.size == 0:
                continue

        n_sample = min(neg_per_subject, pool_positions.size)
        sampled_positions = rng.choice(pool_positions, size=n_sample, replace=False)

        for p in sampled_positions:
            neg_rows.append({
                "subject_property_id": sid,
                "comp_property_id": valid_ids[int(p)],
                "comp_rank": 0,
                "subject_match_score": 100.0,
                "comp_match_score": 100.0,
            })

    neg_df = pd.DataFrame(neg_rows)
    return pd.concat([pairs, neg_df], ignore_index=True)
