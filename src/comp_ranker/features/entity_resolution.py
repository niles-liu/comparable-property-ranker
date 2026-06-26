"""Cross-source property entity resolution.

The core problem: the same physical building can appear in Source A, Source B,
and Source C (see :mod:`comp_ranker.data.schema`) under different names,
abbreviated addresses, and slightly different coordinates. This module resolves
those records into clusters (one cluster = one real-world property) using a
three-tier strategy:

  Tier 1 -- Spatial  (highest confidence, no text required)
      Two records whose coordinates are within ``SPATIAL_RADIUS_M`` metres are
      almost certainly the same building. Uses a BallTree so the comparison
      scales to millions of rows without O(n^2) cost.

  Tier 2 -- Fuzzy address + name  (text fallback)
      Applied to pairs that are within ``FUZZY_CANDIDATE_RADIUS_MILES`` but did
      not pass the tight spatial threshold (e.g. a geocoding error of ~100 m).
      Both the normalised address AND the property name must exceed their
      respective similarity thresholds before a match is declared.

  Tier 3 -- Parcel number  (exact)
      Where an exact parcel/APN identifier can be joined across sources, those
      matches are treated as definitive links. Pass ``apn_pairs`` to
      :func:`match_and_merge` when available.

Usage
-----
    from comp_ranker.features.entity_resolution import match_and_merge, lookup_property

    # source_a, source_b, source_c are DataFrames in the unified schema
    # (see comp_ranker.data.schema). In this repo they come from the synthetic
    # generator; in production they come from the proprietary connector layer
    # (not shipped).
    unified = match_and_merge(source_a, source_b, source_c)

    # Find a property by name or address at query time
    hits = lookup_property("One Penn Plaza", unified, top_k=5)
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.neighbors import BallTree

# ---------------------------------------------------------------------------
# Matching thresholds (tune here)
# ---------------------------------------------------------------------------
SPATIAL_RADIUS_M: float = 50.0          # metres  -- definitive same-building threshold
FUZZY_CANDIDATE_RADIUS_MILES: float = 0.25  # search window for fuzzy candidates
ADDR_SIMILARITY_THRESHOLD: int = 85     # rapidfuzz token_sort_ratio 0-100
NAME_SIMILARITY_THRESHOLD: int = 70     # lower because names vary more

_EARTH_RADIUS_M = 6_371_000.0
_EARTH_RADIUS_MILES = 3_958.8

# ---------------------------------------------------------------------------
# Address normalisation
# ---------------------------------------------------------------------------
# Collapse long forms -> short abbreviations (safer than expanding, avoids
# "St Louis" -> "Street Louis" and "Dr King" -> "Drive King" class of errors).
_ABBREV: dict[str, str] = {
    r"\bstreet\b":      "st",
    r"\bavenue\b":      "ave",
    r"\bboulevard\b":   "blvd",
    r"\bdrive\b":       "dr",
    r"\broad\b":        "rd",
    r"\blane\b":        "ln",
    r"\bcourt\b":       "ct",
    r"\bplace\b":       "pl",
    r"\bplaza\b":       "plz",
    r"\bsquare\b":      "sq",
    r"\bparkway\b":     "pkwy",
    r"\bhighway\b":     "hwy",
    r"\bfreeway\b":     "fwy",
    r"\bexpressway\b":  "expy",
    r"\bsuite\b":       "ste",
}

# Directional normalisation -- collapse to abbreviations
_DIRECTIONALS: dict[str, str] = {
    r"\bnorth\b": "n", r"\bsouth\b": "s", r"\beast\b": "e", r"\bwest\b": "w",
    r"\bnortheast\b": "ne", r"\bnorthwest\b": "nw",
    r"\bsoutheast\b": "se", r"\bsouthwest\b": "sw",
}

# Only replace written numbers at the START of the string ("One Penn Plaza"
# but not "Keystone One Partners")
_WRITTEN_NUMS: dict[str, str] = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

# Strip unit/suite designators only when followed by a digit, so
# "Ste 400" and "#2b" are removed but "Floor Plan Associates" is kept.
_UNIT_PATTERN = re.compile(
    r"\b(suite|ste|unit|apt|apartment|floor|fl|#)\s*\d[\w-]*", re.IGNORECASE
)

# Strip ordinal suffixes: "3rd" -> "3", "42nd" -> "42"
_ORDINAL_PATTERN = re.compile(r"(\d+)(st|nd|rd|th)\b", re.IGNORECASE)


def normalize_address(s: Any) -> str:
    """Normalise an address or property name string for fuzzy comparison.

    Steps
    -----
    1. Unicode -> ASCII (strips accents)
    2. Fold to lower case
    3. Replace written numbers with digits at start of string (one -> 1)
    4. Collapse long-form street types to abbreviations (street -> st)
    5. Collapse directionals (north -> n, southwest -> sw)
    6. Strip ordinal suffixes (3rd -> 3, 42nd -> 42)
    7. Strip unit/suite designators only when followed by a digit
    8. Remove punctuation except hyphens and spaces
    9. Collapse whitespace
    """
    if not s or pd.isna(s):
        return ""
    text = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    text = text.lower().strip()

    # Written numbers -- only at start of string
    for word, digit in _WRITTEN_NUMS.items():
        text = re.sub(rf"^{word}\b", digit, text)

    # Collapse long forms -> abbreviations
    for pattern, replacement in _ABBREV.items():
        text = re.sub(pattern, replacement, text)

    # Directionals
    for pattern, replacement in _DIRECTIONALS.items():
        text = re.sub(pattern, replacement, text)

    # Ordinal suffixes
    text = _ORDINAL_PATTERN.sub(r"\1", text)

    # Unit/suite designators (only when followed by a digit)
    text = _UNIT_PATTERN.sub("", text)

    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_STATE_PATTERN = re.compile(
    r"\b([A-Z]{2})\b\s*\d{5}",
)


def _extract_state(addr: Any) -> str:
    """Extract a two-letter state code from an address string, or empty string."""
    if not addr or pd.isna(addr):
        return ""
    m = _STATE_PATTERN.search(str(addr))
    return m.group(1) if m else ""


def _ensure_norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add _norm_addr, _norm_name, and _state columns if not already present."""
    if "_norm_addr" not in df.columns:
        df = df.copy()
        df["_norm_addr"] = df["full_address"].apply(normalize_address)
        df["_norm_name"] = df["property_name"].apply(normalize_address)
    if "_state" not in df.columns:
        if "full_address" in df.columns:
            df["_state"] = df["full_address"].apply(_extract_state)
        else:
            df["_state"] = ""
    return df


# ---------------------------------------------------------------------------
# Union-Find (for clustering matched pairs)
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, ids: list[str]) -> None:
        self._parent = {i: i for i in ids}
        self._rank: dict[str, int] = defaultdict(int)

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def connected(self, a: str, b: str) -> bool:
        """Return True if a and b are already in the same cluster."""
        return self.find(a) == self.find(b)

    def clusters(self) -> dict[str, str]:
        """Return {property_id: cluster_root_id}."""
        return {i: self.find(i) for i in self._parent}


# ---------------------------------------------------------------------------
# Tier 1: spatial matching via BallTree
# ---------------------------------------------------------------------------

def _to_radians(df: pd.DataFrame) -> np.ndarray:
    """Return (n, 2) array of [lat_rad, lon_rad]."""
    return np.radians(df[["lat", "lon"]].values.astype(float))


def find_spatial_matches(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    radius_m: float = SPATIAL_RADIUS_M,
) -> list[tuple[str, str]]:
    """Return (id_a, id_b) pairs whose coordinates are within radius_m metres.

    Builds a BallTree on df_a, then queries each row of df_b. Both DataFrames
    must have 'property_id', 'lat', 'lon' columns and no NaN coordinates.
    """
    a_clean = df_a.dropna(subset=["lat", "lon"])
    b_clean = df_b.dropna(subset=["lat", "lon"])
    if a_clean.empty or b_clean.empty:
        return []

    radius_rad = radius_m / _EARTH_RADIUS_M
    tree = BallTree(_to_radians(a_clean), metric="haversine")
    indices = tree.query_radius(_to_radians(b_clean), r=radius_rad)

    ids_a = a_clean["property_id"].values  # numpy array -- faster indexing
    ids_b = b_clean["property_id"].values

    pairs: list[tuple[str, str]] = []
    for b_idx, neighbours in enumerate(indices):
        b_id = ids_b[b_idx]
        for a_idx in neighbours:
            pairs.append((ids_a[a_idx], b_id))
    return pairs


# ---------------------------------------------------------------------------
# Tier 2: fuzzy address + name matching
# ---------------------------------------------------------------------------

def find_fuzzy_matches(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    candidate_radius_miles: float = FUZZY_CANDIDATE_RADIUS_MILES,
    addr_threshold: int = ADDR_SIMILARITY_THRESHOLD,
    name_threshold: int = NAME_SIMILARITY_THRESHOLD,
    already_matched: set[tuple[str, str]] | None = None,
) -> list[tuple[str, str, float]]:
    """Return (id_a, id_b, confidence) triples for properties that:
      - are within candidate_radius_miles of each other, AND
      - have normalised address similarity >= addr_threshold, AND
      - have normalised name similarity >= name_threshold.

    confidence is the average of the two similarity scores (0-100).

    Parameters
    ----------
    already_matched : set of (id_a, id_b) or None
        Pairs already linked by Tier 1 spatial matching. These are skipped to
        avoid redundant string comparisons.
    """
    a_clean = df_a.dropna(subset=["lat", "lon"])
    b_clean = df_b.dropna(subset=["lat", "lon"])
    if a_clean.empty or b_clean.empty:
        return []

    # Expect _norm_addr/_norm_name to already exist (pre-normalized)
    a_clean = _ensure_norm_columns(a_clean)
    b_clean = _ensure_norm_columns(b_clean)

    radius_rad = (candidate_radius_miles / _EARTH_RADIUS_MILES)
    tree = BallTree(_to_radians(a_clean), metric="haversine")
    indices = tree.query_radius(_to_radians(b_clean), r=radius_rad)

    a_ids    = a_clean["property_id"].values
    a_addrs  = a_clean["_norm_addr"].values
    a_names  = a_clean["_norm_name"].values
    b_ids    = b_clean["property_id"].values
    b_addrs  = b_clean["_norm_addr"].values
    b_names  = b_clean["_norm_name"].values

    _matched = already_matched or set()

    results: list[tuple[str, str, float]] = []
    for b_idx, neighbours in enumerate(indices):
        if not len(neighbours):
            continue
        b_id   = b_ids[b_idx]
        b_addr = b_addrs[b_idx]
        b_name = b_names[b_idx]

        for a_idx in neighbours:
            a_id = a_ids[a_idx]

            # Skip pairs already linked by Tier 1
            if (a_id, b_id) in _matched:
                continue

            # score_cutoff enables early termination inside rapidfuzz's C code:
            # if the score can't possibly reach the threshold, it returns 0
            # immediately without completing the full comparison.
            addr_score = fuzz.token_sort_ratio(
                a_addrs[a_idx], b_addr, score_cutoff=addr_threshold
            )
            if addr_score == 0:
                continue
            name_score = fuzz.token_sort_ratio(
                a_names[a_idx], b_name, score_cutoff=name_threshold
            )
            if name_score == 0:
                continue
            confidence = (addr_score + name_score) / 2.0
            results.append((a_id, b_id, confidence))

    return results


# ---------------------------------------------------------------------------
# Parallel helpers for pairwise matching
# ---------------------------------------------------------------------------

def _spatial_match_pair(args: tuple) -> list[tuple[str, str]]:
    """Wrapper for find_spatial_matches (used by ThreadPoolExecutor)."""
    df_a, df_b, radius_m = args
    return find_spatial_matches(df_a, df_b, radius_m)


def _fuzzy_match_pair(args: tuple) -> list[tuple[str, str, float]]:
    """Wrapper for find_fuzzy_matches (used by ThreadPoolExecutor)."""
    df_a, df_b, already_matched = args
    return find_fuzzy_matches(df_a, df_b, already_matched=already_matched)


# ---------------------------------------------------------------------------
# Feature merging
# ---------------------------------------------------------------------------

# Source priority for feature merging: Source A data is the most complete,
# then Source B, then Source C.
_SOURCE_PRIORITY = ["source_a", "source_b", "source_c"]

# Fields where we want the MOST RECENT value rather than highest-priority source
_TAKE_MOST_RECENT = {"last_sale_date"}

# Fields we never overwrite with a lower-priority source's value
_SCALAR_FIELDS = [
    "property_name", "full_address", "lat", "lon",
    "sqft", "num_units", "year_built", "year_renovated",
    "bldg_class", "main_type", "submarket", "metro",
    "num_stories", "occupancy", "land_area_ac",
    "property_subtype", "parking_ratio",
    "avg_unit_size",
]


def merge_cluster_features(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple feature dicts for the same physical property into one.

    Rules
    -----
    - Scalar fields: take the first non-None value in source-priority order
      (source_a -> source_b -> source_c).
    - last_sale_date: take the most recent non-None value across all records.
    - property_id: store all source IDs in a list under 'source_ids'.
    """
    ordered = sorted(
        records,
        key=lambda r: _SOURCE_PRIORITY.index(r.get("source", "source_c"))
        if r.get("source") in _SOURCE_PRIORITY else len(_SOURCE_PRIORITY),
    )

    merged: dict[str, Any] = {f: None for f in _SCALAR_FIELDS}
    merged["source_ids"] = [r["property_id"] for r in records]
    merged["source"] = sorted(set(r.get("source", "unknown") for r in records))

    for field in _SCALAR_FIELDS:
        for rec in ordered:
            val = rec.get(field)
            if val is not None and val != "":
                merged[field] = val
                break

    # Most-recent for date fields
    for field in _TAKE_MOST_RECENT:
        dates = [r.get(field) for r in records if r.get(field) is not None and pd.notna(r.get(field))]
        merged[field] = max(dates) if dates else None

    # Use the Source A ID as the canonical ID when available, else first in list
    source_a_ids = [r["property_id"] for r in records if r.get("source") == "source_a"]
    merged["property_id"] = source_a_ids[0] if source_a_ids else records[0]["property_id"]

    return merged


# ---------------------------------------------------------------------------
# Single-property lookup (query-time use)
# ---------------------------------------------------------------------------

def lookup_property(
    query: str,
    unified_df: pd.DataFrame,
    top_k: int = 5,
    name_col: str = "property_name",
    addr_col: str = "full_address",
    state: str | None = None,
) -> pd.DataFrame:
    """Given a free-text name or address, return the top_k most likely matches
    from the unified property DataFrame.

    Scoring: max(name_similarity, address_similarity) using token_sort_ratio.
    This is intended for interactive lookup or resolving a labeled comp-set CSV.

    Parameters
    ----------
    query : str
        The name or address string to search for.
    unified_df : pd.DataFrame
        The merged property table produced by match_and_merge().
    top_k : int
        Number of results to return.
    name_col, addr_col : str
        Column names in unified_df for property name and address.
    state : str, optional
        Two-letter state code (e.g. "NY") to pre-filter the unified table
        before fuzzy matching. Dramatically reduces search space when known.

    Returns
    -------
    pd.DataFrame with columns: property_id, property_name, full_address,
    main_type, metro, source, match_score -- sorted by match_score desc.
    """
    norm_q = normalize_address(query)
    if not norm_q:
        return pd.DataFrame()

    # Pre-compute normalised columns if not already present
    unified_df = _ensure_norm_columns(unified_df)

    # Pre-filter by state using the pre-built _state column (O(1) equality
    # check instead of per-call regex over the full address string).
    search_df = unified_df
    if state:
        state_upper = state.strip().upper()
        mask = unified_df["_state"] == state_upper
        if mask.sum() > 0:
            search_df = unified_df[mask]

    # Batch fuzzy search using rapidfuzz's C++ engine (much faster than apply)
    name_list = search_df["_norm_name"].fillna("").tolist()
    addr_list = search_df["_norm_addr"].fillna("").tolist()

    name_hits = process.extract(
        norm_q, name_list, scorer=fuzz.token_sort_ratio, limit=top_k,
    )
    addr_hits = process.extract(
        norm_q, addr_list, scorer=fuzz.token_sort_ratio, limit=top_k,
    )

    # Merge the two hit lists, keeping the best score per positional index
    best: dict[int, float] = {}
    for _text, score, idx in name_hits:
        best[idx] = score
    for _text, score, idx in addr_hits:
        if score > best.get(idx, -1):
            best[idx] = score

    # Sort by score descending and take top_k
    ranked = sorted(best.items(), key=lambda x: -x[1])[:top_k]
    pos_indices = [i for i, _ in ranked]
    score_vals = [s for _, s in ranked]

    out = search_df.iloc[pos_indices][[
        "property_id", name_col, addr_col, "main_type", "metro", "source",
    ]].copy()
    out["match_score"] = score_vals
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def match_and_merge(
    source_a_df: pd.DataFrame,
    source_b_df: pd.DataFrame,
    source_c_df: pd.DataFrame | None = None,
    apn_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Match records across all three sources and return a unified property
    DataFrame with one row per real-world property.

    Parameters
    ----------
    source_a_df, source_b_df, source_c_df : pd.DataFrame
        Records from each source in the unified schema
        (see :mod:`comp_ranker.data.schema`).
    apn_pairs : list of (source_a_id, other_id), optional
        Definitive same-property pairs from exact parcel-number matching
        (Tier 3). Pass these when available to supplement spatial/fuzzy matching.

    Returns
    -------
    pd.DataFrame where each row is a merged property feature dict. Includes a
    'source_ids' column listing all contributing source IDs.
    """
    # ---- Pre-normalize text columns ONCE across all frames ----------------
    print("[match] Pre-normalizing text columns ...")
    source_a_df = _ensure_norm_columns(source_a_df)
    source_b_df = _ensure_norm_columns(source_b_df)
    if source_c_df is not None and not source_c_df.empty:
        source_c_df = _ensure_norm_columns(source_c_df)

    frames = [source_a_df, source_b_df]
    if source_c_df is not None and not source_c_df.empty:
        frames.append(source_c_df)
    all_props = pd.concat(frames, ignore_index=True)

    all_ids = all_props["property_id"].tolist()
    uf = _UnionFind(all_ids)

    # ---- Tier 3: parcel-number pairs (exact, highest confidence) ----------
    for id_a, id_b in (apn_pairs or []):
        if id_a in uf._parent and id_b in uf._parent:
            uf.union(id_a, id_b)

    # ---- Tier 1: spatial within SPATIAL_RADIUS_M metres -------------------
    db_pairs = [
        (source_a_df, source_b_df),
    ]
    if source_c_df is not None and not source_c_df.empty:
        db_pairs += [(source_a_df, source_c_df), (source_b_df, source_c_df)]

    print(f"[match] Tier 1: spatial matching ({len(db_pairs)} pair(s)) ...")
    spatial_args = [(a, b, SPATIAL_RADIUS_M) for a, b in db_pairs]
    with ThreadPoolExecutor(max_workers=len(db_pairs)) as pool:
        spatial_results = list(pool.map(_spatial_match_pair, spatial_args))

    # Collect all Tier 1 pairs and union them; also build a set for Tier 2 skip
    tier1_matched: set[tuple[str, str]] = set()
    for pairs in spatial_results:
        for id_a, id_b in pairs:
            uf.union(id_a, id_b)
            tier1_matched.add((id_a, id_b))
    print(f"[match] Tier 1: {len(tier1_matched):,} spatial pairs found")

    # ---- Tier 2: fuzzy address + name (wider radius, text gating) ----------
    # Skip pairs already matched in Tier 1 to avoid redundant string compares.
    print(f"[match] Tier 2: fuzzy matching ({len(db_pairs)} pair(s)) ...")

    def _skip_set_for(df_a: pd.DataFrame, df_b: pd.DataFrame) -> set[tuple[str, str]]:
        a_ids = set(df_a["property_id"].values)
        b_ids = set(df_b["property_id"].values)
        return {(a, b) for a, b in tier1_matched if a in a_ids and b in b_ids}

    fuzzy_args = [
        (a, b, _skip_set_for(a, b))
        for a, b in db_pairs
    ]
    with ThreadPoolExecutor(max_workers=len(db_pairs)) as pool:
        fuzzy_results = list(pool.map(_fuzzy_match_pair, fuzzy_args))

    tier2_count = 0
    for triples in fuzzy_results:
        for id_a, id_b, _ in triples:
            uf.union(id_a, id_b)
            tier2_count += 1
    print(f"[match] Tier 2: {tier2_count:,} fuzzy pairs found")

    # ---- Build clusters & merge (vectorized) ------------------------------
    print("[match] Assigning cluster IDs ...")
    cluster_map = uf.clusters()  # {property_id: cluster_root}

    all_props["_cluster"] = all_props["property_id"].map(cluster_map)

    # Sort by source priority so source_a rows come first within each cluster.
    # pandas groupby.first() skips NaN, so the first non-null value per field
    # will come from the highest-priority source.
    _src_order = {s: i for i, s in enumerate(_SOURCE_PRIORITY)}
    all_props["_src_rank"] = all_props["source"].map(
        lambda s: _src_order.get(s, len(_SOURCE_PRIORITY))
    )
    all_props = all_props.sort_values("_src_rank")

    print("[match] Merging clusters ...")
    grouped = all_props.groupby("_cluster", sort=False)

    # Scalar fields: first non-null in priority order (already sorted)
    scalar_cols = [c for c in _SCALAR_FIELDS if c in all_props.columns]
    merged = grouped[scalar_cols].first()

    # last_sale_date: most recent across all records.
    if "last_sale_date" in all_props.columns:
        _sale_dates = pd.to_datetime(all_props["last_sale_date"], errors="coerce")
        merged["last_sale_date"] = (
            _sale_dates.groupby(all_props["_cluster"], sort=False)
            .max()
            .dt.date
            .values
        )

    # source_ids: list of all contributing property IDs
    merged["source_ids"] = grouped["property_id"].apply(list)

    # source: sorted unique source labels
    merged["source"] = grouped["source"].apply(lambda s: sorted(s.unique().tolist()))

    # property_id: prefer the Source A ID as canonical, else first in cluster
    def _pick_canonical_id(group: pd.Series) -> str:
        source_a = [pid for pid in group if str(pid).startswith("A_")]
        return source_a[0] if source_a else group.iloc[0]

    merged["property_id"] = grouped["property_id"].agg(_pick_canonical_id)

    merged = merged.reset_index(drop=True)

    # Drop internal columns that may have leaked through
    merged = merged.drop(columns=["_cluster", "_src_rank", "_norm_addr", "_norm_name"],
                         errors="ignore")

    print(f"[match] {len(all_ids):,} source rows -> {len(merged):,} unique properties")
    return merged
