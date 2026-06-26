"""Ingest expert comp-set labels into trainable subject-comp pairs.

Labels are **expert-selected comparables** drawn from historical underwriting /
valuation decisions: for a set of subject properties, the comps an analyst
actually chose (optionally ranked). They are provided as a free-text CSV --
names and addresses, not IDs -- so each entry must be resolved against the
unified corpus before training.

Label-relative caveat
---------------------
These labels reflect **expert selection**, not ground truth: they are the comps
an expert chose, which is not the same as the complete set of all valid comps.
Every metric computed against them is therefore *agreement-with-expert-selection*
("label-relative"), not absolute retrieval accuracy. Treat the numbers
accordingly.

Pipeline
--------
1. Read a comp-sets CSV (subject + its ranked comps, as free text).
2. Fuzzy-match each subject and comp against the unified corpus
   (:func:`comp_ranker.features.entity_resolution.lookup_property`).
3. Emit resolved ``(subject_property_id, comp_property_id, comp_rank, ...)``
   triplets plus a separate list of entries that could not be matched (which
   need manual review).

Usage
-----
    from comp_ranker.data.schema import load_corpus
    from comp_ranker.features.labels import ingest_comp_sets

    unified = load_corpus()
    pairs, unmatched = ingest_comp_sets(unified, "data/comp_sets.csv")
"""

from __future__ import annotations

import pandas as pd

from comp_ranker.features.entity_resolution import lookup_property

# Minimum fuzzy match score to auto-accept a lookup result.
MIN_MATCH_SCORE: int = 70

# Columns the comp-sets CSV must provide.
REQUIRED_COLUMNS = [
    "set_id",
    "subject_name", "subject_address", "subject_city", "subject_state",
    "comp_rank", "comp_name", "comp_address", "comp_city", "comp_state",
]

# Cache key: (name, address, state) -> (property_id | None, score)
_LookupCache = dict[tuple[str, str, str], tuple[str | None, float]]


def _cache_key(name: str | None, address: str | None, state: str | None) -> tuple[str, str, str]:
    return (
        (name or "").strip().lower(),
        (address or "").strip().lower(),
        (state or "").strip().upper(),
    )


def _lookup_one(
    name: str | None,
    address: str | None,
    state: str | None,
    unified: pd.DataFrame,
    min_score: int = MIN_MATCH_SCORE,
    cache: _LookupCache | None = None,
) -> tuple[str | None, float]:
    """Match a property by address (most specific) then name against the corpus.

    Passes the state code to ``lookup_property`` to pre-filter the search space.
    Results are memoised in *cache* so duplicate queries across comp sets resolve
    only once.
    """
    key = _cache_key(name, address, state)
    if cache is not None and key in cache:
        return cache[key]

    queries = [q.strip() for q in (address, name) if q and str(q).strip()]
    state_filter = state.strip() if state and str(state).strip() else None

    best_id, best_score = None, 0.0
    for q in queries:
        hits = lookup_property(q, unified, top_k=1, state=state_filter)
        if hits.empty:
            continue
        score = float(hits.iloc[0]["match_score"])
        if score > best_score:
            best_score = score
            best_id = hits.iloc[0]["property_id"]
        if best_score >= min_score:
            break

    result = (best_id, best_score) if best_score >= min_score else (None, best_score)
    if cache is not None:
        cache[key] = result
    return result


def ingest_comp_sets(
    unified: pd.DataFrame,
    comp_sets_csv: str,
    min_score: int = MIN_MATCH_SCORE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve a comp-sets CSV into labeled training pairs.

    Parameters
    ----------
    unified : pd.DataFrame
        The unified corpus (see :mod:`comp_ranker.data.schema`).
    comp_sets_csv : str
        Path to a CSV with the columns in :data:`REQUIRED_COLUMNS`.
    min_score : int
        Minimum fuzzy match score to accept a resolution.

    Returns
    -------
    (pairs_df, unmatched_df)
        ``pairs_df``     -- resolved (subject_property_id, comp_property_id,
                            comp_rank, subject_match_score, comp_match_score).
        ``unmatched_df`` -- entries that could not be resolved above threshold.
    """
    template = pd.read_csv(comp_sets_csv, dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in template.columns]
    if missing:
        raise ValueError(f"comp-sets CSV is missing columns: {missing}")

    cache: _LookupCache = {}
    pairs: list[dict] = []
    unmatched: list[dict] = []

    for set_id, group in template.groupby("set_id", sort=False):
        row0 = group.iloc[0]
        subj_id, subj_score = _lookup_one(
            row0["subject_name"], row0["subject_address"], row0["subject_state"],
            unified, min_score=min_score, cache=cache,
        )
        if subj_id is None:
            unmatched.append({
                "set_id": set_id, "role": "subject",
                "name": row0["subject_name"], "address": row0["subject_address"],
                "best_score": subj_score, "note": "No match above threshold",
            })
            continue

        for _, comp_row in group.iterrows():
            comp_id, comp_score = _lookup_one(
                comp_row["comp_name"], comp_row["comp_address"], comp_row["comp_state"],
                unified, min_score=min_score, cache=cache,
            )
            if comp_id is None:
                unmatched.append({
                    "set_id": set_id, "role": "comp",
                    "name": comp_row["comp_name"], "address": comp_row["comp_address"],
                    "best_score": comp_score, "note": "No match above threshold",
                })
                continue
            if comp_id == subj_id:
                continue  # subject matched to itself -- skip

            pairs.append({
                "set_id": set_id,
                "subject_property_id": subj_id,
                "comp_property_id": comp_id,
                "comp_rank": comp_row["comp_rank"],
                "subject_match_score": round(subj_score, 1),
                "comp_match_score": round(comp_score, 1),
            })

    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df["comp_rank"] = pd.to_numeric(pairs_df["comp_rank"], errors="coerce").fillna(0).astype(int)
    unmatched_df = pd.DataFrame(unmatched)
    return pairs_df, unmatched_df
