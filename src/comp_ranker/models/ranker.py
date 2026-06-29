"""Learning-to-rank (and pointwise-classifier) models for comparable selection.

Two model families share one pairwise feature representation and one evaluation
path, so they can be compared apples-to-apples:

* **LambdaMART ranker** (``XGBRanker``, ``rank:ndcg``) -- learns a listwise
  ordering within each subject's candidate group.
* **Binary classifier** (``XGBClassifier``, ``binary:logistic``) -- scores each
  (subject, comp) pair independently as comp / not-comp; the predicted
  probability is used as the ranking score.

The headline finding the demo reproduces in miniature is the **closed-set ->
full-pool reversal**: the ranker is competitive (often better) in closed-set
cross-validation, where every subject is scored against a small fixed candidate
group, but the per-row classifier wins at realistic *full-pool* retrieval, where
each subject is scored against the entire radius pool. ``cross_validate`` measures
the closed-set view; ``full_pool_diagnostic`` measures the full-pool view.

Usage
-----
    from comp_ranker.features.pairwise import (
        build_pairwise_features, add_negatives, FEATURE_COLS_PLUS_OCC,
    )
    from comp_ranker.models.ranker import (
        train_ranker, train_classifier, cross_validate,
        full_pool_diagnostic, predict_comps,
    )

    feats = build_pairwise_features(add_negatives(pairs, unified), unified)
    ranker, _ = train_ranker(feats, feature_cols=FEATURE_COLS_PLUS_OCC)
    clf, _    = train_classifier(feats, feature_cols=FEATURE_COLS_PLUS_OCC)
    comps = predict_comps(clf, subject_id, unified, n=10)
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from comp_ranker.features.pairwise import (
    FEATURE_COLS,
    FEATURE_COLS_PLUS_OCC,
    _FEATS_BY_WIDTH,
    build_pairwise_features,
)
from comp_ranker.scoring.config import MAX_RADIUS_MILES

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
_MODEL_DIR = os.path.join(_ROOT, "models")

# Default feature set when a caller does not specify one. The 18-feature
# production set (see comp_ranker.features.pairwise).
_DEFAULT_FEATURE_COLS = FEATURE_COLS_PLUS_OCC


def _feature_cols_for_model(model: Any) -> list[str]:
    """Pick the feature-column list matching a saved model's input width.

    Models are pinned to the feature width they were trained on. This helper
    reads ``model.n_features_in_`` and returns the registered column slice from
    ``_FEATS_BY_WIDTH``, so ``predict_comps`` keeps working without callers
    having to track which feature set a given model was trained with.
    """
    n = int(getattr(model, "n_features_in_", 0)) or int(model.get_booster().num_features())
    if n not in _FEATS_BY_WIDTH:
        raise ValueError(
            f"Model expects {n} features; register that width in "
            f"comp_ranker.features.pairwise::_FEATS_BY_WIDTH."
        )
    return _FEATS_BY_WIDTH[n]


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

def _ranker_params(seed: int | None = None) -> dict[str, Any]:
    params = {
        "objective": "rank:ndcg",
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "gamma": 1.0,
        "tree_method": "hist",
        "verbosity": 0,
    }
    if seed is not None:
        params["random_state"] = seed
    return params


def _classifier_params(seed: int | None = None, scale_pos_weight: float | None = None) -> dict[str, Any]:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "gamma": 1.0,
        "tree_method": "hist",
        "verbosity": 0,
    }
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    if seed is not None:
        params["random_state"] = seed
    return params


# ---------------------------------------------------------------------------
# Model metadata schema v1
# ---------------------------------------------------------------------------

MODEL_META_SCHEMA_VERSION = 1


def build_meta_v1(
    model: Any,
    *,
    feature_cols: list[str],
    metrics: dict[str, Any],
    xgb_params: dict[str, Any],
    model_name: str = "",
    model_class: str = "XGBRanker",
    n_positive_pairs: Optional[int] = None,
    n_negative_pairs: Optional[int] = None,
    n_subjects: Optional[int] = None,
    seed: Optional[int] = None,
    source_script: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Build a schema-v1 model metadata dict (the sidecar saved next to a model).

    Fields not known at call time can be left at their defaults. ``model_class``
    records which estimator the booster should be loaded into so ``load_model``
    can restore ``predict_proba`` for classifiers.
    """
    return {
        "schema_version": MODEL_META_SCHEMA_VERSION,
        "model_name": model_name,
        "model_class": model_class,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_width": len(feature_cols),
        "feature_cols": list(feature_cols),
        "n_positive_pairs": n_positive_pairs,
        "n_negative_pairs": n_negative_pairs,
        "n_subjects": n_subjects,
        "seed": seed,
        "xgb_params": dict(xgb_params),
        "metrics": dict(metrics),
        "source_script": source_script,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Match-score sample weighting
# ---------------------------------------------------------------------------

def compute_sample_weights(
    feature_df: pd.DataFrame,
    min_match_score: float = 70.0,
) -> np.ndarray:
    """Per-sample training weights from fuzzy entity-resolution confidence.

    Labeled pairs get weight ``min(subject_match_score, comp_match_score) / 100``,
    so a pair resolved with 85% confidence weighs 0.85 and a perfect match 1.0.
    Negatives (added by ``add_negatives``) carry match_score 100 since they are
    exact ``property_id`` references. Pairs below ``min_match_score`` get weight 0
    (XGBoost ignores zero-weight samples), filtering low-confidence labels out of
    training without dropping rows.
    """
    subj = feature_df["subject_match_score"].fillna(100.0).values
    comp = feature_df["comp_match_score"].fillna(100.0).values
    pair_score = np.minimum(subj, comp)
    return np.where(pair_score >= min_match_score, pair_score / 100.0, 0.0)


# ---------------------------------------------------------------------------
# Labels and group splitting
# ---------------------------------------------------------------------------

def _rank_to_relevance(comp_rank: pd.Series, max_rank: int = 10) -> np.ndarray:
    """Convert 1-based ``comp_rank`` to a graded relevance label for LambdaMART.

    Labeled comps get relevance ``max_rank + 1 - rank`` (rank 1 -> highest);
    negatives (rank 0) get relevance 0.
    """
    r = comp_rank.fillna(0).astype(int).values
    return np.where(r > 0, max_rank + 1 - np.clip(r, 1, max_rank), 0)


def _group_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by subject (qid) so no subject appears in both train and test."""
    rng = np.random.default_rng(seed)
    qids = df["qid"].unique()
    rng.shuffle(qids)
    n_test = max(1, int(len(qids) * test_frac))
    test_qids = set(qids[:n_test])
    train = df[~df["qid"].isin(test_qids)].copy()
    test = df[df["qid"].isin(test_qids)].copy()
    return train, test


def _get_group_sizes(df: pd.DataFrame) -> np.ndarray:
    """Group sizes ordered by qid first-appearance (XGBRanker ``group`` arg)."""
    return df.groupby("qid", sort=False).size().values


def _aggregate_weights_to_groups(df: pd.DataFrame, sample_weights: np.ndarray) -> np.ndarray:
    """Aggregate per-row weights to per-group means for ``XGBRanker``.

    XGBRanker requires ``sample_weight`` to have one entry per query group, not
    one per row; we take the mean weight within each group so groups built from
    low-confidence matches are down-weighted as intended.
    """
    return (
        pd.Series(sample_weights, index=df.index)
        .groupby(df["qid"], sort=False)
        .mean()
        .values
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ranker(
    feature_df: pd.DataFrame,
    test_frac: float = 0.2,
    xgb_params: dict[str, Any] | None = None,
    seed: int = 42,
    save: bool = False,
    sample_weights: np.ndarray | None = None,
    feature_cols: list[str] | None = None,
) -> tuple[Any, dict[str, float]]:
    """Train an ``XGBRanker`` (LambdaMART) on the pairwise feature DataFrame.

    Parameters
    ----------
    feature_df : DataFrame
        Output of ``build_pairwise_features`` (with ``add_negatives``); must
        carry the ``feature_cols`` plus ``comp_rank`` and ``qid``.
    sample_weights : ndarray, optional
        Per-row weights (e.g. from ``compute_sample_weights``); aggregated to
        per-group means for the ranker. ``None`` weights every group equally.
    save : bool
        When True, persist the model + metadata sidecar under ``models/``.

    Returns
    -------
    (model, metrics_dict)
    """
    from xgboost import XGBRanker

    if feature_cols is None:
        feature_cols = _DEFAULT_FEATURE_COLS

    feature_df = feature_df.copy()
    feature_df["relevance"] = _rank_to_relevance(feature_df["comp_rank"])

    train_df, test_df = _group_split(feature_df, test_frac=test_frac, seed=seed)

    # Split row weights to the train rows, then aggregate to per-group weights
    # (XGBRanker wants one weight per query group when `group` is used).
    if sample_weights is not None:
        train_mask = feature_df.index.isin(train_df.index)
        w_train = _aggregate_weights_to_groups(train_df, sample_weights[train_mask])
    else:
        w_train = None

    params = _ranker_params(seed=seed)
    if xgb_params:
        params.update(xgb_params)

    model = XGBRanker(**params)
    fit_kwargs: dict[str, Any] = {
        "X": train_df[feature_cols].values,
        "y": train_df["relevance"].values,
        "group": _get_group_sizes(train_df),
        "verbose": False,
    }
    if w_train is not None:
        fit_kwargs["sample_weight"] = w_train
    model.fit(**fit_kwargs)

    metrics = _evaluate_split(model, test_df, feature_cols=feature_cols)
    if save:
        _save_model(model, metrics, params, feature_cols=feature_cols, model_class="XGBRanker")
    return model, metrics


def train_classifier(
    feature_df: pd.DataFrame,
    test_frac: float = 0.2,
    xgb_params: dict[str, Any] | None = None,
    seed: int = 42,
    save: bool = False,
    sample_weights: np.ndarray | None = None,
    feature_cols: list[str] | None = None,
) -> tuple[Any, dict[str, float]]:
    """Train a pointwise ``XGBClassifier`` (comp / not-comp) on the same features.

    Each (subject, comp) row is labeled 1 if it is a labeled comp (``comp_rank``
    > 0) and 0 otherwise, and scored independently -- no query groups. Class
    imbalance (many sampled negatives per positive) is handled with
    ``scale_pos_weight``. The predicted probability is the ranking score
    (see ``_model_scores``), so the classifier is evaluated on the *same*
    NDCG/Recall path as the ranker for a like-for-like comparison.

    Returns
    -------
    (model, metrics_dict)
    """
    from xgboost import XGBClassifier

    if feature_cols is None:
        feature_cols = _DEFAULT_FEATURE_COLS

    feature_df = feature_df.copy()
    feature_df["relevance"] = _rank_to_relevance(feature_df["comp_rank"])
    feature_df["_label"] = (feature_df["comp_rank"].fillna(0).astype(int) > 0).astype(int)

    train_df, test_df = _group_split(feature_df, test_frac=test_frac, seed=seed)

    y_train = train_df["_label"].values
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0

    params = _classifier_params(seed=seed, scale_pos_weight=spw)
    if xgb_params:
        params.update(xgb_params)

    model = XGBClassifier(**params)
    fit_kwargs: dict[str, Any] = {
        "X": train_df[feature_cols].values,
        "y": y_train,
        "verbose": False,
    }
    if sample_weights is not None:
        # Classifiers weight per row (not per group), so pass the train slice as-is.
        train_mask = feature_df.index.isin(train_df.index)
        fit_kwargs["sample_weight"] = sample_weights[train_mask]
    model.fit(**fit_kwargs)

    metrics = _evaluate_split(model, test_df, feature_cols=feature_cols)
    if save:
        _save_model(model, metrics, params, feature_cols=feature_cols, model_class="XGBClassifier")
    return model, metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _ndcg_at_k(relevances: np.ndarray, k: int = 10) -> float:
    """NDCG@k for a single query's ranked relevance scores."""
    rel = relevances[:k]
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(relevances)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _recall_at_k(predicted_ids: set, true_ids: set, k: int = 10) -> float:
    """Fraction of true comps found in the top-k predictions."""
    if not true_ids:
        return 0.0
    return len(predicted_ids & true_ids) / len(true_ids)


def _evaluate_split(
    model: Any,
    test_df: pd.DataFrame,
    k: int = 10,
    feature_cols: list[str] | None = None,
) -> dict[str, float]:
    """Closed-set evaluation: rank each subject's candidate group by model score.

    Works for both model families -- ``_model_scores`` returns ``predict_proba``
    for classifiers and ``predict`` for rankers, so the NDCG/Recall/MRR path is
    identical. Expects ``test_df`` to carry ``relevance``, ``qid``, and
    ``comp_property_id``.
    """
    if feature_cols is None:
        feature_cols = _DEFAULT_FEATURE_COLS

    test_df = test_df.copy()
    test_df["pred_score"] = _model_scores(model, test_df[feature_cols].values)

    ndcgs, recalls, mrrs = [], [], []
    for _qid, group in test_df.groupby("qid"):
        ranked = group.sort_values("pred_score", ascending=False)
        rels = ranked["relevance"].values
        ndcgs.append(_ndcg_at_k(rels, k=k))

        true_comps = set(group[group["relevance"] > 0]["comp_property_id"])
        pred_top_k = set(ranked.head(k)["comp_property_id"])
        recalls.append(_recall_at_k(pred_top_k, true_comps, k=k))

        rel_positions = np.where(rels > 0)[0]
        mrrs.append(1.0 / (rel_positions[0] + 1) if len(rel_positions) else 0.0)

    return {
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "n_test_subjects": len(ndcgs),
    }


def cross_validate(
    feature_df: pd.DataFrame,
    n_folds: int = 5,
    xgb_params: dict[str, Any] | None = None,
    k: int = 10,
    sample_weights: np.ndarray | None = None,
    feature_cols: list[str] | None = None,
    model_kind: str = "ranker",
) -> pd.DataFrame:
    """Group k-fold cross-validation (split by subject), one row per fold.

    This is the **closed-set** view: within each fold every subject is scored
    against its own small candidate group. Pass ``model_kind="classifier"`` to
    cross-validate the pointwise classifier instead of the LambdaMART ranker;
    both are scored on the same NDCG@k / Recall@k metrics.
    """
    from xgboost import XGBRanker, XGBClassifier

    if feature_cols is None:
        feature_cols = _DEFAULT_FEATURE_COLS
    if model_kind not in ("ranker", "classifier"):
        raise ValueError(f"model_kind must be 'ranker' or 'classifier', got {model_kind!r}")

    feature_df = feature_df.copy()
    feature_df["relevance"] = _rank_to_relevance(feature_df["comp_rank"])
    feature_df["_label"] = (feature_df["comp_rank"].fillna(0).astype(int) > 0).astype(int)

    gkf = GroupKFold(n_splits=n_folds)
    groups = feature_df["qid"].values

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(feature_df, groups=groups)):
        train_df = feature_df.iloc[train_idx]
        test_df = feature_df.iloc[test_idx]

        if model_kind == "classifier":
            y_train = train_df["_label"].values
            n_pos = int(y_train.sum())
            spw = ((len(y_train) - n_pos) / n_pos) if n_pos > 0 else 1.0
            params = _classifier_params(scale_pos_weight=spw)
            if xgb_params:
                params.update(xgb_params)
            model: Any = XGBClassifier(**params)
            fit_kwargs: dict[str, Any] = {
                "X": train_df[feature_cols].values,
                "y": y_train,
                "verbose": False,
            }
            if sample_weights is not None:
                fit_kwargs["sample_weight"] = sample_weights[train_idx]
        else:
            params = _ranker_params()
            if xgb_params:
                params.update(xgb_params)
            model = XGBRanker(**params)
            fit_kwargs = {
                "X": train_df[feature_cols].values,
                "y": train_df["relevance"].values,
                "group": _get_group_sizes(train_df),
                "verbose": False,
            }
            if sample_weights is not None:
                fit_kwargs["sample_weight"] = _aggregate_weights_to_groups(
                    train_df, sample_weights[train_idx]
                )
        model.fit(**fit_kwargs)

        metrics = _evaluate_split(model, test_df, k=k, feature_cols=feature_cols)
        metrics["fold"] = fold
        fold_metrics.append(metrics)

    return pd.DataFrame(fold_metrics)


def full_pool_diagnostic(
    model: Any,
    pairs: pd.DataFrame,
    unified: pd.DataFrame,
    k_values: tuple[int, ...] = (10, 25),
    feature_cols: list[str] | None = None,
    same_type_only: bool = True,
) -> dict[str, float]:
    """Full-pool retrieval diagnostic: rank each subject against the WHOLE pool.

    Unlike ``cross_validate`` (closed set), this scores every labeled subject
    against its entire radius candidate pool -- the realistic retrieval setting.
    For each of the subject's true (expert-labeled) comps it records the rank the
    model assigns within the full pool, then reports:

    * ``top{k}_share`` -- fraction of true comps that land in the model's top-k.
    * ``median_rank``  -- median full-pool rank of the true comps (lower is better).

    This is where the per-row classifier tends to overtake the LambdaMART ranker
    even when the ranker wins closed-set -- the closed-set / full-pool reversal.

    Notes
    -----
    Uses the same pooling + scoring path as ``predict_comps`` (same radius and
    type hard filters), so the diagnostic reflects inference-time behavior. A
    true comp filtered out of the pool counts as a miss (rank = pool size + 1).
    """
    if feature_cols is None:
        feature_cols = _feature_cols_for_model(model)
    ks = tuple(int(k) for k in k_values)

    positives = pairs[pairs["comp_rank"].fillna(0).astype(int) > 0]
    hits: dict[int, list[float]] = {k: [] for k in ks}
    ranks: list[int] = []

    pool_size = len(unified)
    for sid, group in positives.groupby("subject_property_id"):
        true_comps = set(group["comp_property_id"])
        if not true_comps:
            continue
        ranked = predict_comps(
            model, sid, unified, n=pool_size,
            feature_cols=feature_cols, same_type_only=same_type_only,
        )
        if ranked.empty:
            continue
        order = ranked["comp_property_id"].tolist()
        rank_of = {cid: i + 1 for i, cid in enumerate(order)}
        miss_rank = len(order) + 1
        for comp_id in true_comps:
            r = rank_of.get(comp_id, miss_rank)
            ranks.append(r)
            for k in ks:
                hits[k].append(1.0 if r <= k else 0.0)

    out: dict[str, float] = {}
    for k in ks:
        out[f"top{k}_share"] = float(np.mean(hits[k])) if hits[k] else 0.0
    out["median_rank"] = float(np.median(ranks)) if ranks else float("nan")
    out["n_subjects"] = int(positives["subject_property_id"].nunique())
    out["n_true_comps"] = len(ranks)
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_comps(
    model: Any,
    subject_id: str,
    unified: pd.DataFrame,
    n: int = 10,
    as_of_date: date | None = None,
    same_type_only: bool = False,
    feature_cols: list[str] | None = None,
    max_type_distance: float | None = None,
) -> pd.DataFrame:
    """Rank candidate comps for one subject and return the top-n.

    Applies the inference hard filters (within ``MAX_RADIUS_MILES``, has
    coordinates, optionally same ``main_type`` or within a curated type
    distance), builds pairwise features for every surviving candidate, scores
    them with the model, and returns the top-n.

    Parameters
    ----------
    same_type_only : bool
        When False (default), considers all property types (the cross-type
        model). True restricts candidates to the subject's ``main_type``.
    feature_cols : list of str, optional
        Feature columns to score on. When None, auto-selected from the model's
        input width via ``_feature_cols_for_model``.
    max_type_distance : float, optional
        When set (and ``same_type_only=False``), restricts candidates to those
        whose ``(main_type, property_subtype)`` tuple is within this distance of
        the subject's per ``data/type_similarity.csv``. Ignored when
        ``same_type_only=True``.

    Returns
    -------
    DataFrame of the top-n comps with ``comp_property_id``, ``pred_score``,
    ``rank_position``, ``distance_miles`` (from the pairwise features), and
    merged-back property attributes.
    """
    import math

    if feature_cols is None:
        feature_cols = _feature_cols_for_model(model)

    subj_row = unified[unified["property_id"] == subject_id]
    if subj_row.empty:
        raise KeyError(f"Subject '{subject_id}' not found in unified table.")
    subj = subj_row.iloc[0]

    # --- Hard type filter --------------------------------------------------
    if same_type_only:
        type_mask = unified["main_type"] == subj["main_type"]
    elif max_type_distance is not None:
        from comp_ranker.scoring.type_similarity import type_distance

        subj_main = subj["main_type"]
        subj_sub_raw = subj.get("property_subtype") if "property_subtype" in unified.columns else ""
        subj_sub = (
            "" if subj_sub_raw is None or (isinstance(subj_sub_raw, float) and pd.isna(subj_sub_raw))
            else str(subj_sub_raw)
        )
        if "property_subtype" in unified.columns:
            sub_col = unified["property_subtype"].fillna("").astype(str)
        else:
            sub_col = pd.Series([""] * len(unified), index=unified.index)
        main_col = unified["main_type"].astype("object")
        pairs_col = pd.Series(list(zip(main_col, sub_col)), index=unified.index)
        allow_map = {
            (t_main, t_sub): (
                (type_distance(subj_main, subj_sub, str(t_main), str(t_sub)) is not None)
                and type_distance(subj_main, subj_sub, str(t_main), str(t_sub)) <= max_type_distance
            )
            for (t_main, t_sub) in pairs_col.dropna().unique()
            if t_main is not None and (not isinstance(t_main, float) or not pd.isna(t_main))
        }
        type_mask = pairs_col.map(allow_map).fillna(False).astype(bool)
    else:
        type_mask = pd.Series(True, index=unified.index)

    pool = unified[
        type_mask
        & (unified["property_id"] != subject_id)
        & unified["lat"].notna()
        & unified["lon"].notna()
    ].copy()
    if pool.empty:
        return pd.DataFrame()

    # --- Radius filter (vectorized haversine) ------------------------------
    lat1 = math.radians(float(subj["lat"]))
    lon1 = math.radians(float(subj["lon"]))
    pool_lat = np.radians(pool["lat"].to_numpy(dtype=float))
    pool_lon = np.radians(pool["lon"].to_numpy(dtype=float))
    dphi = pool_lat - lat1
    dlam = pool_lon - lon1
    a = np.sin(dphi / 2) ** 2 + np.cos(lat1) * np.cos(pool_lat) * np.sin(dlam / 2) ** 2
    dists = 3958.8 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    keep = dists <= MAX_RADIUS_MILES
    pool = pool.loc[keep].copy()
    if pool.empty:
        return pd.DataFrame()

    # --- Build features for every candidate, score, take top-n -------------
    candidate_pairs = pd.DataFrame({
        "subject_property_id": subject_id,
        "comp_property_id": pool["property_id"].values,
    })
    features = build_pairwise_features(candidate_pairs, unified, as_of_date=as_of_date)
    if features.empty:
        return pd.DataFrame()

    features["pred_score"] = _model_scores(model, features[feature_cols].values)
    top = features.nlargest(n, "pred_score")

    merge_cols = [c for c in (
        "property_id", "property_name", "full_address", "sqft", "num_units",
        "year_built", "bldg_class", "submarket", "metro",
    ) if c in unified.columns]
    result = top.merge(
        unified[merge_cols],
        left_on="comp_property_id",
        right_on="property_id",
        how="left",
    )
    result["rank_position"] = range(1, len(result) + 1)
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_model(
    model: Any,
    metrics: dict,
    params: dict,
    feature_cols: list[str] | None = None,
    model_name: str = "model_latest",
    model_class: str = "XGBRanker",
) -> str:
    """Save a model + schema-v1 metadata sidecar under ``models/``."""
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if feature_cols is None:
        feature_cols = _DEFAULT_FEATURE_COLS

    model_path = os.path.join(_MODEL_DIR, f"{model_name}.json")
    model.save_model(model_path)

    meta = build_meta_v1(
        model=model,
        feature_cols=feature_cols,
        metrics=metrics,
        xgb_params=params,
        model_name=model_name,
        model_class=model_class,
        seed=params.get("random_state") if isinstance(params, dict) else None,
        source_script="comp_ranker.models.ranker::_save_model",
    )
    meta_path = os.path.join(_MODEL_DIR, f"{model_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return model_path


def load_model(path: str) -> Any:
    """Load a saved model, restoring the right estimator class.

    Reads the schema-v1 meta sidecar (``<stem>_meta.json``) next to the model
    file. When ``meta["model_class"] == "XGBClassifier"`` the booster loads into
    an ``XGBClassifier`` (so ``predict_proba`` is available); otherwise into an
    ``XGBRanker``. ``predict_comps`` dispatches scoring by class (see
    ``_model_scores``), so callers don't need to know which kind they loaded.
    """
    model_class = "XGBRanker"
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                model_class = json.load(f).get("model_class") or "XGBRanker"
        except (json.JSONDecodeError, OSError):
            model_class = "XGBRanker"

    if model_class == "XGBClassifier":
        from xgboost import XGBClassifier

        model: Any = XGBClassifier()
    else:
        from xgboost import XGBRanker

        model = XGBRanker()
    model.load_model(path)
    return model


def _model_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Ranking score for ``X``: ``P(comp)`` for classifiers, raw margin else.

    ``XGBClassifier.predict`` returns hard 0/1 labels (useless for ranking), so
    classifiers are scored via ``predict_proba``. Rankers have no
    ``predict_proba`` and rank directly on ``predict``. The probability is
    monotone in the booster margin, so candidate ordering is preserved.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)
