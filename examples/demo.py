"""End-to-end demo on the synthetic corpus.

Runs the full pipeline:
  1. Entity resolution    -- merge the three synthetic sources into one corpus.
  2. Label ingestion      -- resolve the expert comp-set CSV against the corpus.
  3. Pairwise features    -- build the 18-feature vectors (+ sampled negatives).
  4. Train + compare      -- a LambdaMART ranker and an XGBoost classifier,
                             reported both closed-set (CV) and full-pool.
  5. Model-vs-expert      -- for one subject, show the expert's comps alongside
                             each model's top picks.

This is illustrative on small synthetic data, not a benchmark. The closed-set
vs full-pool gap and the classifier-vs-ranker comparison are the points of
interest -- see METHODOLOGY.md for the real-data findings.

Run:
    python examples/generate_synthetic_corpus.py   # once, to create the data
    python examples/demo.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from comp_ranker.features.entity_resolution import match_and_merge
from comp_ranker.features.normalize import normalize_unified_df
from comp_ranker.features.labels import ingest_comp_sets
from comp_ranker.features.pairwise import build_pairwise_features, add_negatives, FEATURE_COLS_PLUS_OCC
from comp_ranker.models.ranker import (
    train_ranker,
    train_classifier,
    cross_validate,
    full_pool_diagnostic,
    predict_comps,
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
FEATURE_COLS = FEATURE_COLS_PLUS_OCC  # the 18-feature production set


def _require(path: str) -> str:
    if not os.path.exists(path):
        sys.exit(
            f"Missing {path}.\nRun `python examples/generate_synthetic_corpus.py` first."
        )
    return path


def main() -> None:
    rng = np.random.default_rng(42)

    # --- 1. Entity resolution ---------------------------------------------
    print("=" * 70)
    print("1. CROSS-SOURCE ENTITY RESOLUTION")
    print("=" * 70)
    source_a = pd.read_parquet(_require(os.path.join(DATA_DIR, "source_a.parquet")))
    source_b = pd.read_parquet(_require(os.path.join(DATA_DIR, "source_b.parquet")))
    source_c = pd.read_parquet(_require(os.path.join(DATA_DIR, "source_c.parquet")))
    print(f"Source A: {len(source_a):,} rows | Source B: {len(source_b):,} | "
          f"Source C: {len(source_c):,}")

    unified = match_and_merge(source_a, source_b, source_c)
    unified = normalize_unified_df(unified)
    unified.to_parquet(os.path.join(DATA_DIR, "corpus.parquet"), index=False)
    print(f"Unified corpus: {len(unified):,} resolved properties "
          f"(from {len(source_a) + len(source_b) + len(source_c):,} source rows)")

    # --- 2. Label ingestion ------------------------------------------------
    print("\n" + "=" * 70)
    print("2. EXPERT-LABEL INGESTION (label-relative; see METHODOLOGY.md)")
    print("=" * 70)
    pairs, unmatched = ingest_comp_sets(unified, _require(os.path.join(DATA_DIR, "comp_sets.csv")))
    n_subjects = pairs["subject_property_id"].nunique() if not pairs.empty else 0
    print(f"Resolved {len(pairs)} labeled (subject, comp) pairs across "
          f"{n_subjects} subjects ({len(unmatched)} entries unmatched).")
    if pairs.empty:
        sys.exit("No labeled pairs resolved -- cannot train. Regenerate the synthetic data.")

    # --- 3. Pairwise features + negatives ----------------------------------
    print("\n" + "=" * 70)
    print("3. PAIRWISE FEATURE ENGINEERING")
    print("=" * 70)
    pairs_aug = add_negatives(pairs, unified, neg_per_subject=20, seed=42,
                              same_type_only=True)
    feats = build_pairwise_features(pairs_aug, unified)
    n_pos = int((feats["comp_rank"].fillna(0).astype(int) > 0).sum())
    n_neg = len(feats) - n_pos
    print(f"Feature matrix: {len(feats)} rows ({n_pos} positive, {n_neg} negative), "
          f"{len(FEATURE_COLS)} features each.")

    # --- 4. Train + compare ------------------------------------------------
    print("\n" + "=" * 70)
    print("4. RANKER vs CLASSIFIER -- closed-set (CV) AND full-pool")
    print("=" * 70)
    n_folds = min(5, n_subjects)

    ranker_cv = cross_validate(feats, n_folds=n_folds, feature_cols=FEATURE_COLS,
                               model_kind="ranker")
    clf_cv = cross_validate(feats, n_folds=n_folds, feature_cols=FEATURE_COLS,
                            model_kind="classifier")
    print("\nClosed-set CV (each subject ranked vs ~21 candidates):")
    print(f"  LambdaMART ranker  NDCG@10 = {ranker_cv['ndcg@10'].mean():.3f}  "
          f"Recall@10 = {ranker_cv['recall@10'].mean():.3f}")
    print(f"  XGBoost classifier NDCG@10 = {clf_cv['ndcg@10'].mean():.3f}  "
          f"Recall@10 = {clf_cv['recall@10'].mean():.3f}")

    ranker, _ = train_ranker(feats, feature_cols=FEATURE_COLS, seed=42)
    clf, _ = train_classifier(feats, feature_cols=FEATURE_COLS, seed=42)

    ranker_fp = full_pool_diagnostic(ranker, pairs, unified, k_values=(10, 25),
                                     feature_cols=FEATURE_COLS)
    clf_fp = full_pool_diagnostic(clf, pairs, unified, k_values=(10, 25),
                                  feature_cols=FEATURE_COLS)
    print("\nFull-pool retrieval (each subject ranked vs the WHOLE radius pool):")
    print(f"  LambdaMART ranker  top-10 share = {ranker_fp['top10_share']:.3f}  "
          f"top-25 share = {ranker_fp['top25_share']:.3f}  "
          f"median rank = {ranker_fp['median_rank']:.0f}")
    print(f"  XGBoost classifier top-10 share = {clf_fp['top10_share']:.3f}  "
          f"top-25 share = {clf_fp['top25_share']:.3f}  "
          f"median rank = {clf_fp['median_rank']:.0f}")
    print("\n(On real data the classifier wins full-pool even where the ranker "
          "is competitive\n closed-set -- the closed-set/full-pool reversal. "
          "See METHODOLOGY.md.)")

    # --- 5. Model vs expert ------------------------------------------------
    print("\n" + "=" * 70)
    print("5. MODEL vs EXPERT (one subject)")
    print("=" * 70)
    example_subject = pairs["subject_property_id"].iloc[0]
    expert_comps = set(
        pairs[(pairs["subject_property_id"] == example_subject)
              & (pairs["comp_rank"].astype(int) > 0)]["comp_property_id"]
    )
    subj_row = unified[unified["property_id"] == example_subject].iloc[0]
    print(f"Subject: {subj_row['property_name']}  [{subj_row['main_type']}, "
          f"{subj_row['metro']}]")
    print(f"Expert chose {len(expert_comps)} comps.")

    clf_top = predict_comps(clf, example_subject, unified, n=10,
                            feature_cols=FEATURE_COLS, same_type_only=True)
    if not clf_top.empty:
        hits = clf_top["comp_property_id"].isin(expert_comps)
        print(f"\nClassifier top-10 (matched the expert on {int(hits.sum())} of "
              f"{len(expert_comps)}):")
        for _, r in clf_top.iterrows():
            mark = "<-- expert pick" if r["comp_property_id"] in expert_comps else ""
            print(f"  #{int(r['rank_position']):>2}  {str(r['property_name'])[:42]:42}  "
                  f"{r['distance_miles']:.1f} mi  {mark}")

    print("\nDone. This is a small synthetic illustration; the real findings and "
          "metrics\nare in README.md and METHODOLOGY.md.")


if __name__ == "__main__":
    main()
