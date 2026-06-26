# comp-ranker — learning-to-rank for comparable commercial properties

A learning-to-rank system that, given a subject commercial-real-estate
property, ranks a large national pool of candidates by how good a
**comparable ("comp")** each one is — reproducing, and ideally improving on,
the comp set an expert would select. This is the retrieval step underneath any
valuation or underwriting workflow that begins with "find me the relevant
comps."

The interesting result here is methodological: **a per-row binary classifier
beats a LambdaMART listwise ranker at realistic full-pool retrieval, even
though the ranker wins in closed-set cross-validation** — a closed-set →
full-pool reversal that turns out to be the load-bearing lesson of the whole
project. See [`METHODOLOGY.md`](METHODOLOGY.md).

> **Data not included.** The production system is built against a proprietary
> corpus assembled from three commercial real-estate data sources, which is not
> part of this repository. What ships here is the **code** plus a **synthetic
> demo** that exercises the entire pipeline end-to-end. All metric values quoted
> below are from the real system; the synthetic demo is illustrative only.

---

## What it does

1. **Assemble a corpus** of ~5.9M unified property rows by resolving the same
   physical building across three independent data sources (cross-source entity
   resolution).
2. **Ingest expert labels** — the comps an analyst actually selected for a set
   of subject properties — and resolve them against the corpus.
3. **Engineer 18 pairwise features** describing how similar a (subject, comp)
   pair is (distance, size ratio, age, building class, occupancy, submarket…).
4. **Rank candidates** with one of three models: a rule-based baseline, a
   LambdaMART ranker, or an XGBoost binary classifier.
5. **Serve** the top-n comps for a subject via a candidate-generation →
   score → rank pipeline.

---

## The corpus & cross-source resolution

The corpus is built by merging three sources, each contributing different
coverage and fields (described generically here):

| Source | Role | ID prefix |
|--------|------|-----------|
| **Source A** | a commercial-property listings & inventory database (broadest coverage, most complete physical attributes) | `A_` |
| **Source B** | a commercial real-estate transactions database (sale history) | `B_` |
| **Source C** | an operating / financial-performance data source (occupancy, operating fields) | `C_` |

The same building can appear in more than one source under different names,
abbreviated addresses, and slightly different coordinates. **Entity resolution**
(`comp_ranker.features.entity_resolution`) clusters those records into one row
per real-world property using a three-tier strategy:

- **Tier 1 — spatial:** a `BallTree` over coordinates links records within ~50 m
  (definitive same-building, no text needed; scales to millions of rows without
  O(n²) cost).
- **Tier 2 — fuzzy address + name:** for near-misses (geocoding error), both the
  normalized address *and* the property name must clear similarity thresholds
  before a match is declared.
- **Tier 3 — parcel number:** exact parcel/APN joins are treated as definitive.

Matched records are unioned into clusters and merged field-by-field by source
priority (A > B > C for scalar attributes; most-recent for sale dates).

Result: **~5.87M unified properties** from the three sources.

---

## The 18 pairwise features

Each (subject, comp) pair becomes a feature vector of relative/differential
signals (`comp_ranker.features.pairwise`). The production set ("18f plus_occ")
is 16 base features plus signed/absolute occupancy difference:

- **Location & size:** `distance_miles`, `log_distance`, `size_ratio`,
  `log_size_ratio`, `size_variance_pct`
- **Age:** `year_built_diff` / `abs_…`, `year_renovated_diff` / `abs_…`
- **Quality / context:** `same_bldg_class`, `bldg_class_ord_diff`,
  `same_submarket`, `same_subtype`, `occupancy_diff` / `abs_occupancy_diff`
- **Recency:** `sale_recency_years`
- (plus land-area, stories, and parking-ratio diffs in the wider sets)

A **cross-type** variant adds three type-awareness features (`same_main_type`,
ordinal type encodings) so a model can learn which cross-type pairings — if any
— experts actually accept, and an optional curated `type_distance` feature folds
in a hand-tuned property-type similarity matrix. Missing values are left as NaN,
which the tree models handle natively.

---

## The models

- **Rule-based baseline** (`comp_ranker.scoring`) — an interpretable comparator.
  Each factor (distance, size, age, class, submarket, recency) starts at 100 and
  loses points as the pair diverges; the composite is a weighted average over
  the factors with data on both sides, with metro-specific distance steps and
  hard pre-filters (radius, same type, size tolerance). Fully vectorized and
  GPU-accelerated (with an automatic CPU fallback) so one subject can be scored
  against a large pool quickly.
- **LambdaMART ranker** (`XGBRanker`, `objective="rank:ndcg"`) — a listwise
  learning-to-rank model on the 18 features.
- **XGBoost classifier** (`XGBClassifier`, `binary:logistic`) — a per-row
  "is this a comp?" model on the *same* 18 features.

---

## The central finding

On this data and label volume, the **binary classifier beats the LambdaMART
ranker on every measured dimension**:

- **Closed-set NDCG@10 +0.244** (t = 30.3, 5/5 seeds, leave-one-subject-out
  validated).
- **Full-pool (realistic retrieval) top-10 share ≈ 2.2×** the ranker's; top-100
  share ≈ 5.6×; median rank of the true comps compressed from ~46K to ~16K.
- Critically, **only the classifier survives the full-pool test.** The listwise
  ranking objectives (`rank:ndcg` / `rank:pairwise` / `rank:map`) can win in
  closed-set CV but *lose* at realistic ~100K-candidate retrieval. The per-row
  classifier is the one formulation that extrapolates — because it has no query
  group structure at training time, every row is an independent labeled example,
  and its row-level decision boundary generalizes across pool sizes.
- **Mechanism (SHAP):** the classifier learns a near-uniform "is this close /
  similar?" decision rule (~60% of its attention on distance); the ranker
  spreads attention onto quality features that are near-constant for thin or
  heterogeneous property types, which is *why* it regressed there.

**Honest framing.** This is a strict improvement for any pre-ranking pipeline,
not a solved problem: full-pool top-10 share is still small in absolute terms
(sub-1%). The win is *much better ordering of candidates*, not perfect
retrieval. And the remaining recall ceiling does not live in the ranker at all —
it lives in **candidate generation** (see below and `METHODOLOGY.md`).

The demo reproduces the *shape* of this reversal on synthetic data: the ranker
posts higher closed-set NDCG while the classifier wins full-pool top-k share by
a wide margin.

---

## Evaluation discipline

- **Group-aware splits.** Every train/test split partitions by *subject* (query
  group), so no subject appears in both — preventing the model from memorizing
  specific properties. The most conservative variant is leave-one-subject-out
  (LOSO), which the headline closed-set numbers are validated against.
- **Multi-seed CV.** Model comparisons use the mean over multiple seeds, because
  the negative-sampling draw is the single largest source of metric variance.
- **Closed-set vs full-pool, reported separately.** *Closed-set* ranks each
  subject against its own labeled positives + ~20 sampled negatives (~30
  candidates); *full-pool* ranks against the entire inference-radius pool
  (10K–250K candidates; median ~100K). These two metrics disagree at large
  magnitude on this data, so every production claim is made on full-pool
  numbers — closed-set CV is kept only as a relative-comparison tool. This is
  the discipline that surfaced the reversal in the first place.

---

## The label-relative caveat

The labels are **expert-selected comparables** drawn from historical
underwriting / valuation decisions. They reflect **expert selection, not ground
truth** — they are the comps an expert chose, not a complete, exhaustive set of
all valid comps. Consequently every metric in this project is
*agreement-with-expert-selection* ("label-relative"), **not** absolute retrieval
accuracy. The numbers should be read as "how well does the model reproduce
expert choices," and qualitative review stays on the critical path. Label
volume on the real system: ~219 labeled subjects / ~2,379 labeled subject–comp
pairs.

---

## Where the ceiling actually is

A diagnostic reframe (`METHODOLOGY.md`) shows that on the strict retriever, only
~10.5% of expert-selected comps are even present in the candidate pool — so a
*perfect* ranker would cap at recall ≈ 0.105. **Recall is a candidate-generation
/ data-coverage problem, not an 18-feature-ranking problem.** This is the key
strategic insight: the remaining gains require better candidate generation or
more labels (especially curated negatives), not a better re-ranker.

---

## Repository layout

```
src/comp_ranker/
  data/schema.py            Unified schema + load_corpus() boundary (no real connector)
  features/
    entity_resolution.py    Cross-source 3-tier matching + merge
    normalize.py            Raw rows -> unified schema
    pairwise.py             18 pairwise features (+ cross-type / type_distance variants)
    labels.py               Resolve expert comp-set CSV into trainable pairs
  scoring/
    config.py               Weights, hard-filter params, type map, metro steps
    scorer.py               Rule-based comparator (vectorized, GPU-optional)
    type_similarity.py      Curated (main_type, subtype) distance matrix
  models/ranker.py          LambdaMART + classifier training, CV/LOSO, full-pool diagnostic, inference
  accel.py                  Optional CuPy/GPU backend with CPU fallback
examples/
  generate_synthetic_corpus.py   Fake 3-source corpus + expert labels (fixed seed)
  demo.py                        End-to-end: resolve -> features -> train -> compare -> model-vs-expert
```

## Quickstart

```bash
pip install -r requirements.txt
python examples/generate_synthetic_corpus.py   # writes synthetic data/ files
python examples/demo.py                         # runs the full pipeline
```

The demo prints the closed-set vs full-pool comparison for the ranker and the
classifier, then shows one subject's expert comps next to the classifier's
top-10 picks.

---

## License

MIT — see [`LICENSE`](LICENSE).
