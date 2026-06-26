# Methodology & validation

This document records the evaluation methodology, the central empirical finding
and the reasoning behind it, the closed-set → full-pool reversal lesson, and the
log of levers that were tried and found null or negative. All numbers are from
the real system (built against a proprietary corpus that is not in this repo);
the synthetic demo only reproduces the *shape* of the headline result.

---

## 1. The problem and the data contract

Given a subject commercial property and a valuation date, rank the universe of
candidates by how good a comparable each is. The corpus is ~5.87M unified
property rows assembled by cross-source entity resolution over three data
sources (Source A: listings/inventory; Source B: transactions; Source C:
operating/financial — see the README). Labels are ~219 subjects / ~2,379
subject–comp pairs.

**Labels are expert-selected, not ground truth.** They are the comps an expert
actually chose from historical underwriting/valuation decisions. So every metric
is *label-relative* (agreement with expert selection), not absolute accuracy.
This bounds how the numbers should be read and is itself a reason qualitative
review stays on the critical path.

---

## 2. Evaluation discipline (gate-first)

Three rules govern every comparison:

1. **Group-aware splitting.** Splits partition by subject (query group); no
   subject is in both train and test. The conservative extreme is
   leave-one-subject-out (LOSO), used to validate the headline closed-set lift.

2. **Multi-seed CV.** A single panel/top-k read is too noisy here; comparisons
   use the mean over ≥3 seeds. Seed-variance attribution showed the
   negative-sampling draw dominates metric variance (SD ≈ 0.98 pp NDCG@10) over
   the CV partition (≈ 0.59) and the XGBoost seed (≈ 0.46) — so determinizing
   the negative draw buys most of the achievable variance reduction.

3. **Closed-set vs full-pool, reported separately, decided on full-pool.**
   - *Closed-set:* each subject is ranked against {labeled positives + ~20
     sampled negatives} ≈ 30 candidates. Cheap; good for relative comparison.
   - *Full-pool:* each subject is ranked against the entire inference-radius
     pool (10K–250K candidates; median ~100K, p90 ~201K). This is the realistic
     retrieval metric.

   These disagree at large magnitude on this dataset. The same rule-based scorer
   that posts ~47% Recall@10 closed-set posts ~0% Recall@10 at full-pool. **All
   production claims are made on full-pool numbers**; closed-set CV is kept only
   as a relative-comparison tool. (The repo exposes both: `cross_validate` /
   `leave_one_subject_out` for closed-set, `full_pool_diagnostic` for full-pool.)

The gate-first protocol — pre-committing the ship/reject rule *before* running
an experiment, then applying it mechanically — is what made the reversals below
legible rather than ad hoc.

---

## 3. The central finding: classifier > ranker at full-pool

A multi-method bake-off (rule-based baseline + five model variants) on the same
5-seed × 5-fold harness found that an `XGBClassifier(binary:logistic)` on the 18
features beats LambdaMART (`rank:ndcg`) on every measured dimension:

- **Closed-set NDCG@10:** 0.697 (classifier) vs 0.453 (ranker), **+0.244**
  (t = 30.3), wins on every reportable property type, LOSO-validated (5/5 seeds).
- **Full-pool retrieval (~158 reachable subjects, median pool ~100K):**
  classifier top-10 share **0.40% vs 0.18%** (2.2×), top-100 share 3.23% vs
  0.58% (5.6×), median rank of true comps **16K vs 46K** (≈3× compression). The
  classifier wins at every K from 10 to 1000.

### Why the classifier wins (and why this is counterintuitive)

LambdaMART's gradient is computed *pairwise within query groups*. With ~5
positives + 20 negatives per group across ~160 subjects, the within-group
pairwise signal sums to ~12K comparisons per pass — a small effective sample
size. The classifier instead treats each of the ~5K training rows as an
independent labeled example and learns a global row-level score function. At this
label volume the per-row signal exceeds the within-group pairwise signal, so the
two-decade web-search consensus ("listwise beats pointwise") — calibrated on
datasets with thousands of judgments per query — does not transfer.

The objective-level diagnostic sharpens this. Holding the `XGBRanker`
formulation fixed and varying only the objective:

- **At closed-set,** `rank:pairwise` recovers most of what `rank:ndcg` loses
  (NDCG 0.677 vs 0.454), landing in the same band as the classifier.
- **At full-pool,** the picture *inverts*: `rank:pairwise` (top-10 share 0.09%)
  and `rank:map` (0.13%) both **lose** to `rank:ndcg` (0.18%) at small K. Only
  the binary classifier (0.40%) survives both metrics.

Diagnosis: *all* group-aware objectives calibrate to ~30-row training groups and
extrapolate poorly to 100K-candidate inference pools; the classifier is the
unique winner precisely because it has no group structure at train time.

**SHAP mechanism.** The classifier puts ~60.9% of its attention on
`distance_miles` (vs the ranker's ~20.4%) and zeros out three features the ranker
uses. It learns a global "is-this-close-and-similar?" decision rule — winning
gross discrimination — whereas the ranker spreads attention onto quality features
that are near-constant for thin/heterogeneous types (Office / Lodging / Land),
which is exactly where it regressed.

---

## 4. The closed-set → full-pool reversal (the load-bearing lesson)

Closed-set CV is not merely a weaker signal here — it is *anti-correlated* with
the real full-pool metric for several levers. Three separate reversals were
observed, each where a closed-set "this helps" signal was falsified at full-pool:

1. **Labeled-corpus refresh.** After a label refresh, closed-set CV said the
   strict type-bounded classifier *regressed* (−4.8%); paired full-pool
   *overturned* it — top-10 share went 1.60% → 5.11% (3.2×), with every
   type-strict classifier strengthening.

2. **Per-type distance-mask.** A closed-set per-feature LOO flagged `distance` as
   over-weighted for some types; dropping it at full-pool **hurt every type** at
   top-25 (Office −2.79 pp, Multifamily −1.19 pp, CIs excluding 0). The
   closed-set signal reversed. Distance is essential.

3. **Monotone constraints.** Closed-set was at-or-above parity (with apparent
   Lodging/Industrial gains); full-pool **hurt every K** (top-25 −0.61 pp,
   CI [−1.05, −0.25]) and the flagged types reversed. Over-regularization strips
   discriminative interactions.

**Operating guardrail (now standing):** any "this helps a type / this feature is
over-weighted" signal observed in closed-set CV **must be full-pool-confirmed
before it is believed.** Clustered bootstrap (over the shared-signal cluster —
event/subject — not the row) is used for the per-decision metric CIs, because
decision-level n overstates precision.

---

## 5. The recall reframe: the ceiling is in candidate generation

The most decision-relevant understanding result is not a lift, it's a reframe.
On the strict retriever the binding constraint is **pool membership, not
ranking**: only ~10.5% of expert-selected comps are even present in the
candidate pool (65% of subjects are 0-recoverable; niche types ~0%). A *perfect*
ranker therefore caps at **recall ≈ 0.105**.

Implication: incremental 18-feature *ranking* has a low ceiling on the strict
retriever; **recall lives in candidate generation**, not in the re-ranker. The
forward gains are a data/coverage problem — better candidate generation, or more
labels (especially curated *negative* labels) — not a better re-ranker. A
learning-curve analysis corroborates that label volume has historically been the
single biggest lever.

---

## 6. Calibration

Isotonic regression on the classifier's probabilities cuts the expected
calibration error ~5× (ECE 0.0395 → 0.0072; Platt gives 0.024). Brier is
unchanged (the transform is monotone, so ranking is unaffected). This is
ship-ready *only* if a confidence number is ever surfaced to a user; it is not an
accuracy lever, so it is held conditional rather than wired into the default
path. (The worst-calibrated segments are Land and Industrial.)

---

## 7. The exhausted-lever log

A deliberately exhaustive "push every runnable lever" sweep closed the
accuracy/recall search at the current label volume. Everything below was tested
(most multi-seed, most full-pool-confirmed) and found **null or negative**:

| Lever | Result |
|-------|--------|
| **Feature width** — 25-feature combined set, average-unit-size, subtype, occupancy stacking, type-distance as a trainable feature, conservative interaction terms | all null |
| **Batch feature ingestion** (additional financial/physical columns, 11 materialized feature groups) | closed null under both classifier and ranker — the 18-feature width is exhausted |
| **Hard-negative mining** (retrain with nearby non-comps mined from the classifier's own top-1000) | *catastrophic* — full-pool top-10 share collapses 0.40% → 0.00%, median rank degrades 5×. Label sparsity breaks naive mining: most "hard negatives" are unlabeled-but-comp-like, so the model learns to suppress comp-like features generally |
| **Seed ensembling** | variance-insurance only, no lift |
| **Monotone constraints** | full-pool reject (3rd reversal — see §4) |
| **Per-metro routing** | infeasible — no metro has ≥30 labeled subjects |
| **Negatives-per-subject sweep, MF rebalancing** | null/negative |
| **Two-stage retrieval** (cheap retriever → shortlist re-ranker) | rejected — *helps* a weak retriever (+0.49 pp) but **hurts** the real strict retriever (0.85% vs 1.72% top-10, CI [−1.48, −0.40], 0/5 seeds). No within-type signal in the 18 features to re-rank on |
| **Candidate-generation relaxation** (wider radius / broader type filter) | recall *worse* — ranking collapses faster than coverage grows |
| **Dropping distance / per-type distance-mask** | full-pool reject (2nd reversal — distance is essential) |

The one confirmed *no-new-data* win was a name/address text-similarity feature
(+0.29 pp top-25, 10 seeds), of which ~40% is lodging brand-name matching — a
known circularity, so it is offered address-only for lodging.

**Remaining forward paths are all blockers, not levers:** acquire labels
(curated negatives would unblock the candidate-generation ceiling and the
two-stage approach; more positive labels), or expand the corpus. These are
data-acquisition decisions, not modeling work. The runnable improvement space is
considered exhausted at this label volume.

---

## 8. Soundness checks

- **Leakage:** measured < 0.1% under the group-aware split.
- **Label noise characterized:** ~28.5% of labels resolve below 80%
  match-confidence; ~76% of (type × metro) cells are singletons. Match-score
  weighting and leave-one-out influence analysis are available as automated,
  reproducible label-cleaning passes (`compute_sample_weights`,
  `leave_one_subject_out`) rather than manual review of every set.
- **Reproducibility:** the synthetic generator and the demo are seeded; the
  ranker/classifier training takes an explicit `seed`.

---

## 9. What the demo shows vs what's real

The synthetic demo (`examples/demo.py`) plants expert comp sets as "same type,
same metro, closest by noisy size+age," then runs the real pipeline. It
reproduces the *qualitative* reversal — the ranker posts higher closed-set NDCG
while the classifier wins full-pool top-k share by a wide margin — on a tiny
fake corpus. It is **not** a benchmark; the quantitative findings above are from
the real system and should be read as such.
