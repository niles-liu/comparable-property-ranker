# data/

This directory holds the corpus the pipeline reads. **No real data ships with
this repository**; the production system is built against a proprietary
corpus that is not included.

To run the demo, generate a small synthetic corpus here:

```bash
python examples/generate_synthetic_corpus.py
```

That writes (all git-ignored):

- `source_a.parquet`, `source_b.parquet`, `source_c.parquet`: raw per-source
  rows, with deliberate cross-source overlap for entity resolution to merge.
- `comp_sets.csv`: synthetic expert comp-set labels (subject -> ranked comps).
- `corpus.parquet`: the unified table (written by `examples/demo.py`).

To run against your own data instead, provide a parquet matching
`comp_ranker.data.schema.UNIFIED_COLUMNS` and point `COMP_RANKER_CORPUS` at it
(or pass its path to `comp_ranker.data.schema.load_corpus`).

### Optional: `type_similarity.csv`

`comp_ranker.scoring.type_similarity` can read a curated property-type
similarity matrix from `data/type_similarity.csv` (columns: `subject_main_type,
subject_subtype, comp_main_type, comp_subtype, distance`). It is entirely
optional: when absent, the `type_distance` feature is simply NaN (which the
tree models handle natively) and same/cross-type behaviour falls back to the
documented defaults.
