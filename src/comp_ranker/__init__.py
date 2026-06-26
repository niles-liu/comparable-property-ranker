"""comp_ranker — a learning-to-rank system for comparable commercial-real-estate properties.

Given a subject property, rank a large national pool of candidates by how good a
"comparable" each is, reproducing (and ideally improving on) the comp set an expert
would select.

Package layout
--------------
- ``comp_ranker.data``      Unified property schema + a corpus-loading boundary.
- ``comp_ranker.features``  Entity resolution, normalization, pairwise features, labels.
- ``comp_ranker.scoring``   Interpretable rule-based scorer, type similarity, config.
- ``comp_ranker.models``    LambdaMART ranker + XGBoost classifier training/inference.
- ``comp_ranker.accel``     Optional GPU acceleration with automatic CPU fallback.
"""

__version__ = "0.1.0"
