"""Model training, evaluation, and inference for comparable selection.

Exposes a LambdaMART ranker and a pointwise binary classifier over a shared
pairwise feature representation, a common closed-set CV path, the full-pool
retrieval diagnostic, and the inference entry point.
"""

from comp_ranker.models.ranker import (
    train_ranker,
    train_classifier,
    cross_validate,
    full_pool_diagnostic,
    predict_comps,
    load_model,
)

__all__ = [
    "train_ranker",
    "train_classifier",
    "cross_validate",
    "full_pool_diagnostic",
    "predict_comps",
    "load_model",
]
