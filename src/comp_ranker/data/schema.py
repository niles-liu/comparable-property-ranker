"""Unified property schema + the corpus-loading boundary.

This module documents the single schema every downstream stage expects, and
provides a ``load_corpus()`` boundary that the rest of the package reads
through. The real system populates this corpus from three commercial
real-estate data sources (see "Data sources" below) via the cross-source
entity resolution in :mod:`comp_ranker.features.entity_resolution`; that
proprietary connector layer is intentionally **not** part of this repository.

To run against real data, point ``load_corpus()`` at a parquet file (or supply
your own loader) whose columns match :data:`UNIFIED_COLUMNS`. The synthetic
demo in ``examples/`` writes exactly such a parquet so the whole pipeline runs
end-to-end with no external dependencies.

Data sources (genericized)
--------------------------
The corpus is assembled by resolving the same physical building across three
independent sources, each contributing different coverage and fields:

- **Source A** -- a commercial-property listings & inventory database. Broadest
  coverage and the most complete physical attributes; canonical IDs use the
  ``A_`` prefix.
- **Source B** -- a commercial real-estate transactions database; contributes
  sale history. IDs use the ``B_`` prefix.
- **Source C** -- an operating / financial-performance data source; contributes
  occupancy and operating fields. IDs use the ``C_`` prefix.

A record can appear in more than one source under different names, abbreviated
addresses, and slightly different coordinates. Entity resolution clusters those
records into one row per real-world property and merges their fields by source
priority (A > B > C for scalar attributes; most-recent for sale dates).

Unified schema
--------------
``property_id``     str    Source-prefixed ID, e.g. ``A_123``, ``B_456``.
``property_name``   str
``full_address``    str | None
``lat``             float
``lon``             float
``sqft``            float | None   Total building SF (commercial) or None.
``num_units``       int   | None   Unit count (multifamily) or None.
``year_built``      int   | None
``year_renovated``  int   | None
``bldg_class``      str   | None   Normalized: ``'A'``, ``'B'``, ``'C'``, or raw.
``main_type``       str   | None   Normalized via ``PROPERTY_TYPE_MAP``.
``property_subtype``str   | None
``submarket``       str   | None
``metro``           str   | None
``num_stories``     float | None
``occupancy``       float | None   Fractional 0-1 (normalized across sources).
``land_area_ac``    float | None
``parking_ratio``   float | None
``avg_unit_size``   float | None   Multifamily only.
``last_sale_date``  datetime | None
``source``          str | list     Contributing source label(s).
``source_ids``      list           All contributing source IDs (post-merge).
"""

from __future__ import annotations

import os

import pandas as pd

# The full unified column set the pipeline reads. Any loader that produces a
# DataFrame with (at least) these columns can drive the whole system.
UNIFIED_COLUMNS: list[str] = [
    "property_id",
    "property_name",
    "full_address",
    "lat",
    "lon",
    "sqft",
    "num_units",
    "year_built",
    "year_renovated",
    "bldg_class",
    "main_type",
    "property_subtype",
    "submarket",
    "metro",
    "num_stories",
    "occupancy",
    "land_area_ac",
    "parking_ratio",
    "avg_unit_size",
    "last_sale_date",
    "source",
]

# Source-ID prefixes, by source label. Entity resolution prefers the Source A
# ID as the canonical cluster ID when available.
SOURCE_ID_PREFIXES: dict[str, str] = {
    "source_a": "A_",
    "source_b": "B_",
    "source_c": "C_",
}

# Source priority for scalar-field merging during entity resolution: Source A is
# the most complete, then B, then C.
SOURCE_PRIORITY: list[str] = ["source_a", "source_b", "source_c"]


def load_corpus(path: str | None = None) -> pd.DataFrame:
    """Load the unified property corpus as a DataFrame.

    This is the package's data boundary. In the real deployment the corpus is
    built by fetching three proprietary sources and resolving them with
    :func:`comp_ranker.features.entity_resolution.match_and_merge`; that
    connector layer is not shipped here.

    Default behavior: read a parquet file whose schema matches
    :data:`UNIFIED_COLUMNS`. The path resolves (in order) to the ``path``
    argument, the ``COMP_RANKER_CORPUS`` environment variable, or
    ``data/corpus.parquet`` relative to the repository root. The synthetic demo
    writes that file, so the demo runs with no configuration.

    Parameters
    ----------
    path : str, optional
        Path to a parquet file with the unified schema. If None, falls back to
        the ``COMP_RANKER_CORPUS`` env var, then ``data/corpus.parquet``.

    Returns
    -------
    pd.DataFrame
        One row per resolved real-world property.

    Raises
    ------
    FileNotFoundError
        If no corpus parquet can be located. The message points at the
        synthetic-corpus generator so the demo is discoverable.
    """
    if path is None:
        path = os.environ.get("COMP_RANKER_CORPUS")
    if path is None:
        root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
        path = os.path.abspath(os.path.join(root, "data", "corpus.parquet"))

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No corpus parquet found at {path!r}.\n"
            "This repository ships code + a synthetic demo only; the real "
            "proprietary corpus is not included.\n"
            "Run `python examples/generate_synthetic_corpus.py` to create a "
            "small fake corpus, or set COMP_RANKER_CORPUS to your own parquet "
            "matching comp_ranker.data.schema.UNIFIED_COLUMNS."
        )
    return pd.read_parquet(path)
