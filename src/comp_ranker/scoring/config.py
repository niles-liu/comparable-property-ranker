"""Scoring configuration: weights, hard-filter parameters, and metro distance steps.

Weights must sum to 100.  Set a factor's weight to 0 to disable it.
Factors with NULL values in a property pair are automatically excluded and
their weight is redistributed to the remaining factors.
"""

import functools
import os

import pandas as pd

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 100)
# ---------------------------------------------------------------------------
WEIGHTS: dict[str, float] = {
    "size":           25.0,   # SF (commercial) or unit count (multifamily)
    "location":       25.0,   # Haversine distance, penalised by metro step
    "bldg_class":     15.0,   # Building class match (A/B/C)
    "year_built":     15.0,   # Year built difference
    "year_renovated": 10.0,   # Year renovated difference (skipped if NULL)
    "submarket":       5.0,   # Same submarket bonus
    "sale_recency":    5.0,   # How recently the comp transacted
}

# ---------------------------------------------------------------------------
# Penalty rates (points lost per unit of difference)
# ---------------------------------------------------------------------------
SIZE_PENALTY_PER_PCT: float = 1.0       # 1 pt per 1% symmetric size variance
YEAR_PENALTY_PER_YEAR: float = 2.0      # 2 pts per year difference (built & renovated)
CLASS_PENALTY_MISMATCH: float = 20.0    # Points lost when building class differs
RECENCY_PENALTY_PER_YEAR: float = 10.0  # 10 pts per year beyond the 1-yr grace window
RECENCY_GRACE_YEARS: float = 1.0        # No penalty within this many years of as_of_date

# ---------------------------------------------------------------------------
# Hard-filter parameters
# ---------------------------------------------------------------------------
MAX_RADIUS_MILES: float = 50.0          # Candidates must be within this radius
SIZE_TOLERANCE: float = 0.60            # Symmetric size variance must be <= 60%
DEFAULT_DISTANCE_STEP: float = 0.25     # Fallback step for unknown metros

# ---------------------------------------------------------------------------
# Property type normalisation
# Maps raw source values -> canonical type used for hard-filter matching.
# ---------------------------------------------------------------------------
PROPERTY_TYPE_MAP: dict[str, str] = {
    # Multifamily
    "apartment":        "Multifamily",
    "multi-family":     "Multifamily",
    "multifamily":      "Multifamily",
    # Office
    "office":           "Office",
    # Retail
    "retail":           "Retail",
    "shopping center":  "Retail",
    # Industrial
    "industrial":       "Industrial",
    "flex":             "Industrial",
    # Lodging / Hotel
    "hotel":            "Lodging",
    "hospitality":      "Lodging",
    "lodging":          "Lodging",
    # Healthcare
    "health care":      "Healthcare",
    "healthcare":       "Healthcare",
}

# ---------------------------------------------------------------------------
# Metro distance steps
#
# Different metros have different "how far is too far" intuitions: a quarter-mile
# in a dense downtown is comparable to several miles in a rural market. The
# location score penalises distance in metro-specific steps. The optional CSV
# below maps metro_name -> distance_step_miles; unknown metros fall back to
# DEFAULT_DISTANCE_STEP.
# ---------------------------------------------------------------------------
_STEPS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "metro_distance_steps.csv"
)


@functools.lru_cache(maxsize=1)
def load_metro_steps() -> dict[str, float]:
    """Return a dict mapping metro_name -> distance_step_miles.

    Falls back to DEFAULT_DISTANCE_STEP for unknown metros at scoring time.
    Returns an empty dict if the CSV has not been created yet.
    Cached after first call (the CSV rarely changes within a session).
    """
    if not os.path.exists(_STEPS_CSV):
        return {}
    df = pd.read_csv(_STEPS_CSV)
    return dict(zip(df["metro_name"].str.strip(), df["distance_step_miles"]))
