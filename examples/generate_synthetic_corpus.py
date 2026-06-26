"""Generate a small SYNTHETIC property corpus + expert comp-set labels.

This makes the repository runnable end-to-end with no proprietary data. It
creates fake properties for three "sources" (A/B/C) with deliberate overlaps so
entity resolution has something to merge, then plants a handful of expert
comp-set labels (subject -> ranked comps) drawn from the same fake universe.

Everything is generated with a FIXED numpy seed, so results are reproducible.
Outputs (under ``data/``):

  source_a.parquet, source_b.parquet, source_c.parquet  -- raw per-source rows
  comp_sets.csv                                          -- expert labels (free text)
  corpus.parquet                                         -- unified table (written by demo)

Run:
    python examples/generate_synthetic_corpus.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Make the package importable when run from a fresh checkout.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

SEED = 42
N_PROPERTIES = 600  # distinct real-world buildings before duplication across sources

MAIN_TYPES = ["Multifamily", "Office", "Retail", "Industrial", "Lodging"]
SUBTYPES = {
    "Multifamily": ["Garden", "Mid-Rise", "High-Rise"],
    "Office": ["Low-Rise", "Mid-Rise", "High-Rise"],
    "Retail": ["Strip", "Mall", "Single-Tenant"],
    "Industrial": ["Warehouse", "Flex", "Manufacturing"],
    "Lodging": ["Limited-Service", "Full-Service", "Resort"],
}
BLDG_CLASSES = ["A", "B", "C"]

# A few fake metros with centre coordinates and a per-metro distance step.
METROS = {
    "Metroville":  (40.71, -74.01),
    "Lakeside":    (41.88, -87.63),
    "Bay Harbor":  (37.77, -122.42),
    "Sunbelt City": (29.76, -95.37),
}
STATES = {"Metroville": "NY", "Lakeside": "IL", "Bay Harbor": "CA", "Sunbelt City": "TX"}

STREET_NAMES = [
    "Main Street", "Oak Avenue", "Park Boulevard", "Lake Drive", "Harbor Road",
    "Sunset Lane", "Commerce Court", "Union Plaza", "Market Square", "Cedar Parkway",
]


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def _build_base_properties(rng: np.random.Generator) -> pd.DataFrame:
    """Build the N distinct real-world buildings (pre-duplication)."""
    metros = list(METROS)
    rows = []
    for i in range(N_PROPERTIES):
        metro = metros[rng.integers(len(metros))]
        clat, clon = METROS[metro]
        lat = clat + float(rng.normal(0, 0.15))
        lon = clon + float(rng.normal(0, 0.15))
        main_type = MAIN_TYPES[rng.integers(len(MAIN_TYPES))]
        subtype = SUBTYPES[main_type][rng.integers(len(SUBTYPES[main_type]))]
        is_mf = main_type == "Multifamily"

        street = STREET_NAMES[rng.integers(len(STREET_NAMES))]
        number = int(rng.integers(50, 9999))
        state = STATES[metro]
        zipc = int(rng.integers(10000, 99999))
        address = f"{number} {street}, {metro}, {state} {zipc}"
        name = f"{street.split()[0]} {subtype} {main_type[:4]} {i}"

        year_built = int(rng.integers(1955, 2022))
        renovated = year_built + int(rng.integers(0, 40))
        rows.append({
            "base_id": i,
            "property_name": name,
            "full_address": address,
            "lat": lat,
            "lon": lon,
            "metro": metro,
            "submarket": f"{metro}-{rng.integers(1, 6)}",
            "main_type": main_type,
            "property_subtype": subtype,
            "bldg_class": BLDG_CLASSES[rng.integers(len(BLDG_CLASSES))],
            "sqft": None if is_mf else float(int(rng.integers(10_000, 800_000))),
            "num_units": int(rng.integers(20, 600)) if is_mf else None,
            "year_built": year_built,
            "year_renovated": renovated if renovated > year_built else None,
            "num_stories": int(rng.integers(1, 40)),
            "occupancy": round(float(rng.uniform(0.55, 1.0)), 3),
            "land_area_ac": round(float(rng.uniform(0.2, 25.0)), 2),
            "parking_ratio": round(float(rng.uniform(0.5, 5.0)), 2),
            "avg_unit_size": round(float(rng.uniform(600, 1400)), 0) if is_mf else None,
            "last_sale_date": (
                pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(rng.integers(0, 2800)))
            ).date(),
        })
    return pd.DataFrame(rows)


def _emit_source(base: pd.DataFrame, prefix: str, source_label: str,
                 rng: np.random.Generator, coverage: float,
                 jitter_m: float, drop_fields: list[str]) -> pd.DataFrame:
    """Create one source's view of a coverage-fraction subset of the buildings.

    Coordinates are jittered slightly (simulating geocoding error) and some
    fields are dropped to simulate differing source completeness -- exactly the
    conditions entity resolution has to handle.
    """
    n = int(len(base) * coverage)
    idx = rng.choice(len(base), size=n, replace=False)
    sub = base.iloc[idx].copy().reset_index(drop=True)

    # ~111_111 m per degree latitude; jitter both axes.
    deg = jitter_m / 111_111.0
    sub["lat"] = sub["lat"] + rng.normal(0, deg, size=len(sub))
    sub["lon"] = sub["lon"] + rng.normal(0, deg, size=len(sub))

    sub["property_id"] = [f"{prefix}{int(b)}_{i}" for i, b in enumerate(sub["base_id"])]
    sub["source"] = source_label

    for f in drop_fields:
        if f in sub.columns:
            sub[f] = None

    cols = [
        "property_id", "property_name", "full_address", "lat", "lon", "metro",
        "submarket", "main_type", "property_subtype", "bldg_class", "sqft",
        "num_units", "year_built", "year_renovated", "num_stories", "occupancy",
        "land_area_ac", "parking_ratio", "avg_unit_size", "last_sale_date", "source",
    ]
    return sub[cols]


def _plant_labels(base: pd.DataFrame, rng: np.random.Generator,
                  n_subjects: int = 25) -> pd.DataFrame:
    """Plant expert comp sets: for each subject, pick same-type nearby buildings.

    "Expert selection" is simulated as: same main_type, in the same metro,
    closest by size and age -- with a little noise so it isn't a deterministic
    function of the features (mirroring real expert idiosyncrasy).
    """
    rows = []
    subj_idx = rng.choice(len(base), size=n_subjects, replace=False)
    for s, si in enumerate(subj_idx):
        subj = base.iloc[si]
        same = base[(base["main_type"] == subj["main_type"])
                    & (base["metro"] == subj["metro"])
                    & (base["base_id"] != subj["base_id"])]
        if len(same) < 4:
            continue
        # Rank candidates by a noisy size+age closeness, take the top 3-6.
        size = same["sqft"].fillna(same["num_units"]).to_numpy(dtype=float)
        subj_size = subj["sqft"] if pd.notna(subj["sqft"]) else subj["num_units"]
        size_gap = np.abs(np.log((size + 1) / (float(subj_size) + 1)))
        age_gap = np.abs(same["year_built"].to_numpy(dtype=float) - float(subj["year_built"])) / 50.0
        noise = rng.normal(0, 0.3, size=len(same))
        score = size_gap + age_gap + noise
        order = same.iloc[np.argsort(score)]
        n_comps = int(rng.integers(3, 7))
        chosen = order.head(n_comps)

        for rank, (_, comp) in enumerate(chosen.iterrows(), start=1):
            rows.append({
                "set_id": f"SET_{s:03d}",
                "subject_name": subj["property_name"],
                "subject_address": subj["full_address"],
                "subject_city": subj["metro"],
                "subject_state": STATES[subj["metro"]],
                "comp_rank": rank,
                "comp_name": comp["property_name"],
                "comp_address": comp["full_address"],
                "comp_city": comp["metro"],
                "comp_state": STATES[comp["metro"]],
            })
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = _rng()
    base = _build_base_properties(rng)

    # Source A: broadest coverage + most complete physical attributes.
    source_a = _emit_source(base, "A_", "source_a", rng, coverage=0.85,
                            jitter_m=8.0, drop_fields=[])
    # Source B: transactions DB -- keeps sale dates, sparser physical detail.
    source_b = _emit_source(base, "B_", "source_b", rng, coverage=0.55,
                            jitter_m=20.0,
                            drop_fields=["num_stories", "occupancy", "avg_unit_size"])
    # Source C: operating/financial DB -- keeps occupancy, drops some structure.
    source_c = _emit_source(base, "C_", "source_c", rng, coverage=0.45,
                            jitter_m=15.0,
                            drop_fields=["year_renovated", "land_area_ac", "parking_ratio"])

    source_a.to_parquet(os.path.join(DATA_DIR, "source_a.parquet"), index=False)
    source_b.to_parquet(os.path.join(DATA_DIR, "source_b.parquet"), index=False)
    source_c.to_parquet(os.path.join(DATA_DIR, "source_c.parquet"), index=False)

    labels = _plant_labels(base, rng)
    labels.to_csv(os.path.join(DATA_DIR, "comp_sets.csv"), index=False)

    print(f"Wrote synthetic sources to {DATA_DIR}:")
    print(f"  source_a.parquet : {len(source_a):,} rows")
    print(f"  source_b.parquet : {len(source_b):,} rows")
    print(f"  source_c.parquet : {len(source_c):,} rows")
    print(f"  comp_sets.csv    : {labels['set_id'].nunique()} subjects, {len(labels)} comp rows")
    print("\nNext: run `python examples/demo.py`")


if __name__ == "__main__":
    main()
