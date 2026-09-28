"""Checks on the committed files in Data/US/Processed."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROCESSED = Path("Data/US/Processed")


@pytest.fixture(scope="module")
def assets():

    return json.loads((PROCESSED / "assets.json").read_text())


@pytest.fixture(scope="module")
def profiles():

    return {p.name[:-5]: json.loads(p.read_text()) for p in (PROCESSED / "profiles").glob("*.json")}


def test_metadata_declares_good_2_units():

    metadata = json.loads((PROCESSED / "metadata.json").read_text())

    assert metadata["good_format"] == 2
    assert metadata["units"]["installed_capacity"] == "MW"


def test_profiles_are_hourly_per_unit_years(profiles):

    for key, values in profiles.items():

        values = np.asarray(values, dtype=float)

        assert values.size == 8760, key
        assert np.isfinite(values).all(), key
        assert values.min() >= 0 and values.max() <= 1 + 1e-9, key


def test_every_profile_reference_resolves(assets, profiles):

    missing = {a["profile"] for a in assets.values() if a.get("profile") and a["profile"] not in profiles}

    assert not missing


def test_jurisdictions_are_state_codes(assets, process):

    codes = set(process.STATE_CODES.values())

    for handle, asset in assets.items():

        if asset["_class"] == "Load" or handle.startswith("optional_storage"):

            continue

        assert asset["jurisdiction"] in codes, (handle, asset["jurisdiction"])


def test_capacities_and_costs_are_in_mw_and_dollars(assets):

    for handle, asset in assets.items():

        assert 0 <= asset["installed_capacity"] < 1e5, handle  # MW, not W

        if asset.get("capex_capacity", 0) > 0 and asset["type"] in ("wind", "solar"):

            assert 5e5 < asset["capex_cost"] < 6e6, handle  # $/MW
            assert asset["fom_cost"] > 1e3, handle  # $/MW-yr


def test_existing_capacity_matches_needs():

    needs = pd.read_csv("Data/US/Raw/needs_v617_parsed.csv", low_memory=False)
    assets = json.loads((PROCESSED / "assets.json").read_text())
    installed = sum(a["installed_capacity"] for k, a in assets.items() if k.startswith("installed_"))

    assert installed == pytest.approx(needs["Capacity"].sum(), rel=0.01)


def test_storage_has_duration_and_efficiency(assets):

    stores = [a for a in assets.values() if a["_class"] == "Store"]

    assert stores

    for store in stores:

        assert store["duration"] > 0
        assert 0 < store["charge_efficiency"] <= 1 and 0 < store["discharge_efficiency"] <= 1


def test_nuclear_dispatch_cost_excludes_fixed_costs(assets):

    costs = [a["operating_cost"] for a in assets.values() if a.get("fuel") == "nuclear"]

    assert costs and np.mean(costs) < 15  # $/MWh; fixed O&M was added before 2.0


def test_renewable_flags(assets):

    flags = {a["plant_type"]: a["renewable"] for a in assets.values() if "plant_type" in a}

    assert flags["Offshore Wind"] and flags["Biomass"] and flags["Onshore Wind"]
    assert not flags["Combined Cycle"]


def test_lines_have_losses():

    lines = json.loads((PROCESSED / "lines.json").read_text())

    assert lines
    assert {round(l["efficiency"], 3) for l in lines.values()} <= {0.972, 0.976}


def test_policies_are_declarative_and_match_assets(assets):

    policies = json.loads((PROCESSED / "policies.json").read_text())
    jurisdictions = {a.get("jurisdiction") for a in assets.values()}

    for handle, policy in policies.items():

        assert "inclusion_criteria" not in policy
        assert policy["include"]["jurisdiction"] in jurisdictions, handle


def test_graph_passes_good_validation():

    import good
    import build

    assets, lines, profiles, policies, metadata = build.load_processed(str(PROCESSED))
    graph = build.src.build.build_graph(assets, lines, profiles, good_format=metadata["good_format"])

    good.Network(steps=(0, 8760)).from_graph(graph, policies)
