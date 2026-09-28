"""Unit tests of the processing functions on small synthetic tables."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

DAYS = [dt.date(2021, 1, 1) + dt.timedelta(days=d) for d in range(365)]


def _wind_table(index_column):

    rows = []

    for day in DAYS:

        row = {"Region Name": "R", "State Name": "CA", "Resource Class": 1, "Season": "x",
               "Month": day.month, "Day Of Month": day.day}
        row.update({f"Hour{h:02d}": 1000 * (h - 1) / 23 for h in range(1, 25)})
        rows.append(row)

    table = pd.DataFrame(rows).sample(frac=1.0, random_state=1)  # rows out of order

    if index_column:

        table.insert(0, "Unnamed: 0", range(len(table)))

    return table


@pytest.mark.parametrize("index_column", [False, True])
def test_wind_profiles_have_24_hours_per_day_in_calendar_order(process, index_column):
    """D1: an extra index column and shuffled rows must not change the profile."""

    profiles = process.resource_profiles(_wind_table(index_column))
    values = profiles.loc[0, "Profile"]

    assert len(values) == 8760
    assert values[:24] == pytest.approx(np.arange(24) / 23)
    assert values[-24:] == pytest.approx(np.arange(24) / 23)


def test_hydro_profiles_keep_the_first_hour(process):
    """The hydro table's Hour_0 was dropped before 2.0, leaving 8,759 values."""

    hours = {f"Hour_{h}": (0.5 if h == 0 else 0.3) for h in range(8760)}
    table = pd.DataFrame([{"Region": "R", "PlantType": "Hydro", **hours},
                          {"Region": "R", "PlantType": "Pumped Storage", **hours}])

    profiles = process.hydro_profiles(table)

    assert len(profiles) == 1
    assert len(profiles.loc[0, "Profile"]) == 8760
    assert profiles.loc[0, "Profile"][0] == 0.5


def test_load_profiles_parse_formatted_numbers(process):

    rows = []

    for day in DAYS:

        row = {"Region": "R", "Month": str(day.month), "Day": str(day.day)}
        row.update({f"Hour {h}": f" {30000 + h:,} " for h in range(1, 25)})
        rows.append(row)

    profiles = process.load_profiles(pd.DataFrame(rows))

    assert len(profiles.loc[0, "Profile"]) == 8760
    assert profiles.loc[0, "Profile"][0] == 30001


def test_profile_keys_agree(process):
    """D2: assets and profiles used to disagree ("R:hydro:" vs "R:hydro")."""

    assert process.profile_key("R", "hydro") == "R:hydro"
    assert process.profile_key("R", "load") == "R:load"
    assert process.profile_key("R", "wind", 3) == "R:wind:3"


def test_state_codes_are_clean(process):
    """Missouri was "MO " (trailing space), so its RPS matched no assets."""

    assert process.STATE_CODES["Missouri"] == "MO"
    assert all(len(code) == 2 and code.isupper() for code in process.STATE_CODES.values())


def test_capital_costs_are_base_plus_adder_in_dollars_per_kw(process):
    """D5: base cost times regional factor plus the class adder, in $/kW."""

    unit = pd.DataFrame({
        "year": ["Vintage#2(2023)", "Vintage#2(2023)"],
        "cost": ["Capital(2016$/kW)", "FixedO&M(2016$/kW/yr)"],
        "SolarPhotovoltaic": [1009.0, 10.74],
        "OnshoreWind": [1372.0, 48.72],
    })
    regional = pd.DataFrame({"ModelRegion": ["R"], "OnshoreWind": [1.1], "SolarPV": [0.9]})
    adders = pd.DataFrame({"IPM Region": ["R"], "State": ["CA"], "Resource Class": [1],
                           "1": ["10"], "2": ["1,000"]})

    base, fom = process.base_capital_costs(unit, regional)
    cost = process.add_base_cost(adders, base["wind"])

    assert cost.loc[0, "1"] == pytest.approx(1372 * 1.1 + 10)
    assert cost.loc[0, "2"] == pytest.approx(1372 * 1.1 + 1000)
    assert fom["solar"] == pytest.approx(10.74)


def test_cost_filling_is_reproducible(process):

    frame = pd.DataFrame({
        "FuelType": ["Coal"] * 6,
        "RegionName": ["A", "A", "A", "B", "B", "B"],
        "StateName": ["X"] * 6,
        "NERC": ["N"] * 6,
        "cost": [0.0, 20.0, 22.0, 0.0, 30.0, 31.0],
    })

    first = process._fill_low_costs(frame.copy(), "cost", 0.5, np.random.default_rng(0))
    second = process._fill_low_costs(frame.copy(), "cost", 0.5, np.random.default_rng(0))

    assert first["cost"].tolist() == second["cost"].tolist()
    assert (first["cost"] > 0).all()


def test_lines_get_losses_and_shared_corridors(process):

    capacity = pd.DataFrame([[0, 100], [80, 0]], index=["WEC_A", "WEC_B"], columns=["WEC_A", "WEC_B"])
    cost = pd.DataFrame([[0, 5.0], [5.0, 0]], index=capacity.index, columns=capacity.columns)

    lines = process.format_lines({"capacity": capacity, "cost": cost})

    assert len(lines) == 2
    assert [l["efficiency"] for l in lines.values()] == pytest.approx([0.972, 0.972])
    assert {l["corridor"] for l in lines.values()} == {"WEC_A|WEC_B"}
    assert {l["operating_cost"] for l in lines.values()} == {5.0}
