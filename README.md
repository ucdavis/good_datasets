# GOOD datasets

Builds input data for the [GOOD model](https://github.com/ucdavis/good_model)
(GOOD 2.x format) for the United States from EPA and EIA sources: existing
plants, candidate wind, solar and battery sites, inter-regional transmission,
hourly load, wind, solar and hydro profiles, and state renewable portfolio
standards, organized by EPA IPM region.

## Build

```bash
pip install -r requirements.txt
python build.py
```

This writes the processed files to `Data/US/Processed/` and GOOD graphs to
`Outputs/`, taking about two minutes. Every graph is validated against GOOD's
input schema before it is written.

| Output | Contents |
|---|---|
| `Data/US/Processed/assets.json` | every asset, keyed by handle |
| `Data/US/Processed/lines.json` | transmission lines |
| `Data/US/Processed/profiles/<key>.json` | 8,760 hourly per-unit values per profile |
| `Data/US/Processed/policies.json` | state RPS policies as GOOD attribute filters |
| `Data/US/Processed/metadata.json` | format version, units, and every assumed parameter with its source |
| `Outputs/US.json.gz` | the whole country as one GOOD graph |
| `Outputs/<region>.json.gz` | ERC, FRCC, MIS, NENG, NY, PJM, SPP, S and WEC subgraphs |
| `Outputs/California.json.gz` | the six California regions, aggregated with `good.aggregate` (ratio 0.1) |

`Outputs/` is not tracked; rebuild it with `python build.py`.

```python
import good

graph = good.graph.graph_from_json("Outputs/WEC.json.gz")
policies = good.utilities.read_json("Outputs/policies.json")
network = good.Network(steps=(4608, 4776)).from_graph(graph, policies)
```

## Units

GOOD 2.x units: MW, MWh, hours, $/MWh for variable costs, $/MW for overnight
capital costs, $/MW-yr for fixed O&M, kg/MWh for emission rates and Btu/kWh for
heat rates. Costs are in 2016 dollars, as in the EPA tables. Loads are positive
MW with profiles that peak at 1.0.

## Sources

| File in `Data/US/Raw/` | Source | Used for |
|---|---|---|
| `eGRID2021_data.csv`, `eGRID2020.xlsx` | EPA eGRID 2021 and 2020 | plant location, primary fuel, emission rates |
| `needs_v617_parsed.csv` | EPA NEEDS v6 (parsed) | units, capacity, IPM region, fuel and O&M costs |
| `needs_v6_transmission.csv` | EPA Platform v6 | inter-regional transfer capability and wheeling tariffs |
| `table_2-2.csv` | EPA Platform v6 Table 2-2 | hourly load by region |
| `table_4-38.csv`, `table_4-41.csv` | EPA Platform v6 Tables 4-38, 4-41 | candidate wind and solar capacity by resource and cost class |
| `table_4-39_onshore.csv`, `table_4-43.csv` | EPA Platform v6 Tables 4-39, 4-43 | wind and solar generation profiles |
| `table_4-40.csv`, `table_4-44.csv` | EPA Platform v6 Tables 4-40, 4-44 | wind and solar capital cost adders ($/kW) |
| `table_4-15.xlsx`, `table_4-16.xlsx` | EPA Platform v6 Tables 4-15, 4-16 | regional cost factors, base capital and fixed O&M costs |
| `eia860_2021_energy_storage.csv` | EIA-860 2021, Schedule 3-4 (operable units) | existing battery energy capacity, for durations |
| `capacity_factor.csv` | not recorded | monthly hydro capacity factors by region |
| `rps_fraction.csv` | not recorded (the layout matches NREL ReEDS inputs) | RPS shares by state and year |

EPA Platform v6 is the November 2018 reference case. Please fill in the two
unrecorded sources.

## Assumptions to review

`PARAMETERS` in `Data/US/process.py` holds every value the build introduces
rather than reads from a table, each with its source; `metadata.json` repeats
them. These are placeholders or assumptions:

* Wind and solar capacity credits (0.15 and 0.10). EPA gives 0-90% ranges that
  fall with penetration.
* Pumped hydro duration and efficiency (10 hours, 80%); EIA-860 does not report
  pumped-storage energy.
* A 2-hour default duration for batteries without an EIA-860 match (the 2021
  fleet median), and a 15-year life for new batteries.
* Geothermal runs as must-run at full capacity; its capacity factor is not in
  the source data.
* Mean dispatch costs of coal and oil plants are rescaled to $23 and $32/MWh, a
  calibration carried over from earlier versions whose source is not recorded.
* `renewable` is true for wind, solar, geothermal, biomass and landfill gas. RPS
  eligibility differs by state.

## Tests

```bash
pytest -q                 # unit tests and checks on Data/US/Processed
python build.py --check   # a fresh build must match Data/US/Processed exactly
```

CI runs both. The build is deterministic: missing costs are filled by sampling
with a fixed seed, and hour columns are selected by name, so results do not
depend on the pandas version.

## Changes in GOOD 2.x format (2026)

* Outputs use GOOD 2.x units, classes and declarative policy filters.
* Wind profiles had 25 values per day: `table_4-39_onshore.csv` carries an index
  column, so position-based slicing started each day at "Day Of Month".
  Columns are now selected by name. (Also addressed by data edit in PR #1.)
* Hydro profiles dropped the first hour and were never attached to plants
  (`"REGION:hydro:"` vs `"REGION:hydro"`). They now use the monthly capacity
  factors directly as a daily energy budget.
* New wind and solar capital costs were $/kW divided by 1e6; they are now $/MW,
  with fixed O&M and EPA's capital charge rate.
* Storage has durations (EIA-860 for batteries) and efficiencies; new
  batteries use EPA Table 4-35 costs.
* Transmission lines carry EPA inter-regional losses, and the two directions of
  a path share a corridor.
* Missouri's state code had a trailing space, so its RPS applied to no assets.
* Offshore wind, biomass and landfill gas are flagged renewable.
* Existing wind and solar plants use a capacity-weighted average of their
  region's resource-class profiles instead of whichever class came first.
* Nuclear dispatch cost no longer includes fixed O&M (mean $21.2 to $6.8/MWh).
* Fill-in costs are sampled with a fixed seed.
* The notebooks are replaced by `build.py`; aggregation uses `good.aggregate`.

## Known limitations

Profile file names contain ":" (for example `WEC_BANC:wind:3.json`), which
Windows does not allow, so the repository cannot be checked out on Windows.

## License

GPL-3.0; see `LICENSE`.
