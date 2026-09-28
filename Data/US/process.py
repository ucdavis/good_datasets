'''
Build GOOD 2.x input data for the United States from EPA and EIA sources.

Run through ``python build.py`` at the repository root. The steps are:

1. ``build_*`` turn the raw tables listed in ``codex.json`` into tidy frames.
2. ``format_*`` turn those into GOOD 2.x components (dictionaries).
3. ``write_*`` save them under ``Data/US/Processed/``.

Units follow GOOD 2.x: MW, MWh, hours, $/MWh for variable costs, $/MW for
overnight capital costs, $/MW-yr for fixed O&M, kg/MWh for emission rates and
Btu/kWh for heat rates. Every value this module introduces rather than reads
from a source table is in ``PARAMETERS`` with its source.

Hour columns are always selected by name. Selecting them by position broke
twice: the onshore wind CSV has an extra index column, so every day started
with "Day Of Month" (25 values per day), and the hydro table lost its first
hour. Positional selection also shifts under pandas 3, where
``groupby().apply()`` no longer passes grouping columns.
'''

import json
import os
import re
import time

import numpy as np
import pandas as pd

FORMAT_VERSION = 2

HOURS = 8760
LB_TO_KG = 0.453592

STATE_CODES = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District Of Columbia': 'DC',
    'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY',
    'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI',
    'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE',
    'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
    'Pennsylvania': 'PA', 'Puerto Rico': 'PR', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virgin Islands': 'VI', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY',
}
'''State name to USPS code. Missouri was "MO " (trailing space) before 2.0, so its RPS matched nothing.'''

PARAMETERS = {
    'cost_vintage': {
        'value': '2023',
        'source': 'EPA Platform v6 Table 4-16 vintage used for new wind and solar costs (2016$).',
    },
    'renewable_capital_charge_rate': {
        'value': 0.0978,
        'source': 'EPA Platform v6 Table 10-9, blended real capital charge rate for wind, solar and '
                  'geothermal, 2023.',
    },
    'real_discount_rate': {
        'value': 0.0746,
        'source': 'Real rate at which a 20-year capital recovery factor equals the 9.78% charge rate '
                  '(EPA Platform v6 Tables 10-9 and 10-12).',
    },
    'capacity_credit': {
        'value': {'wind': 0.15, 'solar': 0.10, 'battery': 1.0, 'pumped hydro': 1.0},
        'source': 'Wind and solar are PLACEHOLDERS: EPA Platform v6 Tables 4-21 and 4-32 give 0-90% '
                  'ranges that fall with penetration. Storage: Table 4-35 (100%).',
    },
    'new_battery': {
        'value': {'capex_cost': 1_977_000.0, 'fom_cost': 35_000.0, 'operating_cost': 7.1,
                  'duration': 4.0, 'round_trip_efficiency': 0.85, 'lifetime': 15.0},
        'source': 'EPA Platform v6 Table 4-35 (AEO 2018), 2023 vintage, 2016$: $1,977/kW for a 4-hour '
                  'system, $35/kW-yr FOM, $7.1/MWh VOM, 85% efficiency. Lifetime is an ASSUMPTION.',
    },
    'existing_battery': {
        'value': {'default_duration': 2.0, 'round_trip_efficiency': 0.85},
        'source': 'Duration per plant from EIA-860 2021 Schedule 3-4 (energy / nameplate power); '
                  'the default is the fleet median for plants without a match. Efficiency as for new '
                  'batteries.',
    },
    'pumped_hydro': {
        'value': {'duration': 10.0, 'round_trip_efficiency': 0.80},
        'source': 'PLACEHOLDER: EIA-860 does not report pumped-storage energy capacity.',
    },
    'hydro': {
        'value': {'energy_budget_window': 24, 'capacity_factor_without_profile': 0.4},
        'source': 'Monthly capacity factors from capacity_factor.csv become a daily energy budget; '
                  'regions without that table use a flat 0.4 (ASSUMPTION).',
    },
    'transmission_loss': {
        'value': {'WECC': 0.028, 'other': 0.024},
        'source': 'EPA Platform v6 section 3.3.4.',
    },
    'fuel_cost_targets': {
        'value': {'Coal': 23.0, 'Oil': 32.0},
        'source': 'Calibration targets for mean fuel plus variable O&M cost ($/MWh) carried over from '
                  'GOOD 1.x; source not recorded. Nuclear is no longer rescaled (see '
                  'assign_fuel_costs).',
    },
    'fill_seed': {
        'value': 0,
        'source': 'Seed for sampling fill-in fuel and VOM costs, so builds are reproducible.',
    },
}

RENEWABLE_PLANT_TYPES = {
    'Onshore Wind', 'Offshore Wind', 'Solar PV', 'Solar Thermal', 'Geothermal', 'Biomass', 'Landfill Gas',
}
'''Plant types flagged ``renewable``. Offshore wind, biomass and landfill gas were missing before 2.0.
RPS eligibility differs by state; review for state-specific studies.'''

MUST_RUN_PLANT_TYPES = {'Geothermal'}

TYPES = {
    'Hydro': 'hydro',
    'Geothermal': 'geothermal',
    'Onshore Wind': 'wind',
    'Offshore Wind': 'wind',
    'Solar PV': 'solar',
    'Solar Thermal': 'solar',
    'Pumped Storage': 'storage',
    'Energy Storage': 'storage',
    'New Battery Storage': 'storage',
}

FUELS = {
    'Coal': 'coal', 'IMPORT': 'import', 'Oil': 'oil', 'NaturalGas': 'natural gas', 'Hydro': 'hydro',
    'Non-Fossil': 'non-fossil', 'MSW': 'waste', 'Geothermal': 'geothermal', 'Wind': 'wind',
    'Fwaste': 'waste', 'Biomass': 'biomass', 'LF Gas': 'landfill gas', 'Pumps': 'pump hydro',
    'Solar': 'solar', 'Tires': 'tires', 'EnerStor': 'battery', 'Nuclear': 'nuclear',
}

EMISSIONS = {'nox': 'PLNOXRTA', 'so2': 'PLSO2RTA', 'co2': 'PLCO2RTA', 'ch4': 'PLCH4RTA', 'n2o': 'PLN2ORTA',
             'pm': 'PLPMTRO'}


def parameter(name):

    return PARAMETERS[name]['value']


def _print(string, disp=True):

    if disp:

        print(string, flush=True)


def _numeric(series):
    '''Numbers stored as text with thousands separators and padding, such as " 34,807 ".'''

    return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce')


def _hour_columns(frame, prefix):
    '''Hour columns in numeric order, matched by name ("Hour01", "Hour 1", "Hour_0", ...).'''

    pattern = re.compile(rf'^{re.escape(prefix)}\s*_?(\d+)$')
    columns = [(int(m.group(1)), c) for c in frame.columns if (m := pattern.match(str(c)))]

    return [c for _, c in sorted(columns)]


def profile_key(region, kind, resource_class=None):
    '''The one place profile keys are made; assets and profiles must agree.'''

    if kind in ('load', 'hydro'):

        return f'{region}:{kind}'

    return f'{region}:{kind}:{resource_class}'


# ============================================================================ build

def build_installed_assets(data, rng=None, verbose=False):

    t0 = time.time()
    rng = rng if rng is not None else np.random.default_rng(parameter('fill_seed'))

    plants = merging_data(data['plants_2021'], data['impacts_parsed'])
    plants = assign_fuel_costs(plants, rng)

    for fuel, target in parameter('fuel_cost_targets').items():

        plants = rescale_mean_cost(plants, fuel, target)

    plants = assign_em_rates(plants, data['plants_2020'])

    _print(f'Installed assets built: {time.time() - t0:.1f} s', disp=verbose)

    return plants


def build_optional_assets(data, verbose=False):
    '''Candidate wind and solar sites by region, state, resource class and cost class ($/kW).'''

    t0 = time.time()

    capacity = {
        'wind': _fill_region_state(data['onshore_wind_capacity']),
        'solar': _fill_region_state(data['solar_regional_capacity']),
    }

    adders = {
        'wind': _fill_region_state(data['onshore_wind_capital_cost']),
        'solar': _fill_region_state(data['solar_capital_cost']),
    }

    base, fom = base_capital_costs(data['unit_cost'], data['regional_cost'])

    cost = {kind: add_base_cost(adders[kind], base[kind]) for kind in ('wind', 'solar')}

    _print(f'Optional assets built: {time.time() - t0:.1f} s', disp=verbose)

    return {'capacity': capacity, 'cost': cost, 'fom': fom}


def build_lines(data, verbose=False):

    t0 = time.time()

    table = data['transmission']

    capacity = table.pivot(index='From', columns='To', values='Capacity TTC (MW)').fillna(0)
    cost = table.pivot(index='From', columns='To', values='Transmission Tariff (2016 mills/kWh)').fillna(0)

    _print(f'Lines built: {time.time() - t0:.1f} s', disp=verbose)

    return {'capacity': capacity, 'cost': cost}


def build_profiles(data, verbose=False):

    t0 = time.time()

    profiles = {
        'wind': resource_profiles(data['onshore_wind_generation_profile']),
        'solar': resource_profiles(data['solar_generation_profile']),
        'load': load_profiles(data['load_profile']),
        'hydro': hydro_profiles(data['hydro']),
    }

    _print(f'Profiles built: {time.time() - t0:.1f} s', disp=verbose)

    return profiles


def build_policies(data, verbose=False):

    t0 = time.time()

    policies = {'rps': build_rps(data['rps'])}

    _print(f'Policies built: {time.time() - t0:.1f} s', disp=verbose)

    return policies


def build_storage_durations(data, verbose=False):
    '''Battery duration (h) by ORIS plant code from EIA-860 Schedule 3-4.'''

    t0 = time.time()

    table = data['energy_storage']
    table = table[table['Technology'] == 'Batteries'].copy()
    table['energy'] = _numeric(table['Nameplate Energy Capacity (MWh)'])
    table['power'] = _numeric(table['Nameplate Capacity (MW)'])
    table = table.dropna(subset=['energy', 'power'])
    table = table[table['power'] > 0]

    totals = table.groupby('Plant Code')[['energy', 'power']].sum()
    durations = (totals['energy'] / totals['power']).to_dict()

    _print(f'Storage durations built: {time.time() - t0:.1f} s', disp=verbose)

    return {int(k): float(v) for k, v in durations.items()}


# ============================================================================ profiles

def resource_profiles(table):
    '''
    Wind or solar profiles (per-unit) by region, state and resource class.

    Columns are found by name, and rows are put in calendar order before
    flattening, so each profile has exactly 8,760 values.
    '''

    hours = _hour_columns(table, 'Hour')
    day = 'Day Of Month' if 'Day Of Month' in table.columns else 'Day of Month'

    rows = []

    for (region, state, resource_class), group in table.groupby(['Region Name', 'State Name', 'Resource Class']):

        group = group.sort_values(['Month', day])
        values = group[hours].apply(_numeric).to_numpy(dtype=float).ravel() / 1000  # kWh/MW -> MWh/MW

        rows.append({'Region Name': region, 'State Name': state, 'Resource Class': int(resource_class),
                     'Profile': values})

    return pd.DataFrame(rows)


def load_profiles(table):
    '''Hourly demand (MW) by region.'''

    hours = _hour_columns(table, 'Hour')
    rows = []

    for region, group in table.groupby('Region'):

        group = group.assign(Month=_numeric(group['Month']), Day=_numeric(group['Day'])).sort_values(['Month', 'Day'])
        values = group[hours].apply(_numeric).to_numpy(dtype=float).ravel()

        rows.append({'Region': region, 'Profile': values})

    return pd.DataFrame(rows)


def hydro_profiles(table):
    '''Hourly hydro capacity factors by region (Hour_0 to Hour_8759).'''

    table = table[table['PlantType'] == 'Hydro']
    hours = _hour_columns(table, 'Hour')

    return pd.DataFrame([
        {'Region': row['Region'], 'Profile': row[hours].to_numpy(dtype=float)}
        for _, row in table.iterrows()
    ])


def _check_length(key, values):

    if len(values) != HOURS:

        raise ValueError(f'profile {key!r} has {len(values)} values, expected {HOURS}')

    return values


def potential_by_class(capacity):
    '''Total candidate capacity (MW) by (region, resource class), summed over states and cost classes.'''

    classes = _cost_class_columns(capacity)
    total = capacity[classes].apply(_numeric).sum(axis=1)

    return total.groupby([capacity['IPM Region'], capacity['Resource Class'].astype(int)]).sum().to_dict()


def format_profiles(profiles, potential, verbose=False):
    '''
    Profiles keyed by ``profile_key``, and the peak demand of each region (MW).

    Wind and solar get one profile per resource class, averaged over the
    states a region spans (weighted by candidate capacity), and a ``mean``
    profile for existing plants, weighted the same way across classes.
    Existing plants' resource classes are not in the source data; before
    2.0 they all took whichever class appeared first.
    '''

    t0 = time.time()

    data = {}
    peak = {}

    for kind in ('wind', 'solar'):

        table = profiles[kind]
        by_region = {}

        for (region, resource_class), group in table.groupby(['Region Name', 'Resource Class']):

            stacked = np.vstack(group['Profile'].to_list())
            profile = _check_length(profile_key(region, kind, resource_class), stacked.mean(axis=0))
            weight = potential[kind].get((region, int(resource_class)), 0.0)

            data[profile_key(region, kind, resource_class)] = profile
            by_region.setdefault(region, []).append((profile, weight))

        for region, entries in by_region.items():

            weights = np.array([w for _, w in entries])
            weights = weights if weights.sum() > 0 else np.ones(len(entries))
            stacked = np.vstack([p for p, _ in entries])

            # Elementwise multiply and sum rather than a matrix product: BLAS libraries
            # differ between platforms, and their rounding differences changed results.
            data[profile_key(region, kind, 'mean')] = (stacked * weights[:, None]).sum(axis=0) / weights.sum()

    for _, row in profiles['load'].iterrows():

        values = _check_length(profile_key(row['Region'], 'load'), np.asarray(row['Profile'], dtype=float))
        peak[row['Region']] = float(values.max())
        data[profile_key(row['Region'], 'load')] = values / values.max()

    for _, row in profiles['hydro'].iterrows():

        values = _check_length(profile_key(row['Region'], 'hydro'), np.asarray(row['Profile'], dtype=float))
        data[profile_key(row['Region'], 'hydro')] = np.clip(values, 0.0, 1.0)

    _print(f'Profiles formatted: {time.time() - t0:.1f} s', disp=verbose)

    return data, peak


# ============================================================================ assets

def _emissions(row):

    return {key: float(np.nan_to_num(row[column])) * LB_TO_KG for key, column in EMISSIONS.items()}


def _storage_fields(row, durations):

    fuel = FUELS.get(row['FuelType'], '')

    if fuel == 'pump hydro' or row['PlantType'] == 'Pumped Storage':

        spec = parameter('pumped_hydro')
        duration = spec['duration']

    else:

        spec = parameter('existing_battery')
        duration = durations.get(_oris(row), spec['default_duration'])

    one_way = float(np.sqrt(spec['round_trip_efficiency']))

    return {
        'duration': float(duration),
        'charge_efficiency': one_way,
        'discharge_efficiency': one_way,
        'capacity_credit': parameter('capacity_credit')['pumped hydro' if fuel == 'pump hydro' else 'battery'],
    }


def _oris(row):

    value = row.get('ORISPL')

    return int(value) if value is not None and not pd.isna(value) else None


def format_installed_assets(assets, peak, profiles, durations, verbose=False):
    '''Existing plants and regional base loads as GOOD 2.x components.'''

    t0 = time.time()

    credit = parameter('capacity_credit')
    hydro = parameter('hydro')

    data = {}

    for idx, row in assets.iterrows():

        plant_type = row['PlantType']
        kind = TYPES.get(plant_type, 'generator')
        region = row['RegionName']

        asset = {
            'oris_code': _oris(row) if _oris(row) is not None else 'none',
            'egrid_id': row['UniqueID'] if not pd.isna(row['UniqueID']) else 'none',
            'type': kind,
            'plant_type': plant_type,
            'fuel': FUELS[row['FuelType']],
            'region': region,
            'jurisdiction': STATE_CODES[row['StateName']],
            'nerc': row['NERC'],
            'utility': row['UTLSRVNM'],
            'x': row['LON'],
            'y': row['LAT'],
            'installed_capacity': float(row['Capacity']),
            'combinable': True,
            'renewable': plant_type in RENEWABLE_PLANT_TYPES,
            'operating_cost': float(np.nan_to_num(row['Fuel_VOM_Cost'])),
            'heat_rate': float(np.nan_to_num(row['HeatRate'])),
            **_emissions(row),
        }

        if kind == 'storage':

            asset.update(_class='Store', **_storage_fields(row, durations))

        elif kind in ('wind', 'solar'):

            asset.update(_class='Producer', profile=profile_key(region, kind, 'mean'),
                         capacity_credit=credit[kind])

        elif kind == 'hydro':

            key = profile_key(region, 'hydro')

            if key in profiles:

                asset.update(_class='Producer', profile=key, capacity_factor=1.0,
                             energy_budget_window=hydro['energy_budget_window'])

            else:

                asset.update(_class='Producer', profile=None, capacity_factor=hydro['capacity_factor_without_profile'])

        else:

            asset.update(_class='Producer', profile=None, dispatchable=plant_type not in MUST_RUN_PLANT_TYPES)

        data[f'installed_{idx}'] = asset

    for region in sorted(set(assets['RegionName'])):

        if region not in peak:

            continue

        data[f'base_load_{region}'] = {
            '_class': 'Load',
            'type': 'load',
            'profile': profile_key(region, 'load'),
            'region': region,
            'jurisdiction': None,
            'installed_capacity': peak[region],  # MW; the profile peaks at 1.0
            'combinable': False,
        }

    _print(f'Installed assets formatted: {time.time() - t0:.1f} s', disp=verbose)

    return data


def format_optional_assets(capex, profiles, regions, verbose=False):
    '''Candidate wind and solar sites and one battery option per region.'''

    t0 = time.time()

    credit = parameter('capacity_credit')
    charge_rate = parameter('renewable_capital_charge_rate')

    data = {}
    skipped = 0
    k = -1

    for kind in ('wind', 'solar'):

        capacity = capex['capacity'][kind]
        cost = capex['cost'][kind]
        classes = _cost_class_columns(capacity)

        cost_by_key = {
            (r['IPM Region'], r['State'], int(r['Resource Class'])): r for _, r in cost.iterrows()
        }

        for _, row in capacity.iterrows():

            key = (row['IPM Region'], row['State'], int(row['Resource Class']))

            if key not in cost_by_key:

                continue

            profile = profile_key(key[0], kind, key[2])

            if profile not in profiles:

                skipped += 1

                continue

            k += 1

            for cost_class in classes:

                mw = _numeric(pd.Series([row[cost_class]])).iloc[0]
                per_kw = cost_by_key[key][cost_class]

                if pd.isna(mw) or pd.isna(per_kw) or mw <= 0:

                    continue

                data[f'optional_{k}_{cost_class}'] = {
                    '_class': 'Producer',
                    'type': kind,
                    'fuel': kind,
                    'profile': profile,
                    'region': key[0],
                    'jurisdiction': key[1],
                    'resource_class': key[2],
                    'cost_class': int(cost_class),
                    'installed_capacity': 0.0,
                    'combinable': False,
                    'renewable': True,
                    'capex_capacity': float(mw),
                    'capex_cost': float(per_kw) * 1e3,  # $/kW -> $/MW
                    'fom_cost': capex['fom'][kind] * 1e3,  # $/kW-yr -> $/MW-yr
                    'capital_charge_rate': charge_rate,
                    'capacity_credit': credit[kind],
                    'operating_cost': 0.0,
                }

    battery = parameter('new_battery')
    one_way = float(np.sqrt(battery['round_trip_efficiency']))

    for idx, region in enumerate(sorted(regions)):

        data[f'optional_storage_{idx}'] = {
            '_class': 'Store',
            'type': 'battery',
            'fuel': 'battery',
            'region': region,
            'jurisdiction': None,
            'installed_capacity': 0.0,
            'combinable': False,
            'renewable': False,
            'duration': battery['duration'],
            'charge_efficiency': one_way,
            'discharge_efficiency': one_way,
            'capex_capacity': float('inf'),
            'capex_cost': battery['capex_cost'],
            'fom_cost': battery['fom_cost'],
            'operating_cost': battery['operating_cost'],
            'lifetime': battery['lifetime'],
            'discount_rate': parameter('real_discount_rate'),
            'capacity_credit': credit['battery'],
        }

    _print(f'Optional assets formatted: {time.time() - t0:.1f} s ({skipped} site groups without a profile skipped)',
           disp=verbose)

    return data


def format_lines(transmission, verbose=False):
    '''Directed lines (MW, $/MWh) with EPA losses; the two directions of a path share a corridor.'''

    t0 = time.time()

    loss = parameter('transmission_loss')
    capacity, cost = transmission['capacity'], transmission['cost']

    pairs = []

    for source, row in capacity.iterrows():

        for target, mw in row.items():

            if source not in cost.index or target not in cost.columns:

                continue

            if 'CN_' in source or 'CN_' in target or mw == 0:

                continue

            pairs.append((source, target, float(mw), float(cost.loc[source, target])))

    present = {(s, t) for s, t, _, _ in pairs}
    links = {}

    for k, (source, target, mw, tariff) in enumerate(pairs):

        interconnect = 'WECC' if source.startswith('WEC') else 'other'

        line = {
            'source': source,
            'target': target,
            'type': 'line',
            '_class': 'Transmission',
            'installed_capacity': mw,
            'operating_cost': tariff,  # 2016 mills/kWh = $/MWh
            'efficiency': 1.0 - loss[interconnect],
        }

        if (target, source) in present:

            line['corridor'] = '|'.join(sorted((source, target)))

        links[f'line_{k}'] = line

    _print(f'Lines formatted: {time.time() - t0:.1f} s', disp=verbose)

    return links


def format_policies(policies, jurisdictions, verbose=False):
    '''State renewable portfolio standards as GOOD 2.x attribute filters.'''

    t0 = time.time()

    data = {}
    dropped = []

    for state, ratio in policies['rps'].items():

        if state not in jurisdictions:

            dropped.append(state)

            continue

        data[f'rps_{state}'] = {
            'type': 'rps',
            '_class': 'Portfolio_Standard',
            'ratio': float(ratio),
            'include': {'renewable': True, '_class': 'Producer', 'jurisdiction': state},
            'exclude': {'renewable': {'not': True}, '_class': 'Producer', 'jurisdiction': state},
        }

    _print(f'Policies formatted: {time.time() - t0:.1f} s (no assets for: {", ".join(dropped) or "none"})',
           disp=verbose)

    return data


# ============================================================================ write

def _rounded(value, digits):

    if isinstance(value, dict):

        return {k: _rounded(v, digits) for k, v in value.items()}

    if isinstance(value, (list, tuple, np.ndarray)):

        return [_rounded(v, digits) for v in value]

    if isinstance(value, (float, np.floating)):

        value = float(value)

        return value if not np.isfinite(value) else float(f'{value:.{digits}g}')

    if isinstance(value, np.integer):

        return int(value)

    if isinstance(value, np.bool_):

        return bool(value)

    return value


def _write(data, path):

    with open(path, 'w') as file:

        json.dump(data, file, indent=1, sort_keys=True)
        file.write('\n')


def write_assets(data, output_path='', filename='assets.json', verbose=False):

    _write(_rounded(data, 10), os.path.join(output_path, filename))
    _print(f'Assets written: {len(data)}', disp=verbose)


def write_lines(data, output_path='', filename='lines.json', verbose=False):

    _write(_rounded(data, 10), os.path.join(output_path, filename))
    _print(f'Lines written: {len(data)}', disp=verbose)


def write_profiles(data, output_path='', foldername='profiles', verbose=False):
    '''One file per profile; values rounded to 1e-5 per-unit.'''

    directory = os.path.join(output_path, foldername)
    os.makedirs(directory, exist_ok=True)

    for name in os.listdir(directory):

        if name.endswith('.json'):

            os.remove(os.path.join(directory, name))

    for key, values in data.items():

        with open(os.path.join(directory, f'{key}.json'), 'w') as file:

            json.dump([round(float(v), 5) + 0.0 for v in values], file)
            file.write('\n')

    _print(f'Profiles written: {len(data)}', disp=verbose)


def write_policies(data, output_path='', filename='policies.json', verbose=False):

    _write(_rounded(data, 10), os.path.join(output_path, filename))
    _print(f'Policies written: {len(data)}', disp=verbose)


def write_metadata(output_path='', filename='metadata.json', **extra):

    metadata = {
        'good_format': FORMAT_VERSION,
        'units': {
            'installed_capacity': 'MW', 'capex_capacity': 'MW', 'duration': 'h',
            'operating_cost': '$/MWh', 'capex_cost': '$/MW', 'fom_cost': '$/MW-yr',
            'heat_rate': 'Btu/kWh', 'emissions (nox, so2, co2, ch4, n2o, pm)': 'kg/MWh',
            'profiles': 'per-unit, 8760 hourly values', 'dollar_year': 2016,
        },
        'parameters': {k: {'value': v['value'], 'source': v['source']} for k, v in PARAMETERS.items()},
        **extra,
    }

    _write(_rounded(metadata, 10), os.path.join(output_path, filename))


# ============================================================================ processing

def build_rps(df, states=None, year=2025):
    '''RPS share by state in ``year``, interpolated between the table's years.'''

    states = df['st'].unique() if states is None else states

    return {state: float(np.interp(year, df.loc[df['st'] == state, 't'], df.loc[df['st'] == state, 'rps_all']))
            for state in states}


def _fill_region_state(table):
    '''Region and state are given only on the first row of each block; fill them down.'''

    table = table.copy()
    table['IPM Region'] = table['IPM Region'].ffill()
    table['State'] = table['State'].ffill()

    return table


def _cost_class_columns(table):

    return [c for c in table.columns if str(c).strip().isdigit()]


def base_capital_costs(unit_cost, regional_cost, vintage=None):
    '''
    Regional base capital cost ($/kW) and fixed O&M ($/kW-yr) for new wind and solar.

    Base cost is EPA Table 4-16 for the chosen vintage times the Table 4-15 regional factor.
    '''

    vintage = vintage or parameter('cost_vintage')
    rows = unit_cost[unit_cost['year'].astype(str).str.contains(vintage)]

    capital = rows[rows['cost'].str.startswith('Capital')].iloc[0]
    fom = rows[rows['cost'].str.startswith('FixedO&M')].iloc[0]

    factors = regional_cost.set_index('ModelRegion')

    base = {
        'wind': (factors['OnshoreWind'] * float(capital['OnshoreWind'])).to_dict(),
        'solar': (factors['SolarPV'] * float(capital['SolarPhotovoltaic'])).to_dict(),
    }

    return base, {'wind': float(fom['OnshoreWind']), 'solar': float(fom['SolarPhotovoltaic'])}


def add_base_cost(adders, base):
    '''Full capital cost ($/kW): the regional base cost plus each cost class's adder.'''

    table = adders.copy()

    for column in _cost_class_columns(table):

        table[column] = _numeric(table[column]) + table['IPM Region'].map(base)

    return table


def assign_em_rates(input_df, input_df_old):
    '''Fill missing emission rates from similar plants and add PM rates (lb/MWh).'''

    input_df = input_df.copy()
    rates = ['PLCO2RTA', 'PLNOXRTA', 'PLCH4RTA', 'PLN2ORTA', 'PLSO2RTA']

    input_df.loc[
        input_df['FuelType'].isin(['Pumps', 'Hydro', 'Geothermal', 'Non-Fossil', 'EnerStor', 'Nuclear', 'Solar', 'Wind']),
        rates,
    ] = 0

    for r in range(input_df.shape[0]):

        if not np.isnan(input_df.at[r, 'PLCO2RTA']):

            continue

        same = (input_df['FuelType'] == input_df.at[r, 'FuelType']) & (input_df['PlantType'] == input_df.at[r, 'PlantType'])
        size = (input_df['Capacity'] > input_df.at[r, 'Capacity'] * 0.85) & (input_df['Capacity'] < input_df.at[r, 'Capacity'] * 1.15)
        heat = (input_df['HeatRate'] > input_df.at[r, 'HeatRate'] * 0.85) & (input_df['HeatRate'] < input_df.at[r, 'HeatRate'] * 1.15)
        state = input_df['StateName'] == input_df.at[r, 'StateName']
        nerc = input_df['NERC'] == input_df.at[r, 'NERC']

        # Widen the search until a match is found: similar size and heat rate in the
        # state, NERC region or country, then any size in the state, NERC region or country.
        for mask in (same & state & size & heat, same & nerc & size & heat, same & size & heat,
                     same & state, same & nerc, same):

            input_df.loc[r, rates] = input_df.loc[mask, rates].mean()

            if not np.isnan(input_df.at[r, 'PLCO2RTA']):

                break

        if np.isnan(input_df.at[r, 'PLCO2RTA']) and input_df.at[r, 'Capacity'] < 50:

            input_df.loc[r, rates] = 0

    # PM emissions (AP-42 chapter 1 factors, lb/MMBtu, times heat rate)
    input_df['PLPMTRO'] = np.select(
        [(input_df['FuelType'] == 'Coal') & (input_df['PLPRMFL'] == 'RC'),
         (input_df['FuelType'] == 'Coal') & (input_df['PLPRMFL'] != 'RC'),
         (input_df['FuelType'] == 'Oil') & (input_df['PLPRMFL'] != 'WO'),
         (input_df['FuelType'] == 'NaturalGas'),
         (input_df['FuelType'] == 'LF Gas'),
         (input_df['FuelType'] == 'Biomass') & (input_df['PLPRMFL'].isin(['WDL', 'WDS'])),
         (input_df['FuelType'] == 'Oil') & (input_df['PLPRMFL'] == 'WO')],
        [0.08, 0.04, 1.4 / 145, 5.7 / 1020, 0.55 / 96.75, 0.017, 65 / 145],
        default=0)

    input_df['PLPMTRO'] = input_df['PLPMTRO'] * input_df['HeatRate'] / 1000

    # Outliers: take eGRID 2020 values for large plants with implausible CO2 rates.
    for r in range(input_df.shape[0]):

        if input_df.at[r, 'FuelType'] != 'Oil' and input_df.at[r, 'PLCO2RTA'] > 5000:

            if input_df.at[r, 'PLNGENAN'] > 1000:

                old = input_df_old[input_df_old['ORISPL'] == input_df.at[r, 'ORISPL']]

                if not old.empty:

                    input_df.loc[r, rates] = old[rates].values[0]

        if input_df.at[r, 'FuelType'] != 'Oil' and input_df.at[r, 'PLCO2RTA'] > 250000:

            input_df.loc[r, rates] = input_df.loc[r, rates] / 1000

    return input_df


def map_fuel_type(row_input):

    plant_type = row_input['PlantType']
    primary = row_input['PLPRMFL']

    if plant_type == 'Coal Steam':
        return 'Coal'
    if plant_type == 'Nuclear':
        return 'Nuclear'
    if plant_type == 'O/G Steam':
        return 'Oil'
    if plant_type == 'Biomass':
        return 'Biomass'
    if plant_type == 'IMPORT':
        return 'IMPORT'
    if plant_type == 'IGCC' or plant_type == 'Combined Cycle' or primary == 'NaturalGas':
        return 'NaturalGas'
    if plant_type == 'Geothermal':
        return 'Geothermal'
    if plant_type == 'Combustion Turbine' and primary == 'NG':
        return 'NaturalGas'
    if plant_type == 'Combustion Turbine' and primary == 'DFO':
        return 'Oil'
    if plant_type == 'Combustion Turbine' and primary == 'WDS':
        return 'Biomass'

    return row_input['FuelType']


def rescale_mean_cost(df, fuel, target_mean):
    '''Scale one fuel's dispatch costs so their mean equals ``target_mean`` ($/MWh).'''

    df = df.copy()
    selected = df['FuelType'] == fuel
    current = df.loc[selected, 'Fuel_VOM_Cost'].mean()

    if selected.any() and current > 0:

        df.loc[selected, 'Fuel_VOM_Cost'] = df.loc[selected, 'Fuel_VOM_Cost'] * target_mean / current

    return df


def _fill_low_costs(frame, column, threshold_width, rng):
    '''
    Replace implausibly low costs with a cost sampled from plants of the same
    fuel, searching the region, then the state, NERC region and country.
    '''

    for idx, row in frame.iterrows():

        fuel_type = row['FuelType']
        costs = frame.loc[(frame['FuelType'] == fuel_type) & (frame[column] > 1), column]
        threshold = max(costs.mean() - threshold_width * costs.std(), 0)

        if not row[column] <= threshold:

            continue

        same_fuel = frame['FuelType'] == fuel_type

        for scope in ('RegionName', 'StateName', 'NERC', None):

            mask = same_fuel & (frame[column] > threshold)

            if scope is not None:

                mask &= frame[scope] == row[scope]

            candidates = frame.loc[mask, column]

            if not candidates.empty:

                frame.at[idx, column] = rng.choice(candidates.to_numpy())

            if frame.at[idx, column] > threshold:

                break

    return frame


def assign_fuel_costs(input_df, rng):
    '''
    Dispatch cost ($/MWh) = fuel cost + variable O&M, from NEEDS totals.

    Fixed O&M is no longer added to nuclear dispatch cost: a fixed cost does
    not change with output, and including it raised nuclear's marginal cost
    and moved it in the dispatch order. Missing costs are filled by sampling
    with ``rng`` (seeded), so builds are reproducible.
    '''

    selected_columns = [
        'UniqueID', 'ORISPL', 'PLNGENAN', 'RegionName', 'StateName', 'CountyName', 'NERC',
        'PlantType', 'FuelType', 'FossilUnit', 'Capacity', 'Firing', 'Bottom',
        'EMFControls', 'FOMCost', 'FuelUseTotal', 'FuelCostTotal', 'VOMCostTotal',
        'UTLSRVNM', 'SUBRGN', 'FIPSST', 'FIPSCNTY', 'LAT', 'LON', 'PLPRMFL', 'PLNOXRTA',
        'PLSO2RTA', 'PLCO2RTA', 'PLCH4RTA', 'PLN2ORTA', 'HeatRate',
    ]

    frame = input_df[selected_columns].copy()

    frame['FuelCost[$/MWh]'] = (frame['FuelCostTotal'] / (frame['FuelUseTotal'] + 1)) * frame['HeatRate'] / 1000

    direct_vom = ['Solar', 'Solar PV', 'Wind', 'Hydro', 'Energy Storage', 'Solar Thermal', 'New Battery Storage',
                  'Offshore Wind']

    frame['VOMCost[$/MWh]'] = np.where(
        frame['PlantType'].isin(direct_vom),
        frame['VOMCostTotal'],
        (frame['VOMCostTotal'] / (frame['FuelUseTotal'] + 1)) * frame['HeatRate'] / 1000,
    )

    for idx in frame.index[frame['NERC'].isna()]:

        mode = frame.loc[frame['RegionName'] == frame.at[idx, 'RegionName'], 'NERC'].mode()

        if not mode.empty:

            frame.at[idx, 'NERC'] = mode.iloc[0]

    frame['FuelType'] = frame.apply(map_fuel_type, axis=1)
    frame = frame[~(frame['FuelType'].isna() & (frame['PlantType'] != 'IMPORT'))].reset_index(drop=True)

    frame = _fill_low_costs(frame, 'FuelCost[$/MWh]', 0.5, rng)
    frame = _fill_low_costs(frame, 'VOMCost[$/MWh]', 2.0, rng)

    frame['Fuel_VOM_Cost'] = frame['FuelCost[$/MWh]'] + frame['VOMCost[$/MWh]']

    return frame


def merging_data(plant, parsed):

    parsed = parsed.copy()
    parsed['ORISPL'] = parsed['ORISCode']
    merged = pd.merge(parsed, plant, how='left', on='ORISPL')

    return merged.dropna(how='all')
