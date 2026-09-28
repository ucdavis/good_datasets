'''
Build the GOOD datasets from the raw tables in Data/US/Raw.

    python build.py              rebuild Data/US/Processed and the graphs in Outputs/
    python build.py --check      rebuild into a temporary folder and compare with Data/US/Processed
    python build.py --no-graphs  rebuild Data/US/Processed only

Every graph is validated against GOOD's input schema before it is written.
'''

import argparse
import filecmp
import os
import sys
import tempfile
import time
import warnings

import numpy as np

import good
import src

DATA = os.path.join('Data', 'US')
PROCESSED = os.path.join(DATA, 'Processed')
OUTPUTS = 'Outputs'

INTERCONNECT_PREFIXES = {
    'ERC': 'ERC', 'FRCC': 'FRCC', 'MIS': 'MIS_', 'NENG': 'NENG', 'NY': 'NY',
    'PJM': 'PJM', 'SPP': 'SPP', 'S': 'S_', 'WEC': 'WEC',
}

CALIFORNIA = ['WEC_BANC', 'WEC_CALN', 'WEC_LADW', 'WEC_SDGE', 'WECC_IID', 'WECC_SCE']


def process(output, seed, verbose=True):
    '''Run the processing module and write the processed files to ``output``.'''

    module = src.inputs.write_data.load_module(os.path.join(DATA, 'process.py'))
    data = src.inputs.load_data.load(os.path.join(DATA, 'codex.json'), verbose=verbose)

    rng = np.random.default_rng(seed)

    installed = module.build_installed_assets(data, rng=rng, verbose=verbose)
    optional = module.build_optional_assets(data, verbose=verbose)
    lines = module.build_lines(data, verbose=verbose)
    profiles = module.build_profiles(data, verbose=verbose)
    policies = module.build_policies(data, verbose=verbose)
    durations = module.build_storage_durations(data, verbose=verbose)

    potential = {kind: module.potential_by_class(optional['capacity'][kind]) for kind in ('wind', 'solar')}

    profile_data, peak = module.format_profiles(profiles, potential, verbose=verbose)
    installed_data = module.format_installed_assets(installed, peak, profile_data, durations, verbose=verbose)
    regions = sorted({a['region'] for a in installed_data.values()})
    optional_data = module.format_optional_assets(optional, profile_data, regions, verbose=verbose)
    lines_data = module.format_lines(lines, verbose=verbose)

    assets = {**installed_data, **optional_data}
    jurisdictions = {a['jurisdiction'] for a in assets.values() if a.get('jurisdiction')}
    policies_data = module.format_policies(policies, jurisdictions, verbose=verbose)

    os.makedirs(output, exist_ok=True)

    module.write_assets(assets, output_path=output, verbose=verbose)
    module.write_lines(lines_data, output_path=output, verbose=verbose)
    module.write_profiles(profile_data, output_path=output, verbose=verbose)
    module.write_policies(policies_data, output_path=output, verbose=verbose)
    module.write_metadata(output_path=output, seed=seed)


def load_processed(path):

    assets = good.utilities.read_json(os.path.join(path, 'assets.json'))
    lines = good.utilities.read_json(os.path.join(path, 'lines.json'))
    policies = good.utilities.read_json(os.path.join(path, 'policies.json'))
    profiles = good.utilities.read_jsons(os.path.join(path, 'profiles') + os.sep, output='dict')
    metadata = good.utilities.read_json(os.path.join(path, 'metadata.json'))

    # read_jsons keys files by the text before the first "."; profile keys have none.
    profiles = {name[:-5] if name.endswith('.json') else name: values for name, values in profiles.items()}

    return assets, lines, profiles, policies, metadata


def validate(graph, policies, label):
    '''Check a graph against GOOD's input schema; raises GOOD_ValidationError with every problem.'''

    t0 = time.time()
    good.Network(steps=(0, 8760)).from_graph(graph, policies)
    print(f'{label}: valid ({graph.number_of_nodes()} regions, '
          f'{sum(len(d["assets"]) for _, d in graph.nodes(data=True))} assets, {time.time() - t0:.1f} s)', flush=True)


def build_graphs(path, output):
    '''Write the US graph, one graph per interconnect region and an aggregated California example.'''

    assets, lines, profiles, policies, metadata = load_processed(path)
    attributes = {'good_format': metadata['good_format'], 'source': 'ucdavis/good_datasets'}

    graph = src.build.build_graph(assets, lines, profiles, **attributes)
    validate(graph, policies, 'US')

    os.makedirs(output, exist_ok=True)
    good.graph.graph_to_json(graph, os.path.join(output, 'US.json.gz'))

    for name, prefix in INTERCONNECT_PREFIXES.items():

        nodes = [n for n in graph.nodes if n.startswith(prefix)]
        subgraph = good.graph.subgraph(graph, nodes)
        validate(subgraph, policies, name)
        good.graph.graph_to_json(subgraph, os.path.join(output, f'{name}.json.gz'))

    california = good.aggregate.aggregate(good.graph.subgraph(graph, CALIFORNIA), ratio=0.1)
    validate(california, policies, 'California (aggregated)')
    good.graph.graph_to_json(california, os.path.join(output, 'California.json.gz'))

    good.utilities.write_json(policies, os.path.join(output, 'policies.json'), indent=2)


def _close(a, b, absolute):
    '''Equal JSON values, with numbers allowed to differ by rounding in the last written digit.'''

    if isinstance(a, dict) and isinstance(b, dict):

        return a.keys() == b.keys() and all(_close(a[k], b[k], absolute) for k in a)

    if isinstance(a, list) and isinstance(b, list):

        return len(a) == len(b) and all(_close(x, y, absolute) for x, y in zip(a, b))

    numbers = (int, float)

    if isinstance(a, numbers) and isinstance(b, numbers) and not isinstance(a, bool) and not isinstance(b, bool):

        return a == b or abs(a - b) <= max(absolute, 1e-9 * max(abs(a), abs(b)))

    return a == b


def _same(built, committed, absolute):

    if filecmp.cmp(built, committed, shallow=False):

        return True

    return _close(good.utilities.read_json(built), good.utilities.read_json(committed), absolute)


def check(seed):
    '''
    Rebuild into a temporary folder and report files that differ from Data/US/Processed.

    Numbers may differ in the last written digit (1e-5 for profiles, one part in
    1e9 otherwise), since floating-point results can vary slightly across platforms.
    '''

    with tempfile.TemporaryDirectory() as temporary:

        process(temporary, seed, verbose=False)

        comparison = filecmp.dircmp(temporary, PROCESSED)
        profiles = filecmp.dircmp(os.path.join(temporary, 'profiles'), os.path.join(PROCESSED, 'profiles'))

        top = [f for f in comparison.common_files
               if not _same(os.path.join(temporary, f), os.path.join(PROCESSED, f), 0.0)]
        series = [f for f in profiles.common_files
                  if not _same(os.path.join(temporary, 'profiles', f), os.path.join(PROCESSED, 'profiles', f), 1.5e-5)]

        problems = (
            [f'differs: {f}' for f in top]
            + [f'missing from Processed: {f}' for f in comparison.left_only]
            + [f'differs: profiles/{f}' for f in series]
            + [f'missing from Processed: profiles/{f}' for f in profiles.left_only]
            + [f'not produced by the build: profiles/{f}' for f in profiles.right_only]
        )

    if problems:

        print('Processed data is not reproducible from Raw:\n  ' + '\n  '.join(problems[:50]))

        return 1

    print('Processed data matches a fresh build.')

    return 0


def main(argv=None):

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='compare a fresh build with Data/US/Processed')
    parser.add_argument('--no-graphs', action='store_true', help='skip building graphs in Outputs/')
    parser.add_argument('--seed', type=int, default=0, help='seed for filling missing costs (default 0)')
    args = parser.parse_args(argv)

    warnings.filterwarnings('ignore', category=FutureWarning)

    if args.check:

        return check(args.seed)

    t0 = time.time()
    process(PROCESSED, args.seed)

    if not args.no_graphs:

        build_graphs(PROCESSED, OUTPUTS)

    print(f'Done in {time.time() - t0:.0f} s', flush=True)

    return 0


if __name__ == '__main__':

    sys.exit(main())
