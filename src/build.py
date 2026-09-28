import networkx as nx


def build_graph(assets, lines, profiles, **graph_attributes):
    '''A GOOD 2.x graph: one Region node per region, one Link edge per directed pair.'''

    graph = nx.DiGraph(**graph_attributes)
    graph.add_nodes_from(build_nodes(assets, profiles))
    graph.add_edges_from(build_edges(lines))

    return graph


def build_edges(lines):

    pairs = sorted({(v['source'], v['target']) for v in lines.values()})

    edges = []

    for source, target in pairs:

        edge = {
            'id': f'{source}:{target}',
            '_class': 'Link',
            'lines': {
                k: {key: value for key, value in v.items() if key not in ('source', 'target')}
                for k, v in lines.items() if v['source'] == source and v['target'] == target
            },
        }

        edges.append((source, target, edge))

    return edges


def build_nodes(assets, profiles):

    regions = sorted({p['region'] for p in assets.values()})

    by_region = {}

    for key, value in profiles.items():

        by_region.setdefault(key.split(':')[0], {})[key] = value

    nodes = []

    for region in regions:

        node = {
            'id': region,
            '_class': 'Region',
            'assets': {k: v for k, v in assets.items() if v['region'] == region},
        }

        referenced = {a.get('profile') for a in node['assets'].values()}
        node['profiles'] = {k: v for k, v in by_region.get(region, {}).items() if k in referenced}

        nodes.append((region, node))

    return nodes
