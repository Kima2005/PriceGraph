import networkx as nx
__author__ = "\n".join(['Sebastian Pucilowski <smopucilowski@gmail.com>'])
__all__ = ['collective_influence']


def collective_influence(G, u=None, distance=2):
  
    reduced_degree = lambda node: G.degree(node)-1

    if u is None:
        nodes = G.nodes()
    else:
        nodes = [u]
    collective_influence = dict()
    for node in nodes:
        # Find all nodes at 'distance' steps from node
        path_lengths = nx.single_source_shortest_path_length(
            G, source=node, cutoff=distance
        )
        # print(f"Node: {node}")
        # print(f"Path lengths from node {node}: {path_lengths}")
        frontier_nodes = {
            node for node, path_length in path_lengths.items()
            if path_length == distance
        }
        # print(f"Frontier nodes at distance {distance} from node {node}: {frontier_nodes}")
        node_ci = reduced_degree(node) * sum(map(reduced_degree, frontier_nodes))
        # print(f"Reduced degree of node {node}: {reduced_degree(node)}")
        # print(f"Sum of reduced degrees of frontier nodes: {sum(map(reduced_degree, frontier_nodes))}")
        # print(f"Collective influence of node {node}: {node_ci}")
        collective_influence[node] = node_ci
    if u is not None:
        return collective_influence[u]
    else:
        return collective_influence
