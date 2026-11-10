import igraph as ig
import leidenalg as la
import networkx as nx

def detect_communities_leiden(
    graph: nx.Graph,
    seed: int = 20261108,
) -> list[set[str]]:
    nodes = list(graph.nodes())

    ig_graph = ig.Graph(n=len(nodes), directed=False)

    node_index = {
        node: i
        for i, node in enumerate(nodes)
    }

    edges = [
        (node_index[u], node_index[v])
        for u, v in graph.edges()
    ]

    ig_graph.add_edges(edges)

    weights = [
        float(data["weight"])
        for _, _, data in graph.edges(data=True)
    ]

    partition = la.find_partition(
        ig_graph,
        la.ModularityVertexPartition,
        weights=weights,
        seed=seed,
    )

    return [
        {nodes[i] for i in community}
        for community in partition
    ]


def convert_multiplex_layers(
    graphs: list[nx.Graph],
) -> tuple[list[ig.Graph], list[str]]:
    if not graphs:
        raise ValueError("At least one graph is required.")

    all_nodes = sorted(
        set().union(*(set(g.nodes()) for g in graphs))
    )

    node_index = {
        node: i
        for i, node in enumerate(all_nodes)
    }

    ig_layers = []

    for graph in graphs:
        layer = ig.Graph(
            n=len(all_nodes),
            directed=False,
        )

        layer.vs["name"] = all_nodes

        edges = [
            (node_index[u], node_index[v])
            for u, v in graph.edges()
        ]

        weights = [
            float(data["weight"])
            for _, _, data in graph.edges(data=True)
        ]

        layer.add_edges(edges)
        layer.es["weight"] = weights

        ig_layers.append(layer)

    return ig_layers, all_nodes


def detect_communities_multiplex(
    graphs: list[nx.Graph],
    seed: int = 42,
) -> tuple[list[set[str]], float]:
    layers, nodes = convert_multiplex_layers(graphs)

    layer_weights = [1.0] * len(layers)

   
    membership, quality = la.find_partition_multiplex(
    layers,
    la.ModularityVertexPartition,
    weights="weight",
    layer_weights=layer_weights,
    n_iterations=-1,
    seed=seed,
)

    communities_by_id: dict[int, set[str]] = {}

    for vertex_index, community_id in enumerate(membership):
        communities_by_id.setdefault(
            community_id, set()
        ).add(nodes[vertex_index])

    return list(communities_by_id.values()), float(quality)