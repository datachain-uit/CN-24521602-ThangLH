from __future__ import annotations

import math

import networkx as nx
import pandas as pd

from src.community.leiden import detect_communities_leiden
from src.graph.time_aware import (
    build_layer_graph,
    build_layer_graph_from_pairs,
    prepare_action_pairs,
)

DEFAULT_SEED = 20261108
BETA_GRID_POINTS = 1001


def _leiden_modularity(graph: nx.Graph, beta: float, seed: int) -> float:
    if graph.number_of_edges() == 0:
        return 0.0

    communities = detect_communities_leiden(graph, seed=seed)
    q = float(nx.community.modularity(graph, communities, weight="weight"))
    if not math.isfinite(q):
        raise ValueError(f"Non-finite modularity at beta={beta}")
    return q


def calculate_layer_modularity(
    events: pd.DataFrame,
    action_type: str | None,
    beta: float,
    seed: int = DEFAULT_SEED,
) -> float:
    graph = build_layer_graph(events, action_type, beta)
    return _leiden_modularity(graph, beta, seed)


def calculate_optimal_beta_grid(
    events: pd.DataFrame,
    action_type: str | None,
    seed: int = DEFAULT_SEED,
    verbose: bool = False,
) -> tuple[float, float, pd.DataFrame]:
    pairs = prepare_action_pairs(events, action_type)
    nodes = events["user_id"].unique()

    best_beta = 0.0
    best_q = float("-inf")
    records = []

    for index in range(BETA_GRID_POINTS):
        beta = index / 100.0
        graph = build_layer_graph_from_pairs(
            pairs=pairs, nodes=nodes, beta=beta,
        )
        q = _leiden_modularity(graph, beta, seed)
        records.append({"beta": beta, "modularity_q": q})

        if q > best_q:
            best_beta, best_q = beta, q

        if verbose and index % 100 == 0:
            print(
                f"Checked {index + 1}/{BETA_GRID_POINTS} | "
                f"beta={beta:.2f} | "
                f"best_beta={best_beta:.2f} | "
                f"best_Q={best_q:.6f}",
                flush=True,
            )

    return best_beta, best_q, pd.DataFrame.from_records(records)


def calculate_optimal_beta(
    events: pd.DataFrame,
    action_type: str | None,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    beta, q, _ = calculate_optimal_beta_grid(
        events=events, action_type=action_type, seed=seed,
    )
    return beta, q

def calculate_layer_modularity_leiden(
    events: pd.DataFrame,
    action_type: str | None,
    beta: float,
    seed: int = DEFAULT_SEED,
) -> float:
    return calculate_layer_modularity(
        events=events, action_type=action_type, beta=beta, seed=seed,
    )