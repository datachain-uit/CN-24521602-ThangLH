from __future__ import annotations

import math
import random

import networkx as nx
import pandas as pd
from scipy.optimize import minimize_scalar

from src.community.leiden import detect_communities_leiden
from src.graph.time_aware import (
    build_layer_edges_from_pairs,
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
        graph = build_layer_graph_from_pairs(pairs=pairs, nodes=nodes, beta=beta)
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
    beta, q, _ = calculate_optimal_beta_grid(events=events, action_type=action_type, seed=seed)
    return beta, q


def calculate_layer_modularity_leiden(
    events: pd.DataFrame,
    action_type: str | None,
    beta: float,
    seed: int = DEFAULT_SEED,
) -> float:
    return calculate_layer_modularity(events=events, action_type=action_type, beta=beta, seed=seed)


def calculate_optimal_beta_grid_multi_seed(
    events: pd.DataFrame,
    action_type: str | None,
    seeds: tuple[int, ...],
    verbose: bool = False,
) -> tuple[float, float, pd.DataFrame]:
    if not seeds:
        raise ValueError("At least one seed is required")

    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")

    pairs = prepare_action_pairs(events, action_type, keep_zero=False)
    nodes = sorted(events["user_id"].unique())

    best_beta = 0.0
    best_q = float("-inf")
    records = []

    for index in range(BETA_GRID_POINTS):
        beta = index / 100.0
        graph = build_layer_graph_from_pairs(pairs=pairs, nodes=nodes, beta=beta)
        q_values = [_leiden_modularity(graph, beta, seed) for seed in seeds]
        mean_q = math.fsum(q_values) / len(q_values)

        record = {"beta": beta, "modularity_q": mean_q, "n_seeds": len(seeds)}
        for seed, q in zip(seeds, q_values):
            record[f"q_seed_{seed}"] = q

        records.append(record)

        if mean_q > best_q:
            best_beta, best_q = beta, mean_q

        if verbose and index % 100 == 0:
            print(
                f"Checked {index + 1}/{BETA_GRID_POINTS} | "
                f"beta={beta:.2f} | "
                f"mean_Q={mean_q:.6f} | "
                f"best_beta={best_beta:.2f} | "
                f"best_mean_Q={best_q:.6f}",
                flush=True,
            )

    if verbose and best_beta == 10.0:
        print("Optimal beta is at the upper boundary: beta >= 10", flush=True)

    return best_beta, best_q, pd.DataFrame.from_records(records)


def _edge_graph(pairs: pd.DataFrame, beta: float, tolerance: float) -> nx.Graph:
    """Đồ thị chỉ gồm cạnh (không thêm nút cô lập), thêm theo thứ tự (u, v) tăng dần như combinations(sorted users) của tác giả."""
    edges = build_layer_edges_from_pairs(pairs, action_type=None, beta=beta)
    graph = nx.Graph()
    graph.add_weighted_edges_from((u, v, float(w)) for u, v, w in zip(edges["source_user"], edges["target_user"], edges["weight"]) if w > tolerance)
    return graph


def _louvain_modularity(graph: nx.Graph, rng: random.Random) -> float:
    """Q như calculate_modularity của tác giả (modularity.py:10-40): Louvain networkx rồi modularity, cùng trọng số."""
    if graph.number_of_edges() == 0:
        return 0.0
    communities = nx.community.louvain_communities(graph, weight="weight", seed=rng)
    return float(nx.community.modularity(graph, communities, weight="weight"))


def select_beta_brent(pairs: pd.DataFrame, tau, bounds: tuple[float, float], seed: int, tolerance: float = 1e-9) -> tuple[float, float, int]:
    """Chọn beta bằng Brent có biên như calculate_optimal_beta của tác giả (modularity.py:68-94); trả (beta, Q, nfev).

    pairs: bảng của prepare_action_pairs (dd = Δt phút / 1440) kèm cột t = thời điểm muộn hơn của cặp, đơn vị phút.
    Một RNG random.Random(seed) dùng xuyên suốt mọi lần gọi Louvain, tương đương random.seed(seed) trước mã gốc.
    Lớp không có cặp: trả (nan, nan, 0) và không gọi Louvain.
    """
    from src.eval.t7_checkpoints import assert_not_future, tau_minutes  # import muộn để tránh vòng import với t7_checkpoints
    if "t" not in pairs.columns:
        raise ValueError("pairs cần cột t (thời điểm muộn hơn của cặp, phút) để kiểm U2.")
    assert_not_future(pairs["t"], tau_minutes(tau), "select_beta_brent")
    if (pairs["dd"] < 0).any():
        raise ValueError("Có Δt âm: cặp không theo quy ước T10.")
    if pairs.empty:
        return float("nan"), float("nan"), 0
    rng = random.Random(seed)

    def objective(beta):
        return -_louvain_modularity(_edge_graph(pairs, beta, tolerance), rng)

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(result.x), float(-result.fun), int(result.nfev)
