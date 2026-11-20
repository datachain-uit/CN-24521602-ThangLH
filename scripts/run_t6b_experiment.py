
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import pandas as pd

from src.simulation.synthetic import SimulationConfig, generate_simulation1_dataset, generate_simulation2_dataset, generate_simulation3_dataset

from src.graph.time_aware import build_layer_graph

from src.community.modularity import calculate_optimal_beta_grid_multi_seed

from src.community.leiden import detect_communities_leiden, detect_communities_multiplex

from src.eval.metrics import best_community_f1, clustering_homogeneity, weighted_precision

from time_aware_collaboration.models import get_time_damped_edge_list

from time_aware_collaboration.modularity import calculate_optimal_beta as original_optimal_beta


with open(ROOT / "configs" / "paths.json", encoding="utf-8") as f: PATHS = json.load(f)

CONFIG_PATH = ROOT / PATHS["t6b_config"]

GENERATORS = {1: generate_simulation1_dataset, 2: generate_simulation2_dataset, 3: generate_simulation3_dataset}

BASE_TIME = pd.Timestamp("2020-01-01")


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def round_timestamps(events):
    events = events.copy()
    seconds = np.rint(events["timestamp"].to_numpy(dtype=float) * 60).astype(np.int64)
    events["timestamp"] = seconds / 60.0
    events["time_string"] = [(BASE_TIME + pd.Timedelta(seconds=int(s))).strftime("%Y-%m-%d %H:%M:%S") for s in seconds]
    return events


def generate_data(data_seed):
    datasets = {}
    truths = {}
    node_sets = {}

    for sim_id, generator in GENERATORS.items():
        events = round_timestamps(generator(SimulationConfig(seed=data_seed)))

        datasets[sim_id] = events

        truths[sim_id] = set(events.loc[events["is_inauthentic"], "user_id"])

        node_sets[sim_id] = set(events["user_id"])

    assert truths[1] == truths[2] == truths[3]
    assert node_sets[1] == node_sets[2] == node_sets[3]

    return datasets, truths[1], sorted(node_sets[1])


def original_edge_list(events, users):
    B = nx.Graph()
    B.add_nodes_from(users, bipartite=0)

    for row in events.itertuples(index=False):
        content = (row.action_type, row.content_id)

        B.add_node(content, bipartite=1)

        if B.has_edge(row.user_id, content):
            B[row.user_id][content]["timestamps"].append(row.time_string)
        else:
            B.add_edge(row.user_id, content, timestamps=[row.time_string])

    return list(get_time_damped_edge_list(B, users))


def build_original_graph(events, users, data_seed, sim_id):
    edge_list = original_edge_list(events, users)
    if not edge_list:
        raise ValueError("Original edge list is empty")

    # Control the randomness of original beta optimization.
    optimization_seed = data_seed * 10 + sim_id

    random.seed(optimization_seed)
    np.random.seed(optimization_seed)

    beta = float(original_optimal_beta(edge_list, bounds=(0, 10)))

    G = nx.Graph()
    G.add_nodes_from(users)

    for u, v, weight_fn, _ in edge_list:
        G.add_edge(u, v, weight=float(weight_fn(beta)))

    return G, beta, optimization_seed


def build_student_graph(events, community_seeds):
    beta, mean_q, _ = calculate_optimal_beta_grid_multi_seed(
        events=events, action_type=None, seeds=tuple(community_seeds), verbose=False,
    )

    G = build_layer_graph(events=events, action_type=None, beta=beta, tolerance=1e-9, keep_zero=False)

    return G, float(beta), float(mean_q)


def detect_original(graphs, users, community_seed):
    if len(graphs) == 1:
        G = graphs[0]

        if G.number_of_edges() == 0:
            return [{node} for node in users]

        return [set(c) for c in nx.community.greedy_modularity_communities(G, weight="weight")]

    node_index = {node: i for i, node in enumerate(users)}

    layers = []
    layer_weights = []

    for G in graphs:
        g = ig.Graph(n=len(users), directed=False)

        edges = list(G.edges(data=True))

        g.add_edges([(node_index[u], node_index[v]) for u, v, _ in edges])

        g.es["weight"] = [float(d["weight"]) for _, _, d in edges]

        layers.append(g)
        layer_weights.append(g.ecount())

    membership, _ = la.find_partition_multiplex(layers, la.SurpriseVertexPartition, weights="weight", layer_weights=layer_weights, n_iterations=-1, seed=community_seed)

    groups = {}

    for i, community_id in enumerate(membership):
        groups.setdefault(community_id, set()).add(users[i])

    return list(groups.values())


def detect_student(graphs, community_seed):
    if len(graphs) == 1:
        return detect_communities_leiden(graphs[0], seed=community_seed)

    communities, _ = detect_communities_multiplex(graphs, seed=community_seed)

    return communities


def evaluate(communities, truth, users):
    # Every account must occur in exactly one community.
    members = [node for community in communities for node in community]

    assert len(members) == len(set(members))
    assert set(members) == set(users)

    f1, _ = best_community_f1(communities, truth)

    h = clustering_homogeneity(communities, truth)

    wp = weighted_precision(communities, truth)

    values = [float(f1), float(h), float(wp)]

    assert all(np.isfinite(v) and -1e-12 <= v <= 1 + 1e-12 for v in values)

    return {
        "f1_star": float(f1),
        "homogeneity": float(h),
        "weighted_precision": float(wp),
        "n_communities": len(communities),
    }


def atomic_save(df, path):
    temp_path = path.with_suffix(".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)


def run_one_seed(data_seed, config, output_dir):
    start = time.perf_counter()

    community_seeds = config["community_seeds"]

    datasets, truth, users = generate_data(data_seed)

    graphs = {"student": {}, "original_reference": {}}

    betas = {"student": {}, "original_reference": {}}

    beta_records = []

    for sim_id, events in datasets.items():
        print(f"Data seed={data_seed} | Simulation={sim_id} | Building graphs", flush=True)

        G_student, beta_student, mean_q = build_student_graph(events, community_seeds)

        G_original, beta_original, optimization_seed = build_original_graph(events, users, data_seed, sim_id)

        graphs["student"][sim_id] = G_student
        graphs["original_reference"][sim_id] = G_original

        betas["student"][sim_id] = beta_student
        betas["original_reference"][sim_id] = beta_original

        beta_records.append({
            "data_seed": data_seed,
            "simulation": sim_id,
            "pipeline": "student",
            "beta": beta_student,
            "beta_reported": ">=10" if beta_student == 10.0 else str(beta_student),
            "upper_boundary": beta_student == 10.0,
            "mean_modularity": mean_q,
            "optimization_seed": None,
            "n_edges": G_student.number_of_edges(),
        })

        beta_records.append({
            "data_seed": data_seed,
            "simulation": sim_id,
            "pipeline": "original_reference",
            "beta": beta_original,
            "beta_reported": str(beta_original),
            "upper_boundary": None,
            "mean_modularity": None,
            "optimization_seed": optimization_seed,
            "n_edges": G_original.number_of_edges(),
        })

        print(f"  Student beta={beta_student:.6f} | Original beta={beta_original:.6f}", flush=True)

    results = []

    for pipeline in graphs:
        for configuration in config["configurations"]:
            selected = [graphs[pipeline][i] for i in configuration]

            name = "+".join(map(str, configuration))

            deterministic = pipeline == "original_reference" and len(configuration) == 1

            cached_metrics = None

            for community_seed in community_seeds:
                if deterministic and cached_metrics is not None:
                    metrics = cached_metrics

                else:
                    if pipeline == "student":
                        communities = detect_student(selected, community_seed)
                    else:
                        communities = detect_original(selected, users, community_seed)

                    metrics = evaluate(communities, truth, users)

                    if deterministic:
                        cached_metrics = metrics

                results.append({
                    "data_seed": data_seed,
                    "community_seed": community_seed,
                    "pipeline": pipeline,
                    "configuration": name,
                    "community_seed_effective": not deterministic,
                    "beta_1": betas[pipeline][1] if 1 in configuration else None,
                    "beta_2": betas[pipeline][2] if 2 in configuration else None,
                    "beta_3": betas[pipeline][3] if 3 in configuration else None,
                    **metrics,
                })

    df = pd.DataFrame(results)
    beta_df = pd.DataFrame(beta_records)

    expected = 2 * len(config["configurations"]) * len(community_seeds)

    assert len(df) == expected
    assert len(beta_df) == 6

    keys = ["pipeline", "data_seed", "community_seed", "configuration"]

    assert not df.duplicated(keys).any()

    atomic_save(beta_df, output_dir / f"betas_{data_seed}.csv")
    atomic_save(df, output_dir / f"results_{data_seed}.csv")

    elapsed = time.perf_counter() - start

    print(f"COMPLETED seed={data_seed} | rows={len(df)} | elapsed={elapsed:.2f}s", flush=True)

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pilot", action="store_true", help="Run one data seed for validation")

    args = parser.parse_args()
    config = load_config()

    data_seeds = config["data_seeds"]

    if args.pilot:
        data_seeds = data_seeds[:1]
        output_dir = ROOT / PATHS["t6b_pilot_dir"]
    else:
        output_dir = ROOT / PATHS["t6b_full_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== T6b EXPERIMENT ===", flush=True)
    print("Mode:", "PILOT" if args.pilot else "FULL")
    print("Data seeds:", len(data_seeds))
    print("Community seeds:", config["community_seeds"])
    print("Output directory:", output_dir)

    frames = []
    start = time.perf_counter()

    for index, data_seed in enumerate(data_seeds, 1):
        print(f"\n=== DATA SEED {index}/{len(data_seeds)} ({data_seed}) ===", flush=True)

        df = run_one_seed(data_seed, config, output_dir)

        frames.append(df)

        atomic_save(pd.concat(frames, ignore_index=True), output_dir / "results_all.csv")

    all_results = pd.concat(frames, ignore_index=True)

    print("\n=== EXPERIMENT FINISHED ===")
    print("Total rows:", len(all_results))
    print("Elapsed seconds:", round(time.perf_counter() - start, 2))
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
