"""Run R03 §6.4 diagnostics without changing the archived T6b experiment."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
from itertools import product
import json
import math
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from types import ModuleType

import numpy as np
import pandas as pd
from time_aware_collaboration import models as reference_models
from time_aware_collaboration import modularity as reference_modularity

from scripts.run_t6b_experiment import (
    CONFIG_PATH, GENERATORS, ROOT, SimulationConfig, generate_data, load_config, original_edge_list,
)
from src.community.modularity import _leiden_modularity
from src.graph import time_aware


GRAPH_BASELINE = "0db9c18812928b93507a122a0ee6275737cd3a03"
HISTORY = ROOT / "reports" / "t6b" / "full"
EDGE_TOLERANCE = 1e-9
RESULT_KEYS = ["data_seed", "community_seed", "pipeline", "configuration"]
BETA_KEYS = ["data_seed", "simulation", "pipeline"]
GREEDY_KEY = ["data_seed", "pipeline", "configuration"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_result_keys(results, config):
    configurations = ["+".join(map(str, layers)) for layers in config["configurations"]]
    expected = pd.MultiIndex.from_product(
        [config["data_seeds"], config["community_seeds"], config["pipelines"], configurations],
        names=RESULT_KEYS,
    )
    actual = pd.MultiIndex.from_frame(results[RESULT_KEYS])
    require(expected.is_unique and actual.is_unique, "Duplicate T6b result or configuration key")
    require(actual.sort_values().equals(expected.sort_values()),
            "T6b results do not cover the configured Cartesian product")


def read_archive(config, history):
    results = pd.read_csv(history / "results_all.csv", dtype={"configuration": str})
    validate_result_keys(results, config)
    betas = pd.read_csv(history / "betas_all.csv", dtype={"beta_reported": str})
    expected = set(product(config["data_seeds"], GENERATORS, config["pipelines"]))
    require(
        len(betas) == len(expected) == len(betas[BETA_KEYS].drop_duplicates())
        and set(betas[BETA_KEYS].itertuples(index=False, name=None)) == expected,
        "T6b beta archive has missing, duplicate or unexpected keys",
    )

    for prefix, aggregate, keys, dtype in (
        ("results", results, RESULT_KEYS, {"configuration": str}),
        ("betas", betas, BETA_KEYS, {"beta_reported": str}),
    ):
        frames = []
        for seed in config["data_seeds"]:
            path = history / f"{prefix}_{seed}.csv"
            frame = pd.read_csv(path, dtype=dtype)
            require(frame["data_seed"].eq(seed).all(), f"Wrong data seed in {path.name}")
            frames.append(frame)
        pd.testing.assert_frame_equal(
            aggregate.set_index(keys).sort_index(),
            pd.concat(frames, ignore_index=True).set_index(keys).sort_index(),
            check_dtype=False, rtol=1e-13, atol=1e-15,
        )

    require(results[config["metrics"]].apply(
        lambda series: series.between(-1e-12, 1.0 + 1e-12).all()
    ).all(), "Invalid T6b metrics")
    community_counts = results["n_communities"]
    require(community_counts.between(1, 46).all() and community_counts.mod(1).eq(0).all(),
            "Invalid T6b community count")
    require(betas["beta"].between(*config["beta_bounds"]).all(), "Beta out of bounds")
    student = betas[betas["pipeline"].eq("student")]
    original = betas[betas["pipeline"].eq("original_reference")]
    grid_steps = student["beta"] / config["beta_grid_step"]
    require(np.allclose(grid_steps, grid_steps.round(), rtol=0, atol=1e-10), "Beta off grid")
    on_boundary = student["beta"].eq(config["beta_bounds"][1])
    require(student["upper_boundary"].eq(on_boundary).all()
            and student["beta_reported"].eq(">=10").eq(on_boundary).all(),
            "Student beta boundary flags disagree")
    require(student["optimization_seed"].isna().all()
            and student["mean_modularity"].between(-1.0, 1.0).all()
            and original["mean_modularity"].isna().all()
            and original["upper_boundary"].isna().all()
            and original["optimization_seed"].eq(original["data_seed"] * 10 + original["simulation"]).all(),
            "Beta optimization metadata disagree with the runner")
    for simulation in GENERATORS:
        column = f"beta_{simulation}"
        lookup = betas[betas["simulation"].eq(simulation)].set_index(["data_seed", "pipeline"])["beta"]
        joined = results.join(lookup.rename("expected_beta"), on=["data_seed", "pipeline"])
        selected = joined["configuration"].str.split("+").map(lambda layers: str(simulation) in layers)
        require(joined.loc[~selected, column].isna().all() and np.allclose(
            joined.loc[selected, column], joined.loc[selected, "expected_beta"], rtol=0, atol=1e-13,
        ), f"Result beta does not match beta archive for simulation {simulation}")

    return results, betas.set_index(BETA_KEYS).sort_index()


def effective_results(results, metrics):
    greedy = results["pipeline"].eq("original_reference") & ~results["configuration"].str.contains(
        "+", regex=False
    )
    require(results["community_seed_effective"].eq(~greedy).all(), "Incorrect community seed effect flags")
    deterministic = results.loc[greedy].sort_values("community_seed")
    values = deterministic.groupby(GREEDY_KEY)[[*metrics, "n_communities"]].nunique(dropna=False)
    require(values.eq(1).all().all(), "Greedy community placeholders disagree within a data seed")
    return pd.concat([results.loc[~greedy], deterministic.drop_duplicates(GREEDY_KEY)])


def summarize_results(results, config):
    metrics = config["metrics"]
    effective = effective_results(results, metrics).copy()
    effective[metrics] = effective[metrics].clip(0, 1)
    u4_config = "+".join(map(str, config["u4_configuration"]))
    target = config["u4_f1_threshold"]
    summary, u4 = [], []
    for (pipeline, configuration), group in effective.groupby(["pipeline", "configuration"]):
        record = {
            "pipeline": pipeline,
            "configuration": configuration,
            "n_archived_rows": len(results[
                results["pipeline"].eq(pipeline) & results["configuration"].eq(configuration)
            ]),
            "n_effective_runs": len(group),
            "n_data_seeds": group["data_seed"].nunique(),
        }
        for metric in metrics:
            for q in config["quantiles"]:
                record[f"{metric}_p{round(q * 100)}"] = group[metric].quantile(q)
        summary.append(record)
        if configuration == u4_config:
            f1 = group["f1_star"]
            u4.append({
                "pipeline": pipeline,
                "n_archived_rows": record["n_archived_rows"],
                "n_effective_runs": len(f1),
                "n_data_seeds": record["n_data_seeds"],
                "f1_star_p10": record["f1_star_p10"],
                "f1_star_p50": record["f1_star_p50"],
                "f1_star_p90": record["f1_star_p90"],
                "published_f1": target,
                "n_f1_ge_threshold": int(f1.ge(target).sum()),
                "fraction_f1_ge_threshold": float(f1.ge(target).mean()),
                "u4_pass": record["f1_star_p10"] <= target <= record["f1_star_p90"],
            })
    return pd.DataFrame(summary), pd.DataFrame(u4)


def timestamp_counts(raw, rounded):
    counts = {}
    for name, events in (("before", raw), ("after", rounded)):
        pairs = time_aware.find_next_action_pairs(events, keep_zero=True)
        counts[f"n_pairs_{name}"] = len(pairs)
        counts[f"n_zero_{name}"] = int(pairs["delta_t"].eq(0).sum())
    return counts


def graph_weights(graph):
    return {tuple(sorted((u, v))): weight for u, v, weight in graph.edges(data="weight")}


def reference_weights(edge_list, beta):
    return {(u, v): float(weight_fn(beta)) for u, v, weight_fn, _ in edge_list}


def compare_weights(student, reference):
    union = student.keys() | reference.keys()
    return {
        "n_student_edges": len(student),
        "n_reference_edges": len(reference),
        "n_student_only": len(student.keys() - reference.keys()),
        "n_reference_only": len(reference.keys() - student.keys()),
        "max_absolute_error": max(
            (abs(student.get(edge, 0.0) - reference.get(edge, 0.0)) for edge in union),
            default=0.0,
        ),
    }


def check_edge_weights(pairs, users, edge_list, beta, tolerance=EDGE_TOLERANCE):
    graph = time_aware.build_layer_graph_from_pairs(pairs, users, beta, tolerance)
    reference = reference_weights(edge_list, beta)
    filtered = {edge: weight for edge, weight in reference.items() if weight > tolerance}
    return compare_weights(graph_weights(graph), filtered)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def diagnose(config, betas):
    # Load the known scalar T10 revision to compare its weighted pair inputs
    # with the working-tree vectorized implementation on each archived layer.
    baseline = ModuleType("time_aware_t10_baseline")
    source = git("show", f"{GRAPH_BASELINE}:src/graph/time_aware.py")
    exec(compile(source, f"{GRAPH_BASELINE}:src/graph/time_aware.py", "exec"), baseline.__dict__)
    counts, edge_checks, compatibility = [], [], []
    for seed in config["data_seeds"]:
        rounded, _, users = generate_data(seed)
        for simulation, events in rounded.items():
            keys = {"data_seed": seed, "simulation": simulation}
            raw = GENERATORS[simulation](SimulationConfig(seed=seed))
            counts.append({**keys, **timestamp_counts(raw, events)})
            student_beta = betas.loc[(seed, simulation, "student")]
            original_beta = betas.loc[(seed, simulation, "original_reference")]
            pairs = time_aware.prepare_action_pairs(events, None, keep_zero=False)
            scalar_pairs = baseline.prepare_action_pairs(events, None, keep_zero=False)
            edge_list = original_edge_list(events, users)
            for role, beta in (
                ("lower_bound", config["beta_bounds"][0]),
                ("student_beta", student_beta["beta"]),
                ("original_beta", original_beta["beta"]),
                ("upper_bound", config["beta_bounds"][1]),
            ):
                edge_checks.append({
                    **keys, "beta_role": role, "beta": beta,
                    **check_edge_weights(pairs, users, edge_list, beta),
                })

            # Evaluate the objective at the saved beta only, using the node
            # order used by the grid search. The archived reference is unfiltered.
            student_graph = time_aware.build_layer_graph_from_pairs(pairs, users, student_beta["beta"])
            original_weights = reference_weights(edge_list, original_beta["beta"])
            mean_q = math.fsum(
                _leiden_modularity(student_graph, student_beta["beta"], community_seed)
                for community_seed in config["community_seeds"]
            ) / len(config["community_seeds"])
            compatibility.append({
                **keys,
                "scalar_pairs_equal": pairs.equals(scalar_pairs),
                "student_edges_archived": int(student_beta["n_edges"]),
                "student_edges_recomputed": student_graph.number_of_edges(),
                "original_edges_archived": int(original_beta["n_edges"]),
                "original_edges_recomputed": len(original_weights),
                "original_edges_at_or_below_tolerance": sum(
                    weight <= EDGE_TOLERANCE for weight in original_weights.values()
                ),
                "student_mean_q_archived": student_beta["mean_modularity"],
                "student_mean_q_recomputed": mean_q,
                "student_mean_q_absolute_error": abs(mean_q - student_beta["mean_modularity"]),
            })
    return pd.DataFrame(counts), pd.DataFrame(edge_checks), pd.DataFrame(compatibility)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for diagnostic outputs")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    config = load_config()
    results, betas = read_archive(config, HISTORY)
    counts, edges, compatibility = diagnose(config, betas)
    counts.to_csv(output / "delta_t_zero_by_seed.csv", index=False)
    edges.to_csv(output / "u1_original_edge_weights.csv", index=False)
    compatibility.to_csv(output / "archive_graph_validation.csv", index=False)
    max_error = float(edges["max_absolute_error"].max())
    require(edges[["n_student_only", "n_reference_only"]].eq(0).all().all()
            and max_error < EDGE_TOLERANCE, "U1-original failed. Inspect u1_original_edge_weights.csv.")
    require(compatibility["scalar_pairs_equal"].all()
            and compatibility["student_edges_archived"].eq(compatibility["student_edges_recomputed"]).all()
            and compatibility["original_edges_archived"].eq(compatibility["original_edges_recomputed"]).all()
            and compatibility["student_mean_q_absolute_error"].lt(1e-12).all(),
            "Archive graph validation failed. Inspect archive_graph_validation.csv.")

    summary, u4 = summarize_results(results, config)
    summary.to_csv(output / "summary_statistics.csv", index=False)
    u4.to_csv(output / "u4_distribution.csv", index=False)
    fingerprinted = [
        CONFIG_PATH, HISTORY / "results_all.csv", HISTORY / "betas_all.csv",
        Path(__file__).resolve(), ROOT / "scripts" / "run_t6b_experiment.py",
        ROOT / "src" / "simulation" / "synthetic.py", ROOT / "src" / "graph" / "time_aware.py",
        ROOT / "src" / "community" / "modularity.py", ROOT / "src" / "community" / "leiden.py",
        ROOT / "src" / "eval" / "metrics.py",
    ]
    reference_root = ROOT / "references" / "time-aware-collaboration"
    reference_source = {}
    for module in (reference_models, reference_modularity):
        active = Path(module.__file__)
        vendored = reference_root / "time_aware_collaboration" / active.name
        reference_source[active.name] = {
            "sha256": sha256(active.read_bytes()).hexdigest(),
            "matches_vendored_source": active.read_text(encoding="utf-8") == vendored.read_text(encoding="utf-8"),
        }
    metadata = {
        "project_head": git("rev-parse", "HEAD").decode().strip(),
        "baseline_graph_revision": GRAPH_BASELINE,
        "reference_head": git("-C", str(reference_root), "rev-parse", "HEAD").decode().strip(),
        "reference_source": reference_source,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in (
            "numpy", "pandas", "networkx", "scipy", "igraph", "leidenalg", "scikit-learn", "pytest",
        )},
        "sha256": {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest() for path in fingerprinted},
        "n_archived_rows": len(results),
        "n_archived_beta_rows": len(betas),
        "max_absolute_error": max_error,
        "elapsed_wall_seconds": round(perf_counter() - start, 3),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(results)} archived rows and {len(edges)} same-beta comparisons")
    print(counts.groupby("simulation").sum(numeric_only=True).drop(columns="data_seed").to_string())
    print(f"U1 maximum absolute error: {max_error:.12g}")
    print(u4.to_string(index=False))
    print(f"Elapsed wall seconds: {metadata['elapsed_wall_seconds']}")


if __name__ == "__main__":
    main()
