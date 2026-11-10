from pathlib import Path
import pandas as pd
from src.simulation.synthetic import (
    SimulationConfig,
    generate_simulation1_dataset,
    generate_simulation2_dataset,
    generate_simulation3_dataset,
)
from src.community.modularity import calculate_optimal_beta_grid
from src.community.leiden import (
    detect_communities_leiden,
    detect_communities_multiplex,
)
from src.graph.time_aware import build_layer_graph
from src.eval.metrics import (
    best_community_f1,
    clustering_homogeneity,
    weighted_precision,
)

SEED = 20261108
OUTPUT_DIR = Path('reports')
GENERATORS = {
    1: generate_simulation1_dataset,
    2: generate_simulation2_dataset,
    3: generate_simulation3_dataset,
}
CONFIGURATIONS = [
    (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3),
]

def build_simulation_graphs():
    graphs = {}
    truth = None
    beta_rows = []
    config = SimulationConfig(seed=SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sim_id, generator in GENERATORS.items():
        events = generator(config)
        current_truth = set(events.loc[events['is_inauthentic'], 'user_id'])
        if truth is None:
            truth = current_truth
        elif current_truth != truth:
            raise ValueError(f'Ground-truth mismatch in Simulation {sim_id}')

        print(f'\n=== Simulation {sim_id}: beta grid ===', flush=True)
        beta, q, curve = calculate_optimal_beta_grid(
            events, action_type=None, seed=SEED, verbose=True
        )
        curve.to_csv(
            OUTPUT_DIR / f't6_simulation{sim_id}_beta_grid.csv',
            index=False,
        )
        graph = build_layer_graph(events, action_type=None, beta=beta)
        graphs[sim_id] = graph
        beta_rows.append({
            'simulation': sim_id,
            'beta': beta,
            'modularity_q': q,
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
        })
        print(f'Simulation {sim_id}: beta={beta:.2f}, Q={q:.6f}, '
              f'nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}',
              flush=True)

    pd.DataFrame(beta_rows).to_csv(
        OUTPUT_DIR / 't6_optimal_betas.csv', index=False,
    )
    return graphs, truth

def evaluate_configuration(sim_ids, graphs, truth):
    selected = [graphs[i] for i in sim_ids]
    if len(selected) == 1:
        communities = detect_communities_leiden(selected[0], seed=SEED)
    else:
        communities, _ = detect_communities_multiplex(selected, seed=SEED)
    assigned = [u for group in communities for u in group]
    expected = set().union(*(set(graph.nodes()) for graph in selected))
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise ValueError(f'Invalid partition for {sim_ids}')
    f1, _ = best_community_f1(communities, truth)
    return {
        'configuration': ' + '.join(map(str, sim_ids)),
        'n_layers': len(selected),
        'n_communities': len(communities),
        'f1_star': f1,
        'homogeneity': clustering_homogeneity(communities, truth),
        'weighted_precision': weighted_precision(communities, truth),
    }

def main():
    graphs, truth = build_simulation_graphs()
    results = pd.DataFrame([
        evaluate_configuration(ids, graphs, truth)
        for ids in CONFIGURATIONS
    ])
    metrics = ['f1_star', 'homogeneity', 'weighted_precision']
    results[metrics] = results[metrics].clip(0, 1)
    path = OUTPUT_DIR / 't6_table2_reproduced.csv'
    results.to_csv(path, index=False, float_format='%.6f')
    print('\n=== Seven configurations ===')
    print(results.to_string(index=False))
    print(f'\nSaved: {path}')

if __name__ == '__main__':
    main()