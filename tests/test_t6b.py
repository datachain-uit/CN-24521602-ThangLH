
import numpy as np
import pandas as pd
import pytest

from src.community import modularity

from src.simulation.synthetic import (
    SimulationConfig,
    generate_simulation1_timestamps,
    generate_simulation1_dataset,
    generate_simulation2_dataset,
    generate_simulation3_dataset,
)

DATA_SEEDS = range(20261108, 20261138)


@pytest.mark.parametrize("seed", DATA_SEEDS)
def test_simulation1_burst_event_count(seed):
    config = SimulationConfig(seed=seed)

    timestamps = generate_simulation1_timestamps(config)

    assert config.n_inauthentic_events_min <= len(timestamps) <= config.n_inauthentic_events_max

    assert np.isfinite(timestamps).all()
    assert np.all(np.diff(timestamps) >= 0)

    assert np.all(timestamps >= 1920.0)
    assert np.all(timestamps <= 1935.0)


@pytest.mark.parametrize("seed", DATA_SEEDS)
def test_all_simulations_constraints(seed):
    config = SimulationConfig(seed=seed)

    generators = [generate_simulation1_dataset, generate_simulation2_dataset, generate_simulation3_dataset]

    node_sets = []
    truth_sets = []

    for generator in generators:
        events = generator(config)

        authentic = events.loc[~events["is_inauthentic"]]
        coordinated = events.loc[events["is_inauthentic"]]

        assert len(authentic) == 1000
        assert 15 <= len(coordinated) <= 20

        assert authentic["user_id"].nunique() == 40
        assert coordinated["user_id"].nunique() == 6
        assert events["user_id"].nunique() == 46

        assert not events.isna().any().any()

        assert events["timestamp"].between(0, config.duration_days * 1440).all()

        assert events["action_type"].nunique() == 3

        node_sets.append(set(events["user_id"]))
        truth_sets.append(set(coordinated["user_id"]))

    assert node_sets[0] == node_sets[1] == node_sets[2]
    assert truth_sets[0] == truth_sets[1] == truth_sets[2]


def test_simulation1_seed_reproducibility():
    config = SimulationConfig(seed=20261109)

    first = generate_simulation1_dataset(config)
    second = generate_simulation1_dataset(config)

    pd.testing.assert_frame_equal(first, second)


def test_simulation1_invalid_constraints():
    config = SimulationConfig(n_inauthentic_events_min=21, n_inauthentic_events_max=20)

    with pytest.raises(ValueError):
        generate_simulation1_timestamps(config)


def test_beta_mean_before_argmax(monkeypatch):
    # Construct known modularity values for three
    # beta candidates and two community seeds.
    q_values = {
        (0.00, 42): 0.80,
        (0.00, 43): 0.00,
        (0.01, 42): 0.60,
        (0.01, 43): 0.60,
        (0.02, 42): 0.00,
        (0.02, 43): 1.00,
    }

    monkeypatch.setattr(modularity, "BETA_GRID_POINTS", 3)
    monkeypatch.setattr(modularity, "prepare_action_pairs", lambda *args, **kwargs: None)
    monkeypatch.setattr(modularity, "build_layer_graph_from_pairs", lambda **kwargs: None)

    monkeypatch.setattr(
        modularity, "_leiden_modularity", lambda graph, beta, seed: q_values[(round(beta, 2), seed)],
    )

    events = pd.DataFrame({"user_id": ["u1", "u2"]})

    beta, mean_q, records = modularity.calculate_optimal_beta_grid_multi_seed(
        events=events, action_type=None, seeds=(42, 43),
    )

    assert beta == pytest.approx(0.01)
    assert mean_q == pytest.approx(0.60)

    assert records["modularity_q"].tolist() == pytest.approx([0.40, 0.60, 0.50])


@pytest.mark.parametrize("seeds", [(), (42, 42)])
def test_beta_invalid_seeds(seeds):
    with pytest.raises(ValueError):
        modularity.calculate_optimal_beta_grid_multi_seed(
            events=pd.DataFrame(), action_type=None, seeds=seeds,
        )
