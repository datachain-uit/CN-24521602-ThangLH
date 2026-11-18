import math

import pandas as pd
import pytest

from scripts import diagnose_t6b as diagnostics
from scripts.run_t6b_experiment import load_config, original_edge_list, round_timestamps
from src.graph.time_aware import PAIR_COLUMNS, find_next_action_pairs, prepare_action_pairs


def events(rows):
    return pd.DataFrame(
        [(user, content, second / 60, "hashtag") for user, content, second in rows],
        columns=["user_id", "content_id", "timestamp", "action_type"],
    )


def test_rounding_tie_uses_canonical_direction_and_preserves_raw_timestamps():
    raw = events([("b", "x", 0.4), ("a", "x", 0.1), ("b", "x", 2)])
    rounded = round_timestamps(raw)
    assert raw["timestamp"].tolist() == [0.4 / 60, 0.1 / 60, 2 / 60]
    assert diagnostics.timestamp_counts(raw, rounded) == {
        "n_pairs_before": 1, "n_zero_before": 0, "n_pairs_after": 2, "n_zero_after": 1,
    }
    pairs = find_next_action_pairs(rounded, keep_zero=True)
    assert pairs.loc[pairs["delta_t"].eq(0), ["source_user", "target_user"]].values.tolist() == [["b", "a"]]
    original = find_next_action_pairs(rounded, keep_zero=False)
    assert original["delta_t"].tolist() == [2 / 60]


@pytest.mark.parametrize("beta", [0.0, 3.7, 10.0])
def test_same_beta_uses_content_degree_and_matches_reference(beta):
    rounded = round_timestamps(events([
        ("b", "x", 0), ("a", "x", 0), ("b", "x", 1), ("a", "x", 2), ("c", "x", 3),
        ("a", "y", 0), ("b", "y", 3),
    ]))
    users = sorted(rounded["user_id"].unique())
    pairs = prepare_action_pairs(rounded, None, keep_zero=False)
    edge_list = original_edge_list(rounded, users)
    result = diagnostics.check_edge_weights(pairs, users, edge_list, beta)
    assert result["n_student_edges"] == result["n_reference_edges"] == 3
    assert result["n_student_only"] == result["n_reference_only"] == 0
    assert result["max_absolute_error"] < 1e-9
    expected = math.exp(-beta / 86400) + math.exp(-beta * 3 / 86400)
    assert diagnostics.reference_weights(edge_list, beta)[("a", "b")] == pytest.approx(expected)


def test_filter_excludes_tie_only_and_weak_edges():
    rounded = round_timestamps(events([
        ("a", "tie", 0.1), ("b", "tie", 0.4), ("a", "strong", 0), ("c", "strong", 1),
        ("b", "weak", 0), ("c", "weak", 604800),
    ]))
    users = ["a", "b", "c"]
    pairs = prepare_action_pairs(rounded, None, keep_zero=False)
    edge_list = original_edge_list(rounded, users)
    unfiltered = diagnostics.reference_weights(edge_list, 10)
    assert unfiltered[("a", "b")] == 0
    assert 0 < unfiltered[("b", "c")] < 1e-9
    result = diagnostics.check_edge_weights(pairs, users, edge_list, 10)
    assert result["n_student_edges"] == result["n_reference_edges"] == 1
    assert result["n_student_only"] == result["n_reference_only"] == 0


def test_filter_is_strict_at_cutoff_and_applies_to_aggregated_weight():
    tolerance = diagnostics.EDGE_TOLERANCE
    pairs = pd.DataFrame([
        ("a", "b", 0.0, tolerance),
        ("b", "c", 0.0, tolerance * 0.75),
        ("b", "c", 0.0, tolerance * 0.75),
    ], columns=PAIR_COLUMNS)
    edge_list = [
        ("a", "b", lambda beta: tolerance, []),
        ("b", "c", lambda beta: tolerance * 1.5, []),
    ]
    result = diagnostics.check_edge_weights(pairs, ["a", "b", "c"], edge_list, beta=0)
    assert result["n_student_edges"] == result["n_reference_edges"] == 1
    assert result["n_student_only"] == result["n_reference_only"] == 0


def test_edge_union_does_not_hide_a_small_one_sided_edge():
    result = diagnostics.compare_weights({("a", "b"): 1e-10}, {("a", "c"): 3e-10})
    assert result["n_student_only"] == result["n_reference_only"] == 1
    assert result["max_absolute_error"] == 3e-10


def test_archive_covers_all_combinations_and_excludes_greedy_placeholders_from_u4():
    config = load_config()
    results, betas = diagnostics.read_archive(config, diagnostics.HISTORY)
    assert len(results) == 2100
    assert len(betas) == 180
    summary, u4 = diagnostics.summarize_results(results, config)
    assert len(summary) == 14
    assert u4.set_index("pipeline")["n_effective_runs"].to_dict() == {"original_reference": 30, "student": 150}
    assert u4["u4_pass"].all()
    with pytest.raises(ValueError, match="Cartesian product"):
        diagnostics.validate_result_keys(results.iloc[1:], config)
    with pytest.raises(ValueError, match="Duplicate"):
        diagnostics.validate_result_keys(pd.concat([results, results.iloc[:1]]), config)
    corrupted = results.copy()
    greedy = corrupted["pipeline"].eq("original_reference") & corrupted["configuration"].eq("3")
    corrupted.loc[corrupted.index[greedy][0], "f1_star"] = -1.0
    with pytest.raises(ValueError, match="placeholders disagree"):
        diagnostics.effective_results(corrupted, config["metrics"])


def test_diagnostics_reproducible_for_all_three_simulations_of_one_seed():
    config = load_config()
    _, betas = diagnostics.read_archive(config, diagnostics.HISTORY)
    seed = config["data_seeds"][0]
    config["data_seeds"] = [seed]
    first = diagnostics.diagnose(config, betas)
    repeated = diagnostics.diagnose(config, betas)
    for before, after in zip(first, repeated):
        pd.testing.assert_frame_equal(before, after, check_exact=True)
        assert set(before["simulation"]) == {1, 2, 3}
        assert set(before["data_seed"]) == {seed}
