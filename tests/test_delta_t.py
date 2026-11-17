from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from src.graph.time_aware import matched_delta_t, find_next_action_pairs, build_layer_edges
from time_aware_collaboration.models import _matched_delta_t_list


BASE = pd.Timestamp("2020-01-01")
ATOL = 1e-9

def fmt(minutes):
    return [(BASE + pd.Timedelta(minutes=int(t))).strftime("%Y-%m-%d %H:%M:%S") for t in minutes]


def make_events(rows):
    return pd.DataFrame(rows, columns=["action_type", "content_id", "user_id", "timestamp"])

def reference_edge_weights(events, beta, keep_zero=True):
    weights = {}

    for (_, _), group in events.groupby(["action_type", "content_id"]):
        by_user = {user_id: user_events["timestamp"].to_numpy() for user_id, user_events in group.groupby("user_id")}
        n_users = len(by_user)

        if n_users < 2:
            continue

        for u, v in combinations(sorted(by_user), 2):
            deltas = matched_delta_t(by_user[u], by_user[v], keep_zero=keep_zero)

            if len(deltas) == 0:
                continue

            weight = np.exp(-float(beta) * deltas / 1440.0).sum() / (n_users - 1)

            edge = (u, v)
            weights[edge] = weights.get(edge, 0.0) + float(weight)

    return weights


def pipeline_edge_weights(events, beta, keep_zero=True):
    edges = build_layer_edges(events, action_type="retweet", beta=beta, keep_zero=keep_zero)

    return {(row.source_user, row.target_user): float(row.weight) for row in edges.itertuples(index=False)}


def assert_edge_weights_match(actual, expected):
    assert actual.keys() == expected.keys(), f"Different edge sets: actual={actual.keys()}, expected={expected.keys()}"

    max_error = max((abs(actual[edge] - expected[edge]) for edge in expected), default=0.0)

    assert max_error < ATOL, f"U1 edge weights failed: max_error={max_error:.12g}"

    return max_error

def test_u1_goc_khop_ma_tac_gia():
    rng = np.random.default_rng(0)

    for _ in range(2000):
        tu = rng.integers(0, 30, size=rng.integers(1, 6))
        tv = rng.integers(0, 30, size=rng.integers(1, 6))

        ours = np.sort(matched_delta_t(tu, tv, keep_zero=False))
        original = np.sort(np.asarray(_matched_delta_t_list(fmt(tu), fmt(tv)), dtype=float))

        np.testing.assert_allclose(ours, original, rtol=0, atol=ATOL)

def test_dt0_dem_dung_mot_lan():
    assert sorted(matched_delta_t([0], [0])) == [0]
    assert sorted(matched_delta_t([0, 10], [0, 5], keep_zero=True)) == [0, 5, 5]


def test_b1_goc_loai_delta_t_0():
    result = matched_delta_t([0, 10], [0, 5], keep_zero=False)

    assert sorted(result) == [5, 5]


@pytest.mark.parametrize("keep_zero", [True, False])
def test_empty_input(keep_zero):
    assert len(matched_delta_t([], [1, 2], keep_zero=keep_zero)) == 0
    assert len(matched_delta_t([1, 2], [], keep_zero=keep_zero)) == 0


def test_find_next_action_pairs_zero():
    events = make_events([
        ("retweet", "A", "u", 0),
        ("retweet", "A", "u", 10),
        ("retweet", "A", "v", 0),
        ("retweet", "A", "v", 5),
    ])

    standard = find_next_action_pairs(events, keep_zero=True)
    original = find_next_action_pairs(events, keep_zero=False)

    assert sorted(standard["delta_t"].tolist()) == [0, 5, 5]
    assert sorted(original["delta_t"].tolist()) == [5, 5]


def test_nhieu_hanh_dong_trung_timestamp():
    tu = [0, 0]
    tv = [0]

    standard = matched_delta_t(tu, tv, keep_zero=True)
    original = matched_delta_t(tu, tv, keep_zero=False)

    assert sorted(standard) == [0]
    assert len(original) == 0

@pytest.mark.parametrize("keep_zero", [True, False])
@pytest.mark.parametrize("beta", [0.0, 1.0, 10.0])
def test_u1_edge_weights(beta, keep_zero):
    events = make_events([
        ("retweet", "A", "u", 0),
        ("retweet", "A", "u", 10),
        ("retweet", "A", "v", 0),
        ("retweet", "A", "v", 5),
        ("retweet", "B", "u", 0),
        ("retweet", "B", "v", 0),
        ("retweet", "B", "w", 5),
        ("retweet", "B", "w", 8),
    ])

    expected = reference_edge_weights(events, beta=beta, keep_zero=keep_zero)
    actual = pipeline_edge_weights(events, beta=beta, keep_zero=keep_zero)

    assert_edge_weights_match(actual, expected)


def test_b1_goc_va_chuan_khac_nhau():
    events = make_events([
        ("retweet", "A", "u", 0),
        ("retweet", "A", "v", 0),
    ])

    standard = pipeline_edge_weights(events, beta=1.0, keep_zero=True)
    original = pipeline_edge_weights(events, beta=1.0, keep_zero=False)

    assert set(standard) == {("u", "v")}
    assert np.isclose(standard[("u", "v")], 1.0)
    assert original == {}

@pytest.mark.parametrize("keep_zero", [True, False])
def test_u1_khong_phu_thuoc_thu_tu_du_lieu(keep_zero):
    events = make_events([
        ("retweet", "A", "u", 0),
        ("retweet", "A", "u", 10),
        ("retweet", "A", "v", 0),
        ("retweet", "A", "v", 5),
        ("retweet", "A", "w", 0),
        ("retweet", "A", "w", 5),
    ])

    expected = pipeline_edge_weights(events, beta=1.0, keep_zero=keep_zero)

    for seed in range(100):
        shuffled = events.sample(frac=1, random_state=seed).reset_index(drop=True)
        actual = pipeline_edge_weights(shuffled, beta=1.0, keep_zero=keep_zero)

        assert_edge_weights_match(actual, expected)