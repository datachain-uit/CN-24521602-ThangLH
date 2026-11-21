from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.community.leiden import detect_communities_multiplex
from src.eval.seckin_protocol import (
    CHECKPOINT_KEYS, EpisodeRules, ablation_deltas, ablation_summary, aggregate_episodes, baseline_scores,
    compare_campaigns, daily_coverage, evaluate_ablations, evaluate_baselines, harmonize_time,
    make_checkpoints, make_episodes, observed_at, oracle_top_k, prepare_variant, select_campaign,
    zero_delta_share,
)
from src.graph.time_aware import build_layer_graph_from_pairs, find_next_action_pairs, prepare_action_pairs


START = pd.Timestamp("2020-01-01", tz="UTC")
RULES = EpisodeRules()


def posts(rows):
    return harmonize_time(pd.DataFrame([
        dict(
            post_time=START + pd.Timedelta(days=day, minutes=minute, seconds=31),
            accountid=user, is_control=control, is_repost=True, reposted_postid=content,
            hashtags=["h"], urls=[], account_mentions=[],
        )
        for day, minute, user, control, content in rows
    ]))


@pytest.fixture
def campaign():
    rows = [
        (0, 0, "old_a", False, "p"), (0, 1, "old_b", False, "q"), (0, 0, "old_c", True, "p"),
        (7, 0, "excluded", False, "skip"), (7, 1, "excluded2", False, "skip"),
        (6, 0, "midnight", True, "next_day"),
    ]
    rows += [(5, minute, f"i{i}", False, "p") for i, minute in enumerate([0, 1, 1, 4, 5])]
    rows += [(5, i + 2, f"c{i}", True, "p") for i in range(5)]
    rows += [(5, 5, "i0", False, "other"), (5, 1439, "c5", True, "day_end")]
    rows += [(9, 0, "i0", False, "p"), (9, 1, "i1", False, "r"), (9, 1, "c0", True, "p")]
    rows += [(12, 0, "i0", False, "p"), (12, 1, "i4", False, "r"), (12, 2, "i0", False, "r")]
    rows += [(12, 2, "c0", True, "r"), (12, 7, "after_io", True, "r")]
    frame = posts(rows)
    daily = daily_coverage(frame, RULES)
    stream, episodes = make_episodes(frame, daily, RULES, 1)
    checkpoints = make_checkpoints(stream, episodes, RULES).assign(campaign="test")
    return frame, daily, stream, episodes, checkpoints


def test_p1_preserves_raw_time_and_floors_in_utc():
    source = pd.DataFrame({
        "post_time": ["2020-01-01T07:00:00.001+07:00", "2019-12-31T23:59:59Z"],
        "accountid": ["9223372036854775809", "003"], "is_control": [False, True],
        "is_repost": [False, True],
    })
    frame = harmonize_time(source)
    assert frame.accountid.tolist() == source.accountid.tolist()
    assert frame.post_time.iloc[0] == START + pd.Timedelta(milliseconds=1)
    assert frame.t_min.tolist() == [START, START - pd.Timedelta(minutes=1)]
    assert frame.day.tolist() == [START, START - pd.Timedelta(days=1)]


def test_covered_day_thresholds_count_distinct_and_empty_stream():
    rows = [(0, minute, "i1", False, "r") for minute in range(5)]
    rows += [(0, 0, "c1", True, "r"), (1, 0, "i1", False, "r"), (1, 0, "i2", False, "r")]
    for day, n_ctrl in [(2, 1), (3, 5), (4, 10)]:
        rows += [(day, 0, user, False, "p") for user in ("i1", "i2")]
        rows += [(day, 0, f"c{i}", True, "p") for i in range(n_ctrl)]
    frame = posts(rows)
    daily = daily_coverage(frame, RULES)
    assert daily.loc[START, ["io_accounts", "io_actions"]].tolist() == [1, 5]
    assert daily[["covered_c1", "covered_c5", "covered_c10"]].sum().tolist() == [3, 2, 1]
    small = frame.loc[frame.day.le(START + pd.Timedelta(days=1))]
    stream, episodes = make_episodes(small, daily_coverage(small, RULES), RULES, 1)
    assert stream.empty and episodes.empty
    assert make_checkpoints(stream, episodes, RULES).empty


def test_episode_gap_and_stream_include_unqualified_days_only_if_covered(campaign):
    frame, daily, stream, episodes, _ = campaign
    assert episodes.index.tolist() == [1, 2]
    assert episodes.qualified.tolist() == [False, True]
    assert episodes.covered_days.tolist() == [1, 3]
    assert episodes.span_days.tolist() == [1, 8]
    assert episodes.io_accounts.tolist() == [2, 5]
    assert episodes.io_actions.tolist() == [2, 11]
    assert set(stream.day) == set(daily.index[daily.covered_c1])
    assert set(stream.accountid).isdisjoint({"excluded", "excluded2", "midnight"})
    selection = select_campaign(daily, episodes, RULES, 1)
    assert selection["io_actions_total"] == len(frame.loc[~frame.is_control]) == 15
    assert selection["io_share_qualified"] == pytest.approx(11 / 15)
    assert selection["io_share_without_control"] == pytest.approx(2 / 15)
    assert not selection["in_main_set"]


@pytest.mark.parametrize("n_qualified,total,expected", [(3, 75, True), (3, 76, False), (2, 50, False)])
def test_campaign_selection_requires_both_thresholds(n_qualified, total, expected):
    daily = pd.DataFrame({"io_actions": [total], "ctrl_accounts": [1], "covered_c1": [True]})
    episodes = pd.DataFrame({"qualified": [True] * n_qualified, "io_actions": [5] * n_qualified})
    assert select_campaign(daily, episodes, RULES, 1)["in_main_set"] is expected


def test_checkpoint_ceil_full_minute_and_current_evaluation_population(campaign):
    _, _, stream, _, checkpoints = campaign
    io = checkpoints.loc[checkpoints.checkpoint_type.eq("io_pct")].set_index("checkpoint_value")
    days = checkpoints.loc[checkpoints.checkpoint_type.eq("day")].set_index("checkpoint_value")
    assert io.index.tolist() == [10, 25, 50, 100]
    assert io.loc[25, "tau"] == START + pd.Timedelta(days=5, minutes=1)
    assert io.loc[25, "io_seen"] == 3
    assert io.eligible.tolist() == [False, False, False, True]
    assert days.index.tolist() == [1, 3, 7]
    assert days.loc[1, "tau"] == START + pd.Timedelta(days=5, hours=23, minutes=59)
    assert days.loc[1, "ctrl_seen"] == 6
    assert days.loc[3, "io_seen"] == days.loc[1, "io_seen"]
    history, current = observed_at(stream, io.loc[100])
    assert set(history.accountid) - set(current.accountid) == {"old_a", "old_b", "old_c"}
    assert "after_io" not in current.accountid.tolist()
    assert history.t_min.max() == io.loc[100, "tau"]


def test_p5_scores_are_separate_and_use_the_t10_matching_direction():
    current = posts([(0, 0, "a", False, "p"), (0, 0, "b", True, "p"), (0, 5, "c", False, "q")])
    scores = baseline_scores(current)
    assert scores.loc["a", "zero_delta_repost"] == 0.0
    assert scores.loc["b", "zero_delta_repost"] == 1.0
    assert scores.loc["c", "zero_delta_repost"] == 0.0
    assert scores.loc["a", "first_seen"] < scores.loc["c", "first_seen"]
    assert scores.actions_per_active_day.eq(1).all()
    known = oracle_top_k(pd.Series([1., 2., 3., 4.], index=list("abcd")),
                         pd.Series([1, 1, 0, 0], index=list("abcd")))
    assert [(row["direction"], row["auc"]) for row in known] == [("high", 0), ("low", 1)]
    assert [row["F1_star"] for row in known] == [pytest.approx(2 / 3), 1]


@pytest.mark.parametrize("seed", range(40))
def test_p5_zero_delta_share_matches_t10_pair_enumeration(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 60))
    users = np.array(["10", "9", "a", "b", "B", "003", "3"])[: int(rng.integers(1, 8))]
    events = pd.DataFrame({
        "user_id": pd.array(rng.choice(users, n), dtype="string"),
        "timestamp": rng.integers(0, 6, n) + 26_000_000,
        "action_type": rng.choice(["retweet", "hashtag"], n),
        "content_id": rng.choice(["p", "q", "r"], n),
    })
    pairs = find_next_action_pairs(events, keep_zero=True)
    expected = pairs["delta_t"].eq(0).groupby(pairs["source_user"]).mean() if len(pairs) else pd.Series(dtype=float)
    observed = zero_delta_share(events)
    pd.testing.assert_series_equal(observed.sort_index(), expected.sort_index().astype(float),
                                   check_names=False, check_index_type=False)


def test_p4b_bootstraps_episode_means_and_tests_one_sign_per_campaign():
    rows = []
    for campaign, episode_scores in (("many", [0.2, 0.8, 0.8]), ("one", [0.1])):
        for episode, value in enumerate(episode_scores, 1):
            for seed in (42, 43):
                mark = dict(campaign=campaign, episode=episode, checkpoint_type="io_pct", checkpoint_value=25,
                            tau=START + pd.Timedelta(days=episode), eligible=True, io_actions_episode=episode, seed=seed)
                rows += [{**mark, "method": "a", "F1": value}, {**mark, "method": "b", "F1": 0.3}]
    results = pd.DataFrame(rows)
    summary = aggregate_episodes(results, "F1", seed=42, n_bootstrap=500)
    estimates = summary.loc[summary.campaign.eq("many") & summary.method.eq("a")].set_index("weighting")
    assert estimates.n_episodes.tolist() == [3, 3]
    assert estimates.loc["equal", "estimate"] == pytest.approx(0.6)
    assert estimates.loc["io_actions", "estimate"] == pytest.approx(0.7)
    assert estimates.ci95_low.lt(estimates.ci95_high).all()
    assert summary.loc[summary.campaign.eq("one"), "ci95_low"].isna().all()
    effects, tests = compare_campaigns(results, "F1", "a", "b")
    assert effects.n_paired_episodes.tolist() == [3, 1]
    assert tests[["n_campaigns", "wins", "losses"]].iloc[0].tolist() == [2, 1, 1]


def test_p6_variants_and_scores_ignore_future_suffix(campaign):
    _, _, stream, _, checkpoints = campaign
    mark = checkpoints.query("checkpoint_type == 'day' and checkpoint_value == 1")
    point = SimpleNamespace(tau=mark.tau.iloc[0], episode=mark.episode.iloc[0])
    history, current = observed_at(stream, point)
    shuffled, _ = prepare_variant(history, current, "shuffle_account_day", 1, seed=42)
    assert shuffled.t_min.le(point.tau).all()
    assert history.groupby(["accountid", "day"]).size().equals(shuffled.groupby(["accountid", "day"]).size())
    order = ["accountid", "day", "t_min"]
    assert_frame_equal(history[order].sort_values(order, ignore_index=True),
                       shuffled[order].sort_values(order, ignore_index=True))
    assert not shuffled.t_min.equals(history.t_min)
    assert shuffled.reposted_postid.equals(history.reposted_postid)
    capped, _ = prepare_variant(history, current, "cap_account_day", 1, seed=42)
    assert capped.groupby(["accountid", "day"]).t_min.first().equals(history.groupby(["accountid", "day"]).t_min.min())

    inputs = []

    def adapter(events, nodes, layers, seed):
        assert set(events.columns) == {"user_id", "timestamp", "action_type", "content_id"}
        assert events.timestamp.le(point.tau.timestamp() // 60).all()
        inputs.append((tuple(nodes), tuple(layers), seed))
        graphs = [build_layer_graph_from_pairs(prepare_action_pairs(events, layer), nodes, beta=1.0)
                  for layer in layers]
        return detect_communities_multiplex(graphs, seed=seed)[0]

    replacement = stream.copy()
    replacement.loc[replacement.t_min.gt(point.tau), "reposted_postid"] = "different_future"
    original_scores, clusters = evaluate_ablations(stream, mark, adapter, [42, 43], 1, reset_state=True)
    updated_scores, updated_clusters = evaluate_ablations(replacement, mark, adapter, [42, 43], 1, reset_state=True)
    assert_frame_equal(original_scores.drop(columns="runtime_seconds"), updated_scores.drop(columns="runtime_seconds"))
    assert_frame_equal(clusters, updated_clusters)
    assert_frame_equal(evaluate_baselines(stream, mark), evaluate_baselines(replacement, mark))
    assert len(original_scores) == 10
    assert len(ablation_deltas(original_scores)) == 8
    assert any("old_a" not in nodes for nodes, _, _ in inputs)
    assert any("hashtag" not in layers for _, layers, _ in inputs)
    assert original_scores[CHECKPOINT_KEYS].drop_duplicates().shape[0] == 1
    assert np.isfinite(original_scores[["F1_star", "WP"]].to_numpy()).all()
    assert 0 < len(clusters) <= 10
    assert set(clusters.columns).isdisjoint({"user_id", "accountid", "content_id", "members"})


def test_p6_summary_averages_episodes_within_seed_then_across_seeds():
    rows = [dict(campaign="A", episode=e, checkpoint_type="percent", checkpoint_value=25, tau=START + pd.Timedelta(days=e), method=m, seed=s, F1_star=f, WP=f / 2)
            for e, s, m, f in [(0, 1, "main", .8), (1, 1, "main", .6), (0, 2, "main", .8), (1, 2, "main", .6),
                               (0, 1, "cap_account_day", .7), (1, 1, "cap_account_day", .3), (0, 2, "cap_account_day", .8), (1, 2, "cap_account_day", .6)]]
    summary = ablation_summary(ablation_deltas(pd.DataFrame(rows)))
    row = summary.iloc[0]
    assert len(summary) == 1 and row.method == "cap_account_day" and row.seeds == 2 and row.episodes == 2
    assert row.delta_F1_star_mean == pytest.approx(-.1) and row.sd_delta_F1_star == pytest.approx(np.std([-.2, 0], ddof=1))
    assert row.F1_star_main_mean == pytest.approx(.7) and row.delta_WP_mean == pytest.approx(-.05)
