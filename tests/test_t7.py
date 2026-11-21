import random

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.metrics import homogeneity_score, normalized_mutual_info_score
from loguru import logger
from time_aware_collaboration.benchmarking.random_labeler import evaluate_random_labelling
from time_aware_collaboration.modularity import calculate_optimal_beta
from time_aware_collaboration.scoring.scores import compute_weighted_precision

from scripts import run_t7
from scripts.run_t6b_experiment import detect_original, generate_data, original_edge_list, round_timestamps
from time_aware_collaboration.models import get_time_damped_edge_list  # noqa: F401  (original_edge_list gọi hàm này)
from src.community import modularity
from src.eval import t7_checkpoints
from src.community.modularity import select_beta_brent
from src.eval.metrics import expected_random_wp, partition_metrics
from pathlib import Path

from src.eval.seckin_protocol import EpisodeRules, daily_coverage, evaluate_ablations, harmonize_time, make_checkpoints, make_episodes, observed_at, oracle_top_k, baseline_scores
from src.eval.seckin_protocol import MODEL_COLUMNS, build_events
from src.eval.t7_checkpoints import EPOCH, RESULT_COLUMNS, TRIV_METHODS, append_result, assert_not_future, build_b1_graphs, checkpoint_pairs, count_true_detections, ds2_job, ds2_trivial_rows, infer_partition, iter_campaign_rows, layer_diagnostics, leiden_multiplex, load_config, observe, random_partition, read_done_keys, result_key, tau_minutes, trivial_rows, variant_pairs
from src.graph.time_aware import PAIR_COLUMNS, build_layer_edges, build_layer_edges_from_pairs, find_next_action_pairs, prepare_action_pairs


# Fixture chép từ tests/test_seckin_protocol.py (T9). Không import chéo vì tên package "tests" trùng với thư mục tests/ của mã tác giả.
START = pd.Timestamp("2020-01-01", tz="UTC")
RULES = EpisodeRules()


def posts(rows):
    return harmonize_time(pd.DataFrame([dict(post_time=START + pd.Timedelta(days=day, minutes=minute, seconds=31), accountid=user, is_control=control, is_repost=True, reposted_postid=content, hashtags=["h"], urls=[], account_mentions=[]) for day, minute, user, control, content in rows]))


@pytest.fixture
def campaign():
    rows = [(0, 0, "old_a", False, "p"), (0, 1, "old_b", False, "q"), (0, 0, "old_c", True, "p"), (7, 0, "excluded", False, "skip"), (7, 1, "excluded2", False, "skip"), (6, 0, "midnight", True, "next_day")]
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


def test_u2_observe_d_tau_gom_dot_truoc_va_trung_thoi_diem(campaign):
    _, _, stream, _, checkpoints = campaign
    for mark in checkpoints.itertuples(index=False):
        history, current = observe(stream, mark.tau, mark.episode)
        assert_frame_equal(history, stream.loc[stream.t_min.le(mark.tau)])
        assert history.t_min.max() <= mark.tau
        assert current.episode.eq(mark.episode).all()
        expected_history, expected_current = observed_at(stream, mark)
        assert_frame_equal(history, expected_history)
        assert_frame_equal(current, expected_current)
    io100 = checkpoints.loc[checkpoints.checkpoint_type.eq("io_pct") & checkpoints.checkpoint_value.eq(100)].iloc[0]
    history, current = observe(stream, io100.tau, io100.episode)
    assert set(history.accountid) - set(current.accountid) == {"old_a", "old_b", "old_c"}


def test_u2_khong_phu_thuoc_phan_tuong_lai(campaign):
    _, _, stream, _, checkpoints = campaign
    for mark in checkpoints.itertuples(index=False):
        future = stream.t_min.gt(mark.tau)
        changed = stream.copy()
        changed.loc[future, "accountid"] = "future_user"
        changed.loc[future, "reposted_postid"] = "future_post"
        extra = changed.tail(1).assign(t_min=stream.t_min.max() + pd.Timedelta(days=30), accountid="later")
        changed = pd.concat([changed, extra])
        assert_frame_equal(observe(changed, mark.tau, mark.episode)[0], observe(stream, mark.tau, mark.episode)[0])


def test_u2_bat_duoc_ro_ri_tuong_lai():
    tau = START + pd.Timedelta(minutes=5)
    assert_not_future([START, tau], tau)
    assert_not_future([], tau)
    with pytest.raises(AssertionError, match="U2"):
        assert_not_future([START, tau + pd.Timedelta(minutes=1)], tau, "graph")
    with pytest.raises(AssertionError, match="U2"):
        assert_not_future(np.array([1919.0, 1920.5]), 1920.0)


def test_observe_yeu_cau_luong_da_sap_xep():
    stream = pd.DataFrame({"timestamp": [2.0, 1.0], "user_id": ["a", "b"]})
    with pytest.raises(ValueError, match="sắp tăng dần"):
        observe(stream, 1.5, time_col="timestamp")


def test_observe_thoi_gian_float_mot_dot():
    stream = pd.DataFrame({"timestamp": [1.0, 1919.5, 1920.0, 1920.0, 1920.5], "user_id": list("abcde")})
    history, current = observe(stream, 1920.0, time_col="timestamp")
    assert history.user_id.tolist() == ["a", "b", "c", "d"]
    assert_frame_equal(current, history)
    assert observe(stream, 0.5, time_col="timestamp")[0].empty


BOUNDS = (0.0, 10.0)
XATOL = 1e-5  # xatol mặc định của minimize_scalar(method="bounded"), mã tác giả không đổi tham số này


def pairs_with_time(events, keep_zero=True):
    """Cặp của prepare_action_pairs kèm cột t = thời điểm hành động đích (muộn hơn) để kiểm U2."""
    return prepare_action_pairs(events, None, keep_zero=keep_zero, with_time=True)


def random_events(seed, n_users=12, n_contents=6, n_events=60, span=600):
    """Sự kiện ngẫu nhiên, thời điểm phút nguyên khác nhau đôi một nên không có Δt = 0."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.choice(span, size=n_events, replace=False)).astype(float)
    users = [f"u{i:02d}" for i in rng.integers(0, n_users, n_events)]
    contents = [f"c{i}" for i in rng.integers(0, n_contents, n_events)]
    return pd.DataFrame({"user_id": users, "timestamp": times, "action_type": "a", "content_id": contents, "is_inauthentic": False})


def author_beta(events, seed):
    """calculate_optimal_beta của tác giả trên đồ thị hai phía dựng như T6b; random.seed cố định Louvain không seed."""
    events = round_timestamps(events)
    edge_list = original_edge_list(events, sorted(events["user_id"].unique()))
    random.seed(seed)
    return float(calculate_optimal_beta(edge_list, bounds=BOUNDS)), edge_list


INTERIOR = {(600, 4), (2 * 1440, 1), (2 * 1440, 4)}  # các ca có beta tối ưu nằm trong (0, 10), không ở biên


@pytest.mark.parametrize("span,data_seed", [(span, seed) for span in (600, 2 * 1440) for seed in range(5)])
def test_brent_khop_calculate_optimal_beta_cua_tac_gia(span, data_seed):
    events = random_events(data_seed, span=span)
    pairs = pairs_with_time(events, keep_zero=False)
    ref, edge_list = author_beta(events, 42)
    beta, q, nfev = select_beta_brent(pairs, events["timestamp"].max(), BOUNDS, 42)
    assert build_layer_edges_from_pairs(pairs, None, BOUNDS[1]).weight.min() > 1e-9
    ours = build_layer_edges_from_pairs(pairs, None, beta)
    assert {(u, v) for u, v, _, _ in edge_list} == set(zip(ours.source_user, ours.target_user))
    assert abs(beta - ref) <= XATOL
    assert nfev > 0 and np.isfinite(q)
    if (span, data_seed) in INTERIOR:
        assert 0.5 < beta < 9.9


def test_brent_nguong_1e9_va_tolerance_0():
    """Δt nhiều ngày: ngưỡng 1e-9 bỏ cạnh ở beta lớn (khác tác giả); tolerance=0 cho cùng tập cạnh và khớp tác giả."""
    events = random_events(7, span=20 * 1440)
    pairs = pairs_with_time(events, keep_zero=False)
    weights = build_layer_edges_from_pairs(pairs, None, BOUNDS[1]).weight
    assert (weights <= 1e-9).any()
    ref, _ = author_beta(events, 42)
    beta, _, _ = select_beta_brent(pairs, events["timestamp"].max(), BOUNDS, 42, tolerance=0.0)
    assert abs(beta - ref) <= XATOL
    assert 0.5 < beta < 9.9


def test_brent_don_vi_phut_cho_timestamp_va_float():
    start = pd.Timestamp("2020-01-01 08:00", tz="UTC")
    offsets = random_events(3).timestamp
    ds2 = random_events(3)
    ds1 = ds2.assign(timestamp=((start + pd.to_timedelta(offsets, unit="min")) - EPOCH) // pd.Timedelta(minutes=1))
    tau_ds1 = start + pd.Timedelta(minutes=float(offsets.max()))
    assert tau_minutes(tau_ds1) == ds1.timestamp.max()
    pairs_ds1, pairs_ds2 = pairs_with_time(ds1), pairs_with_time(ds2)
    assert np.allclose(pairs_ds1.dd, pairs_ds2.dd, atol=1e-12)
    half_day = pairs_with_time(pd.DataFrame({"user_id": ["a", "b"], "timestamp": [0.0, 720.0], "action_type": "a", "content_id": "c"}))
    assert half_day.dd.tolist() == [0.5]
    assert select_beta_brent(pairs_ds1, tau_ds1, BOUNDS, 42) == select_beta_brent(pairs_ds2, float(offsets.max()), BOUNDS, 42)


def test_brent_u2_bat_cap_tuong_lai():
    events = random_events(1)
    pairs = pairs_with_time(events)
    with pytest.raises(AssertionError, match="select_beta_brent"):
        select_beta_brent(pairs, events["timestamp"].max() - 1, BOUNDS, 42)
    with pytest.raises(ValueError, match="cột t"):
        select_beta_brent(pairs.drop(columns="t"), events["timestamp"].max(), BOUNDS, 42)


def test_brent_lop_rong_tra_nan_khong_goi_louvain(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Không được gọi Louvain cho lớp rỗng.")

    monkeypatch.setattr(modularity, "_louvain_modularity", fail)
    alone = pd.DataFrame({"user_id": ["a", "b"], "timestamp": [0.0, 5.0], "action_type": "a", "content_id": ["x", "y"]})
    beta, q, nfev = select_beta_brent(pairs_with_time(alone), 5.0, BOUNDS, 42)
    assert np.isnan(beta) and np.isnan(q) and nfev == 0


def test_brent_tai_lap_theo_seed():
    pairs = pairs_with_time(random_events(2))
    assert select_beta_brent(pairs, 600.0, BOUNDS, 42) == select_beta_brent(pairs, 600.0, BOUNDS, 42)


LAYERS = ("retweet", "hashtag")


def tied_events(seed, n_events=80):
    """Hai lớp, phút nguyên trong khoảng hẹp nên có nhiều Δt = 0; thêm một cặp chỉ có Δt = 0 và một tài khoản cô lập."""
    rng = np.random.default_rng(seed)
    rows = pd.DataFrame({"user_id": [f"u{i:02d}" for i in rng.integers(0, 10, n_events)], "timestamp": rng.integers(0, 40, n_events).astype(float), "action_type": rng.choice(LAYERS, n_events), "content_id": [f"c{i}" for i in rng.integers(0, 4, n_events)]})
    extra = pd.DataFrame({"user_id": ["z1", "z2", "solo"], "timestamp": [50.0, 50.0, 60.0], "action_type": ["retweet", "retweet", "hashtag"], "content_id": ["tie_only", "tie_only", "alone"]})
    return pd.concat([rows, extra], ignore_index=True).sort_values("timestamp", kind="stable", ignore_index=True)


def test_with_time_mac_dinh_giu_pair_columns():
    events = tied_events(0)
    plain = prepare_action_pairs(events, "retweet")
    timed = prepare_action_pairs(events, "retweet", with_time=True)
    raw = find_next_action_pairs(events.loc[events.action_type.eq("retweet")])
    assert plain.columns.tolist() == PAIR_COLUMNS
    assert_frame_equal(timed[PAIR_COLUMNS], plain)
    assert np.array_equal(timed["t"].to_numpy(), raw["target_time"].to_numpy(dtype=float))
    assert prepare_action_pairs(events.iloc[:0], "retweet", with_time=True).columns.tolist() == [*PAIR_COLUMNS, "t"]


@pytest.mark.parametrize("seed", range(4))
def test_b1_goc_loc_lai_trung_keep_zero_false(seed):
    events = tied_events(seed)
    for layer, pairs in checkpoint_pairs(events, LAYERS, events.timestamp.max()).items():
        assert pairs["dd"].eq(0).any()
        assert_frame_equal(variant_pairs(pairs, True), prepare_action_pairs(events, layer, keep_zero=True, with_time=True))
        assert_frame_equal(variant_pairs(pairs, False), prepare_action_pairs(events, layer, keep_zero=False, with_time=True))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("keep_zero", [True, False])
@pytest.mark.parametrize("beta", [0.0, 0.7, 5.0])
def test_u1_tai_dung_cap_khop_tinh_lai(seed, keep_zero, beta):
    events = tied_events(seed)
    for layer, pairs in checkpoint_pairs(events, LAYERS, events.timestamp.max()).items():
        reused = build_layer_edges_from_pairs(variant_pairs(pairs, keep_zero), layer, beta).set_index(["source_user", "target_user"])["weight"]
        fresh = build_layer_edges(events, layer, beta, keep_zero=keep_zero).set_index(["source_user", "target_user"])["weight"]
        assert reused.index.equals(fresh.index)
        assert np.max(np.abs(reused.to_numpy() - fresh.to_numpy()), initial=0.0) < 1e-9


def test_checkpoint_pairs_u2():
    events = tied_events(0)
    with pytest.raises(AssertionError, match="checkpoint_pairs"):
        checkpoint_pairs(events, LAYERS, events.timestamp.max() - 1)


@pytest.mark.parametrize("seed", range(4))
def test_dem_canh_khop_cach_dem_cua_tac_gia(seed):
    """(i) cặp chỉ có Δt = 0 = số cạnh trọng số 0 của tác giả; (iii) số cạnh tác giả = len(get_time_damped_edge_list)."""
    events = tied_events(seed)
    for layer, pairs in checkpoint_pairs(events, LAYERS, events.timestamp.max()).items():
        part = round_timestamps(events.loc[events.action_type.eq(layer)])
        edge_list = original_edge_list(part, sorted(part.user_id.unique()))
        used = variant_pairs(pairs, False)
        found = layer_diagnostics(pairs, used, nx_graph(used))
        assert found["edges_author"] == len(edge_list)
        assert found["zero_only_pairs"] == sum(weight_fn(0.0) == 0 for _, _, weight_fn, _ in edge_list)
        assert found["edges_author"] - found["zero_only_pairs"] == found["edges_before_tolerance"]
    assert found["dropped_by_tolerance"] == 0


def nx_graph(pairs):
    edges = build_layer_edges_from_pairs(pairs, None, 0.0)
    graph = nx.Graph()
    graph.add_weighted_edges_from(zip(edges.source_user, edges.target_user, edges.weight))
    return graph


def test_build_b1_graphs_hai_bien_the_va_lop_rong():
    events = tied_events(1)
    tau = events.timestamp.max()
    pairs = checkpoint_pairs(events, (*LAYERS, "url"), tau)
    nodes = sorted(events.user_id.unique())
    graphs = {}
    for keep_zero in (True, False):
        layer_graphs, info = build_b1_graphs(pairs, nodes, keep_zero, tau, BOUNDS, 42)
        graphs[keep_zero] = layer_graphs
        assert [g.number_of_nodes() for g in layer_graphs] == [len(nodes)] * 3
        assert list(info) == [*LAYERS, "url"]
        assert np.isnan(info["url"]["beta"]) and info["url"]["nfev"] == 0 and layer_graphs[2].number_of_edges() == 0
        for layer in LAYERS:
            row = info[layer]
            assert row["edges"] == layer_graphs[LAYERS.index(layer)].number_of_edges()
            assert row["at_bound"] == (row["beta"] >= BOUNDS[1] - XATOL)
        assert info["retweet"]["zero_only_pairs"] >= 1
    assert graphs[True][0].has_edge("z1", "z2") and not graphs[False][0].has_edge("z1", "z2")
    assert graphs[True][1].degree("solo") == 0


def test_build_b1_graphs_u2_va_nut_ngoai_d_tau():
    events = tied_events(2)
    tau = events.timestamp.max()
    pairs = checkpoint_pairs(events, LAYERS, tau)
    latest_pair = max(part["t"].max() for part in pairs.values())
    assert latest_pair < tau  # sự kiện cuối (solo) không tạo cặp, nên phải đặt tau dưới cặp muộn nhất
    with pytest.raises(AssertionError, match="build_b1_graphs"):
        build_b1_graphs(pairs, sorted(events.user_id.unique()), True, latest_pair - 1, BOUNDS, 42)
    with pytest.raises(ValueError, match="ngoài tập nút"):
        build_b1_graphs(pairs, ["u00"], True, tau, BOUNDS, 42)



# --- U3: chỉ số khớp mã tác giả trên cùng phân hoạch ---
# Chép logic từ time_aware_collaboration/scoring/scores.py:1684-1772 (score_time_aware_fourplex), commit bae7b15b.
# MIT License, Copyright (c) 2025 letiziaia (references/time-aware-collaboration/LICENSE).
# Chỉ gộp mỗi câu lệnh về một dòng và bỏ logger; không đổi logic, kể cả tie-break "f1 > best_f1" và share cho cụm 1 phần tử.
def author_inline_scores(community_dict, inauthentic):
    gt_labels, comm_labels, cluster_share, comm_share_labels, cluster_labels_to_node_ids = [], [], [], [], {}
    best_p, best_r, best_f1 = 0.0, 0.0, 0.0
    for comm_id in range(max(community_dict.values()) + 1):
        community_nodes = set(node for node, c_id in community_dict.items() if c_id == comm_id)
        cluster_labels_to_node_ids[comm_id] = list(community_nodes)
        relevant_retrieved = community_nodes.intersection(inauthentic)
        precision = len(relevant_retrieved) / len(community_nodes) if len(community_nodes) > 0 else 0
        recall = len(relevant_retrieved) / len(inauthentic) if len(inauthentic) > 0 else 0
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            if f1 > best_f1:
                best_p, best_r, best_f1 = precision, recall, f1
        comm_labels.extend([comm_id] * len(community_nodes))
        gt_labels.extend([1 if node in inauthentic else 0 for node in community_nodes])
        share = len(relevant_retrieved) / len(community_nodes) if len(community_nodes) > 1 else 0
        cluster_share.append(share)
        comm_share_labels.extend([1] * len(community_nodes) if share >= 0.5 else [0] * len(community_nodes))
    q = compute_weighted_precision(cluster_labels_to_node_ids, inauthentic)
    return dict(f1_star=best_f1, precision_star=best_p, recall_star=best_r, wp=q, homogeneity=homogeneity_score(gt_labels, comm_labels), nmi=normalized_mutual_info_score(gt_labels, comm_labels), nmi50=normalized_mutual_info_score(gt_labels, comm_share_labels))


def assert_u3(partition, population, truth):
    restricted = [c & population for c in partition if c & population]
    ref = author_inline_scores({node: i for i, c in enumerate(restricted) for node in c}, list(truth & population))
    ours = partition_metrics(partition, population, truth)
    assert ours.keys() == ref.keys()
    for key in ref:
        assert abs(ours[key] - ref[key]) < 1e-9, (key, ours[key], ref[key])
    return ours


def random_partition_fixture(seed, n_nodes=30, n_io=8):
    rng = np.random.default_rng(seed)
    nodes = [f"n{i:02d}" for i in range(n_nodes)]
    labels = rng.integers(0, int(rng.integers(1, 9)), n_nodes)
    partition = [set(np.array(nodes)[labels == k]) for k in np.unique(labels)]
    partition += [{f"s{i}"} for i in range(int(rng.integers(0, 4)))]
    truth = set(rng.choice(nodes, n_io, replace=False))
    return partition, set().union(*partition), truth


@pytest.mark.parametrize("seed", range(50))
def test_u3_chi_so_khop_ma_tac_gia_phan_hoach_ngau_nhien(seed):
    partition, nodes, truth = random_partition_fixture(seed)
    assert_u3(partition, nodes, truth)
    population = set(sorted(nodes)[::2])
    assert_u3(partition, population, truth)


def test_u3_hoa_f1_lay_cum_dau_tien():
    truth = {f"io{i}" for i in range(8)}
    small = {"io0", "io1"}
    large = {"io2", "io3", "io4", "io5", *{f"c{i}" for i in range(8)}}
    rest = [{"io6"}, {"io7"}, {"c8"}]
    nodes = small | large | set().union(*rest)
    first = assert_u3([small, large, *rest], nodes, truth)
    second = assert_u3([large, small, *rest], nodes, truth)
    assert first["f1_star"] == second["f1_star"] == pytest.approx(0.4)
    assert (first["precision_star"], first["recall_star"]) == (1.0, 0.25)
    assert (second["precision_star"], second["recall_star"]) == (pytest.approx(1 / 3), 0.5)


@pytest.mark.parametrize("truth", [set(), {"a", "b", "c", "d"}])
def test_u3_khong_co_io_va_toan_io(truth):
    partition = [{"a", "b"}, {"c"}, {"d"}]
    assert_u3(partition, {"a", "b", "c", "d"}, truth)


# --- Leiden đa lớp ---
def b1_layers(seed):
    events = tied_events(seed)
    tau = events.timestamp.max()
    nodes = sorted(events.user_id.unique())
    graphs, info = build_b1_graphs(checkpoint_pairs(events, LAYERS, tau), nodes, True, tau, BOUNDS, 42)
    return graphs, info, nodes


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("community_seed", [42, 46])
def test_leiden_multiplex_trung_detect_original_cua_t6b(seed, community_seed):
    graphs, _, nodes = b1_layers(seed)
    assert all(g.number_of_edges() for g in graphs)
    ours = leiden_multiplex(graphs, nodes, community_seed, [g.number_of_edges() for g in graphs])
    ref = detect_original(graphs, nodes, community_seed)
    assert sorted(map(sorted, ours)) == sorted(map(sorted, ref))
    assert sorted(node for c in ours for node in c) == nodes


def test_leiden_multiplex_tai_lap_lop_rong_va_layer_weights():
    graphs, info, nodes = b1_layers(1)
    weights = [info[layer]["edges_author"] for layer in LAYERS]
    assert leiden_multiplex(graphs, nodes, 42, weights) == leiden_multiplex(graphs, nodes, 42, weights)
    empty = [nx.empty_graph(nodes) for _ in LAYERS]
    assert leiden_multiplex(empty, nodes, 42, [0, 0]) == [{node} for node in nodes]
    with pytest.raises(ValueError):
        leiden_multiplex(graphs, nodes, 42, weights[:1])


def test_infer_partition_lam_adapter_p6_cua_t9(campaign):
    _, _, stream, _, checkpoints = campaign
    cfg = load_config()
    scores, _ = evaluate_ablations(stream, checkpoints, lambda events, nodes, layers, seed: infer_partition(events, nodes, layers, seed, cfg=cfg), (42,), cap_k=3)
    assert len(scores) and scores.F1_star.between(0, 1).all()


def test_leiden_cung_tap_dinh_va_thu_tu_moi_lop_brent_chi_gom_canh(monkeypatch):
    """a_only có cạnh ở retweet nhưng cô lập ở hashtag; quiet có trong D(tau) nhưng không có sự kiện ở lớp nào."""
    events = pd.DataFrame({"user_id": ["a_only", "b", "c", "a_only", "b", "c"], "timestamp": [0.0, 3.0, 5.0, 10.0, 12.0, 14.0], "action_type": ["retweet", "retweet", "retweet", "hashtag", "hashtag", "hashtag"], "content_id": ["r", "r", "r", "h_alone", "h", "h"]})
    nodes = ["a_only", "b", "c", "quiet"]
    brent_graphs, leiden_layers = [], []
    real_louvain, real_leiden = modularity._louvain_modularity, t7_checkpoints.la.find_partition_multiplex

    def spy_louvain(graph, rng):
        brent_graphs.append(graph.copy())
        return real_louvain(graph, rng)

    def spy_leiden(layers, *args, **kwargs):
        leiden_layers.extend(layer.copy() for layer in layers)
        return real_leiden(layers, *args, **kwargs)

    monkeypatch.setattr(modularity, "_louvain_modularity", spy_louvain)
    monkeypatch.setattr(t7_checkpoints.la, "find_partition_multiplex", spy_leiden)
    graphs, info = build_b1_graphs(checkpoint_pairs(events, LAYERS, 14.0), nodes, True, 14.0, BOUNDS, 42)
    assert [sorted(g.nodes()) for g in graphs] == [nodes, nodes]
    assert graphs[0].degree("a_only") > 0 and graphs[1].degree("a_only") == 0 and graphs[1].degree("quiet") == 0
    assert all("quiet" not in g for g in brent_graphs) and any(set(g.nodes()) == {"b", "c"} for g in brent_graphs)
    partition = leiden_multiplex(graphs, nodes, 42, [info[layer]["edges_author"] for layer in LAYERS])
    assert len(leiden_layers) == 2 and all(layer.vcount() == len(nodes) for layer in leiden_layers)
    for layer, graph in zip(leiden_layers, graphs):
        assert {frozenset((nodes[e.source], nodes[e.target])) for e in layer.es} == {frozenset(edge) for edge in graph.edges()}
    assert sorted(node for cluster in partition for node in cluster) == nodes



# --- B0 và S2 ---
S2 = load_config()["s2"]
S2_SEED = load_config()["community_seeds"][0]


def test_b0_random_partition_tai_lap_va_phu_du():
    nodes = [f"n{i:02d}" for i in range(46)]
    for k in (2, 3):
        partition = random_partition(reversed(nodes), k, 42)
        assert partition == random_partition(nodes, k, 42) and partition != random_partition(nodes, k, 43)
        assert len(partition) <= k and sorted(node for c in partition for node in c) == nodes
    assert random_partition(["a"], 3, 42) == [{"a"}]


@pytest.fixture
def quiet_author_logger():
    logger.disable("time_aware_collaboration")
    yield
    logger.enable("time_aware_collaboration")


@pytest.mark.parametrize("k", [2, 3])
def test_s2a_random_labeler_cua_tac_gia_tai_hien_052_036(k, quiet_author_logger):
    """S2(a): evaluate_random_labelling (random_labeler.py:7-84) cho WP ≈ 1/K vì p_k = tp/(tp+fn) là recall của control (dòng 18-20, 61-75)."""
    np.random.seed(S2_SEED)
    wp = np.mean([evaluate_random_labelling(S2["n_inauthentic"], S2["n_authentic"], k)[0] for _ in range(S2["repetitions"])])
    assert abs(wp - S2["author_wp"][str(k)]) <= S2["tolerance_author"], wp


@pytest.mark.parametrize("n_io,n_ctrl", [(S2["n_inauthentic"], S2["n_authentic"]), (50, 50)])
@pytest.mark.parametrize("k", [2, 3])
def test_s2b_b0_wp_chuan_khop_ky_vong_giai_tich(n_io, n_ctrl, k):
    """S2(b): B0 của ta với WP chuẩn (scores.py:58-83) khớp E[WP] = π + (1 − π)(K − 1)/(N − 1).

    Sai lệch nhỏ còn lại do cụm rỗng/cụm 1 phần tử (p = 0) mà công thức bỏ qua, không đáng kể ở cỡ N này.
    """
    nodes = [f"n{i:03d}" for i in range(n_io + n_ctrl)]
    truth = set(nodes[:n_io])
    wp = np.mean([partition_metrics(random_partition(nodes, k, S2_SEED + r), set(nodes), truth)["wp"] for r in range(S2["repetitions"])])
    assert abs(wp - expected_random_wp(n_io, n_ctrl, k)) <= S2["tolerance_standard"], (wp, expected_random_wp(n_io, n_ctrl, k))


def test_expected_random_wp_cong_thuc_va_ca_bien():
    assert expected_random_wp(6, 40, 1) == pytest.approx(6 / 46)
    assert expected_random_wp(6, 40, 2) == pytest.approx(6 / 46 + (40 / 46) / 45)
    assert expected_random_wp(50, 50, 3) == pytest.approx(0.5 + 0.5 * 2 / 99)
    assert expected_random_wp(0, 40, 3) == 0.0
    assert expected_random_wp(5, 0, 3) == 1.0
    for args in ((1, 0, 2), (3, 3, 0), (-1, 5, 2)):
        with pytest.raises(ValueError):
            expected_random_wp(*args)



# --- F1* tầm thường ---
def eligible_marks(campaign):
    _, _, stream, _, checkpoints = campaign
    return stream, checkpoints.loc[checkpoints.eligible]


def labels_of(current):
    return ~current.groupby("accountid")["is_control"].first()


def test_trivial_rows_khop_p5_cua_t9(campaign):
    stream, marks = eligible_marks(campaign)
    assert len(marks)
    """Tham chiếu: oracle_top_k của T9 (không sửa) chạy trên cùng điểm số baseline_scores, với ID đã đổi bí danh theo triv_tiebreak_seed."""
    for mark in marks.itertuples(index=False):
        _, current = observe(stream, mark.tau, mark.episode)
        rows = trivial_rows(current, labels_of(current), mark.tau)
        assert [row["method"] for row in rows] == [f"TRIV_{m}_{d}" for m in ("first_seen", "actions_per_active_day", "zero_delta_repost") for d in ("high", "low")] + ["TRIV_best"]
        scores, labels = baseline_scores(current), labels_of(current)
        alias = t7_checkpoints._alias_ids(scores.index, CFG["triv_tiebreak_seed"])
        ref = pd.DataFrame([dict(method=method, **found) for method, score in scores.items() for found in oracle_top_k(score.rename(index=alias), labels.reindex(list(alias)).rename(index=alias))])
        for row in rows[:-1]:
            method, direction = row["method"][5:].rsplit("_", 1)
            hit = ref.loc[ref.method.eq(method) & ref.direction.eq(direction)].iloc[0]
            assert abs(row["f1_star"] - hit.F1_star) < 1e-12 and row["oracle_k"] == hit.oracle_k
            p, r = row["precision_star"], row["recall_star"]
            assert abs(row["f1_star"] - (2 * p * r / (p + r) if p + r else 0.0)) < 1e-12
        assert rows[-1]["f1_star"] == max(row["f1_star"] for row in rows[:-1]) == ref.F1_star.max()


def test_trivial_rows_u2(campaign):
    stream, marks = eligible_marks(campaign)
    mark = marks.iloc[-1]
    _, current = observe(stream, mark.tau, mark.episode)
    with pytest.raises(AssertionError, match="trivial_rows"):
        trivial_rows(current, labels_of(current), current.t_min.max() - pd.Timedelta(minutes=1))


def test_trivial_rows_diem_bang_nhau_khong_phu_thuoc_thu_tu_dong_va_lay_k_nho_nhat():
    """a(IO), b, c, d(IO) cùng điểm ở mọi điểm số: thứ tự là thứ tự bí danh; k* phải là k nhỏ nhất đạt F1 lớn nhất trên thứ tự đó."""
    current = posts([(0, 0, user, control, f"p_{user}") for user, control in (("a", False), ("b", True), ("c", True), ("d", False))])
    rows = trivial_rows(current, labels_of(current), current.t_min.max())
    alias = t7_checkpoints._alias_ids(["a", "b", "c", "d"], CFG["triv_tiebreak_seed"])
    y = np.array([0 if account in ("b", "c") else 1 for account in sorted(alias, key=alias.get)])
    f1 = 2 * np.cumsum(y) / (np.arange(1, 5) + 2)
    k = int(np.argmax(f1)) + 1
    assert f1[k - 1] == f1.max() and not (f1[:k - 1] == f1.max()).any()
    for row in rows:
        assert row["oracle_k"] == k and row["f1_star"] == pytest.approx(f1.max()) and row["precision_star"] == pytest.approx(np.cumsum(y)[k - 1] / k)
    for seed in range(5):
        shuffled = current.sample(frac=1, random_state=seed)
        assert trivial_rows(shuffled, labels_of(shuffled), current.t_min.max()) == rows


def test_trivial_rows_dao_dong_tren_fixture_t9(campaign):
    stream, marks = eligible_marks(campaign)
    mark = marks.iloc[-1]
    _, current = observe(stream, mark.tau, mark.episode)
    expected = trivial_rows(current, labels_of(current), mark.tau)
    for seed in range(5):
        shuffled = current.sample(frac=1, random_state=seed)
        assert trivial_rows(shuffled, labels_of(shuffled), mark.tau) == expected


# --- Phần 8: vòng lặp, schema CSV và append ---
CFG = load_config()
SEEDS = (42, 43)
EXPECTED_COLUMNS = ["campaign", "episode", "q", "tau", "method", "seed", "n_io", "n_ctrl", "eligible", "f1_star", "precision_star", "recall_star", "wp", "homogeneity", "nmi", "nmi50", "runtime_s", "beta_retweet", "beta_hashtag", "beta_mention", "beta_url", "beta_sim1", "beta_sim2", "beta_sim3"]


def ds1_marks(campaign):
    _, _, stream, _, checkpoints = campaign
    marks = checkpoints.loc[checkpoints.checkpoint_type.eq("io_pct"), ["episode", "q", "tau", "n_io", "n_ctrl", "eligible"]].reset_index(drop=True)
    return stream, marks


def run_rows(stream, marks, done=frozenset()):
    events_fn = lambda history: build_events(history[MODEL_COLUMNS], tuple(CFG["ds1_layers"]))  # noqa: E731
    return list(iter_campaign_rows(stream, marks, events_fn, "test", CFG, CFG["ds1_layers"], SEEDS, done=done, progress=lambda _: None))


def write_csv(path, rows):
    for row in rows:
        append_result(path, row)


def test_schema_24_cot_dung_thu_tu_va_khong_co_accountid(campaign, tmp_path):
    stream, marks = ds1_marks(campaign)
    path = tmp_path / "t7_results.csv"
    write_csv(path, run_rows(stream, marks))
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert RESULT_COLUMNS == EXPECTED_COLUMNS and list(frame.columns) == EXPECTED_COLUMNS
    accounts = set(stream.accountid.astype(str))
    assert not accounts & set(frame.columns) and not accounts & set(frame.to_numpy().ravel())
    with pytest.raises(ValueError, match="sai schema"):
        append_result(path, {**frame.iloc[0].to_dict(), "accountid": "i0"})


def test_moc_khong_dat_moi_method_mot_dong_nan_va_moc_dat_du_dong(campaign):
    stream, marks = ds1_marks(campaign)
    rows = pd.DataFrame(run_rows(stream, marks))
    model_methods = ["B0_K2", "B0_K3", "B1_goc", "B1_chuan"]
    for (q, eligible), part in rows.groupby(["q", "eligible"]):
        if eligible:
            assert len(part) == len(model_methods) * len(SEEDS) + len(TRIV_METHODS)
            assert part.loc[part.method.isin(model_methods), "f1_star"].notna().all()
            assert part.loc[part.method.str.startswith("B1_"), "beta_retweet"].notna().all() and part.beta_sim1.isna().all()
            assert part.loc[part.method.str.startswith("TRIV_"), "seed"].isna().all()
        else:
            assert sorted(part.method) == sorted(model_methods + TRIV_METHODS) and part.seed.isna().all()
            assert part[["f1_star", "wp", "nmi", "nmi50", "runtime_s"]].isna().all().all()


def test_moc_pre_t0_chay_b0_b1_triv_nan(campaign):
    stream, marks = ds1_marks(campaign)
    mark = marks.loc[~marks.eligible].iloc[[-1]].assign(pre_t0=True)
    rows = pd.DataFrame(run_rows(stream, mark))
    models = rows.loc[~rows.method.str.startswith("TRIV_")]
    assert len(models) == 4 * len(SEEDS) and models.f1_star.notna().all() and not rows.eligible.any()
    assert rows.loc[rows.method.str.startswith("TRIV_"), "f1_star"].isna().all()


def test_append_ghi_doc_lai_chay_lai_khong_them_dong(campaign, tmp_path):
    stream, marks = ds1_marks(campaign)
    path = tmp_path / "t7_results.csv"
    write_csv(path, run_rows(stream, marks))
    n_rows = len(pd.read_csv(path))
    assert run_rows(stream, marks, read_done_keys(path)) == []
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["seed"] = frame.seed.map(lambda s: f"{float(s):.1f}" if s else "nan")
    frame["episode"] = frame.episode.map(lambda e: f"{float(e):.1f}")
    frame["q"] = frame.q.map(lambda q: str(float(q)) if q else "")
    frame["tau"] = frame.tau.map(lambda t: pd.Timestamp(t).isoformat())
    frame.to_csv(path, index=False)
    assert "42.0" in set(frame.seed) and "nan" in set(frame.seed) and frame.tau.str.contains(r"\+00:00").all()
    assert len(read_done_keys(path)) == n_rows
    assert run_rows(stream, marks, read_done_keys(path)) == []
    frame.drop(index=frame.index[frame.method.eq("B1_goc") & frame.seed.eq("42.0")][0]).to_csv(path, index=False)
    redo = run_rows(stream, marks, read_done_keys(path))
    assert [(row["method"], row["seed"]) for row in redo] == [("B1_goc", SEEDS[0])]


def test_result_key_chuan_hoa_nan_seed_episode_q_tau():
    base = dict(campaign="c", episode=57, q=0.25, tau=pd.Timestamp("2019-01-24 10:31", tz="UTC"), method="B1_goc", seed=42)
    same = [dict(base, episode=57.0, seed=42.0), dict(base, episode="57.0", seed="42.0", q="0.25"), dict(base, q="0.2500", tau="2019-01-24T10:31:00+00:00"), dict(base, tau="2019-01-24T10:31:00Z"), dict(base, tau=pd.Timestamp("2019-01-24 11:31", tz="Europe/Madrid"))]
    assert all(result_key(row) == result_key(base) for row in same)
    missing = dict(base, q=float("nan"), seed=float("nan"), tau=1919.0)
    assert all(result_key(dict(missing, q=q, seed=s, tau=t)) == result_key(missing) for q, s, t in [(None, "", "1919"), ("nan", "nan", "1919.000000"), ("", None, 1919)])
    assert result_key(dict(base, seed=43)) != result_key(base) and result_key(dict(base, tau=1919.5)) != result_key(dict(base, tau=1919.0))


def test_run_json_va_thu_tu_moc_uu_tien(campaign, tmp_path):
    stream, marks = ds1_marks(campaign)
    ordered = run_t7.ordered_marks(marks, "Catalonia", CFG)
    assert np.allclose(ordered.q.iloc[:marks.q.eq(0.25).sum()], 0.25) and len(ordered) == len(marks)
    assert run_t7.ordered_marks(marks, "Iran_5", CFG).equals(marks)
    path = tmp_path / "t7_results.csv"
    write_csv(path, run_rows(stream, marks))
    args = run_t7.parse_args(["--campaigns", "Iran_5", "--max-checkpoints", "3", "--seeds", "42"], CFG)
    assert args.campaigns == ["Iran_5"] and args.max_checkpoints == 3 and args.seeds == [42]
    assert run_t7.parse_args([], CFG).campaigns == CFG["run_order"]
    run_path = tmp_path / "run.json"
    run_t7.write_run_json(run_path, CFG, args, {"test": 1.5}, path, "2026-11-20T00:00:00+00:00")
    run_t7.write_run_json(run_path, CFG, args, {"test": 2.0}, path, "2026-11-20T00:00:00+00:00")
    payload = __import__("json").loads(run_path.read_text(encoding="utf-8"))
    assert payload["runtime_s_by_campaign"] == {"test": 3.5} and payload["git_head"] and "code_dirty" in payload
    assert set(payload["notes"]) >= {"layer_weights", "zero_only_pairs", "b0", "s3_ds2", "ds2_l3", "triv_zero_delta_ds2"}
    assert {(row["layer"], row["variant"]) for row in payload["beta_at_bound"]} == {(layer, variant) for layer in ("retweet", "hashtag") for variant in ("B1_goc", "B1_chuan")}
    assert payload["packages"]["time-aware-collaboration"] == "0.1.0"
    assert len(payload["git_diff_sha256"]) == 64 and all(len(h) == 64 for h in payload["untracked_sha256"].values())


def test_beta_bound_table_dem_bien_tren_va_bien_duoi(tmp_path):
    path = tmp_path / "t7_results.csv"
    base = {column: float("nan") for column in RESULT_COLUMNS}
    values = [(1, 10.0 - 5e-6, 0.0), (2, 9.99, 5e-6), (3, 4.0, 0.5)]
    for episode, hashtag, retweet in values:
        for method in ("B1_goc", "B1_chuan"):
            for seed in (42, 43):
                append_result(path, {**base, "campaign": "c", "episode": episode, "q": 0.25, "tau": float(episode), "method": method, "seed": seed, "eligible": True, "beta_hashtag": hashtag, "beta_retweet": retweet})
    append_result(path, {**base, "campaign": "c", "episode": 1, "q": 0.25, "tau": 1.0, "method": "B0_K2", "seed": 42, "eligible": True})
    table = {(row["layer"], row["variant"]): row for row in run_t7.beta_bound_table(path, CFG)}
    assert set(table) == {(layer, variant) for layer in ("retweet", "hashtag") for variant in ("B1_goc", "B1_chuan")}
    for variant in ("B1_goc", "B1_chuan"):
        assert (table[("hashtag", variant)]["n_selected"], table[("hashtag", variant)]["n_at_upper"], table[("hashtag", variant)]["n_at_lower"]) == (3, 1, 0)
        assert (table[("retweet", variant)]["n_selected"], table[("retweet", variant)]["n_at_upper"], table[("retweet", variant)]["n_at_lower"]) == (3, 0, 2)


# --- Phần 9: DS2 và S3 ---
DS2 = CFG["ds2"]
S3 = CFG["s3"]
T0 = float(DS2["t0_minutes"])


@pytest.fixture(scope="module")
def ds2_data():
    return {seed: generate_data(seed) for seed in DS2["data_seeds"][:2]}


def ds2_jobs(ds2_data):
    return [ds2_job(datasets, truth, sims, f"DS2_{conf}_d{seed}", CFG, 5) for seed, (datasets, truth, _) in ds2_data.items() for conf, sims in DS2["configurations"].items()]


@pytest.mark.parametrize("data_seed", DS2["data_seeds"])
def test_ds2_t0_regression_khong_doi_bo_sinh(data_seed):
    """D2.5: t0 = 1920 lấy từ config; bộ sinh không đổi. Mọi lớp: hành động phối hợp đầu tiên >= t0, và có hành động phối hợp trong [t0, t0 + 15)."""
    datasets, truth, users = generate_data(data_seed)
    firsts = [events.loc[events.is_inauthentic, "timestamp"].min() for events in datasets.values()]
    assert min(firsts) >= T0 and min(firsts) < T0 + 15
    assert all(not events.loc[events.user_id.isin(truth) & events.timestamp.lt(T0)].shape[0] for events in datasets.values())
    assert len(truth) == 6 and len(users) == 46


def test_ds2_job_moc_nhan_cap_tai_khoan_va_lop(ds2_data):
    for name, stream, marks, events_fn, layers in ds2_jobs(ds2_data):
        seed = int(name.rsplit("_d", 1)[1])
        truth = ds2_data[seed][1]
        assert stream.t_min.is_monotonic_increasing and stream.episode.eq(1).all()
        assert stream.is_control.eq(~stream.accountid.isin(truth)).all()
        assert layers == (["sim1", "sim2", "sim3"] if "_L123_" in name else ["sim3"]) and set(stream.layer) == set(layers)
        assert marks.tau.is_monotonic_increasing and len(marks) == len(DS2["pre_t0_offsets_minutes"]) + len(CFG["q_values"])
        pre = marks.loc[marks.pre_t0]
        assert pre.q.isna().all() and sorted(pre.tau) == sorted(T0 - offset for offset in DS2["pre_t0_offsets_minutes"])
        assert pre.n_io.eq(0).all() and not pre.eligible.any()
        io_times = np.sort(stream.loc[stream.is_inauthentic, "t_min"].to_numpy())
        for mark in marks.loc[~marks.pre_t0].itertuples(index=False):
            assert mark.tau == io_times[int(np.ceil(round(mark.q * 100) * len(io_times) / 100)) - 1] and mark.tau >= T0
            seen = stream.loc[stream.t_min.le(mark.tau)]
            assert mark.n_io == seen.loc[~seen.is_control, "accountid"].nunique() and mark.eligible == (mark.n_io >= 5 and mark.n_ctrl >= 5)
        events = events_fn(stream)
        assert set(events.action_type) == set(layers) and events.content_id.str.contains(":").all()


def test_s3_khong_co_phat_hien_dung_truoc_t0(ds2_data):
    """S3 trên DS2 là test chống rò rỉ tương lai: trước t0 tài khoản phối hợp chưa có trong D(tau), nên không được có cụm phối hợp đúng."""
    for name, stream, marks, events_fn, layers in ds2_jobs(ds2_data):
        truth = ds2_data[int(name.rsplit("_d", 1)[1])][1]
        for mark in marks.loc[marks.pre_t0].itertuples(index=False):
            history, current = observe(stream, mark.tau, mark.episode)
            nodes, population = sorted(history.accountid.unique()), set(current.accountid)
            assert not truth & set(nodes)
            pairs = checkpoint_pairs(events_fn(history), layers, mark.tau)
            for keep_zero in (True, False):
                graphs, info = build_b1_graphs(pairs, nodes, keep_zero, mark.tau, BOUNDS, 42)
                for seed in (42, 46):
                    partition = leiden_multiplex(graphs, nodes, seed, [info[layer]["edges_author"] for layer in layers])
                    assert count_true_detections(partition, population, truth, S3["min_coordinated"], S3["precision_threshold"]) == 0
            for k in CFG["b0_clusters"]:
                assert count_true_detections(random_partition(population, k, 42), population, truth, S3["min_coordinated"], S3["precision_threshold"]) == 0


def test_s3_bat_duoc_ro_ri_tuong_lai(ds2_data):
    """Cố tình đưa toàn bộ luồng (gồm sau t0) vào mốc trước t0: U2 phải báo lỗi; bỏ qua U2 thì phép đo S3 thấy phát hiện đúng (chứng minh S3 đủ nhạy)."""
    found = []
    for name, stream, marks, events_fn, layers in ds2_jobs(ds2_data):
        truth = ds2_data[int(name.rsplit("_d", 1)[1])][1]
        tau = float(marks.loc[marks.pre_t0, "tau"].max())
        with pytest.raises(AssertionError, match="U2"):
            checkpoint_pairs(events_fn(stream), layers, tau)
        with pytest.raises(AssertionError, match="U2"):
            ds2_trivial_rows(stream, ~stream.groupby("accountid").is_control.first(), tau)
        leaked_tau = float(stream.t_min.max())
        nodes = sorted(stream.accountid.unique())
        graphs, info = build_b1_graphs(checkpoint_pairs(events_fn(stream), layers, leaked_tau), nodes, True, leaked_tau, BOUNDS, 42)
        partition = leiden_multiplex(graphs, nodes, 42, [info[layer]["edges_author"] for layer in layers])
        found.append(count_true_detections(partition, set(nodes), truth, S3["min_coordinated"], S3["precision_threshold"]))
    assert all(count >= 1 for count in found), found


def test_count_true_detections_nguong():
    truth = {"c1", "c2", "c3"}
    population = truth | {"a1", "a2", "a3"}
    assert count_true_detections([{"c1", "c2", "a1"}, {"c3", "a2", "a3"}], population, truth, 2, 0.5) == 1
    assert count_true_detections([{"c1", "a1", "a2"}, {"c2", "c3", "a3", "a2"}], population, truth, 2, 0.5) == 1
    assert count_true_detections([{"c1", "c2", "a1", "a2", "a3"}], population, truth, 2, 0.5) == 0
    assert count_true_detections([{"c1", "c2", "out"}], population, truth, 2, 0.5) == 1


def test_ds2_trivial_zero_delta_hang_so_va_iter_campaign_rows(ds2_data):
    name, stream, marks, events_fn, layers = ds2_jobs(ds2_data)[1]
    mark = marks.loc[marks.eligible].iloc[-1]
    _, current = observe(stream, mark.tau, mark.episode)
    rows = {row["method"]: row for row in ds2_trivial_rows(current, ~current.groupby("accountid").is_control.first(), mark.tau)}
    n_io, n = mark.n_io, mark.n_io + mark.n_ctrl
    assert rows["TRIV_zero_delta_repost_high"]["f1_star"] == rows["TRIV_zero_delta_repost_low"]["f1_star"] >= 2 * n_io / (n + n_io) - 1e-12
    result = pd.DataFrame(iter_campaign_rows(stream, marks, events_fn, name, CFG, layers, (42,), trivial_fn=ds2_trivial_rows, progress=lambda _: None))
    pre_rows = result.loc[result.q.isna()]
    assert pre_rows.loc[pre_rows.method.str.startswith(("B0_", "B1_")), "f1_star"].eq(0).all() and pre_rows.loc[pre_rows.method.str.startswith("TRIV_"), "f1_star"].isna().all()
    b1 = result.loc[result.method.str.startswith("B1_") & result.eligible]
    assert len(b1) and b1.beta_sim3.notna().all() and b1[["beta_retweet", "beta_sim1"]].isna().all().all()


def test_load_ds2_cua_script():
    jobs = run_t7.load_ds2(CFG)
    assert [job[0] for job in jobs] == [f"DS2_{conf}_d{seed}" for seed in DS2["data_seeds"] for conf in DS2["configurations"]]
    assert all(len(job) == 6 and job[5] is ds2_trivial_rows for job in jobs)


# --- Sửa trước Bước 3: phá hòa TRIV, trùng sự kiện DS2, --output-dir, bảng mốc DS2 ---
def constant_scores(io_ids, ctrl_ids):
    ids = [*io_ids, *ctrl_ids]
    scores = pd.DataFrame({"first_seen": 1.0, "actions_per_active_day": 2.0, "zero_delta_repost": 0.0}, index=pd.Index(ids, name="accountid"))
    return scores, pd.Series([True] * len(io_ids) + [False] * len(ctrl_ids), index=scores.index)


def test_triv_pha_hoa_khong_phu_thuoc_nhan_nao_co_id_nho_hon():
    """Điểm bằng nhau hoàn toàn: IO có ID nhỏ hơn (a*) hay lớn hơn (z*) control (m*) thì phân phối F1* tầm thường phải như nhau."""
    ctrl = [f"m{i}" for i in range(6)]
    small, large = constant_scores([f"a{i}" for i in range(4)], ctrl), constant_scores([f"z{i}" for i in range(4)], ctrl)
    raw = [max(row["F1_star"] for row in oracle_top_k(scores["first_seen"], labels)) for scores, labels in (small, large)]
    assert raw[0] - raw[1] > 0.3
    means = [np.mean([t7_checkpoints._rows_from_scores(scores, labels, seed)[0]["f1_star"] for seed in range(400)]) for scores, labels in (small, large)]
    assert abs(means[0] - means[1]) < 0.03, means
    assert t7_checkpoints._rows_from_scores(*small, 7) == t7_checkpoints._rows_from_scores(small[0].iloc[::-1], small[1].iloc[::-1], 7)


def test_alias_ids_la_hoan_vi_co_seed():
    ids = [f"u{i}" for i in range(12)]
    alias = t7_checkpoints._alias_ids(reversed(ids), 3)
    assert alias == t7_checkpoints._alias_ids(ids, 3) and alias != t7_checkpoints._alias_ids(ids, 4)
    assert sorted(alias.values()) == [f"{i:02d}" for i in range(12)]


def test_ds2_diem_i_ii_bo_su_kien_trung_giua_cac_lop(ds2_data):
    jobs = {name.split("_")[1]: (stream, marks) for name, stream, marks, _, _ in ds2_jobs(ds2_data)[:2]}
    full, single = (t7_checkpoints.ds2_scores(jobs[conf][0]) for conf in ("L123", "L3"))
    truth = ds2_data[DS2["data_seeds"][0]][1]
    authentic = full.index.difference(sorted(truth))
    assert np.allclose(full.loc[authentic, "actions_per_active_day"], single.loc[authentic, "actions_per_active_day"])
    stream = jobs["L123"][0]
    coordinated = stream.loc[stream.accountid.isin(truth)].groupby("accountid").size() / stream.loc[stream.accountid.isin(truth)].groupby("accountid").day.nunique()
    assert np.allclose(full.loc[coordinated.index, "actions_per_active_day"], coordinated)
    assert stream.duplicated(["accountid", "t_min", "action_type", "content_id"]).sum() > 0


def test_output_dir_va_bang_moc_ds2(ds2_data, tmp_path):
    paths = __import__("json").loads((Path(__file__).resolve().parents[1] / "configs" / "paths.json").read_text(encoding="utf-8"))
    assert run_t7.resolve_output_dir(run_t7.parse_args(["--output-dir", str(tmp_path)], CFG), paths) == tmp_path
    assert run_t7.resolve_output_dir(run_t7.parse_args([], CFG), paths) == run_t7.ROOT / paths["t7_output_dir"]
    name, stream, marks, events_fn, layers = ds2_jobs(ds2_data)[1]
    path = tmp_path / "t7_results.csv"
    write_csv(path, iter_campaign_rows(stream, marks.loc[~marks.pre_t0], events_fn, name, CFG, layers, (42,), trivial_fn=ds2_trivial_rows, progress=lambda _: None))
    table = {row["q"]: row for row in run_t7.ds2_ineligible_table(path)}
    assert set(table) == set(CFG["q_values"]) and all(row["configuration"] == "L3" and row["n_marks"] == 1 for row in table.values())
    assert [table[q]["n_ineligible"] for q in sorted(table)] == [int(not mark.eligible) for mark in marks.loc[~marks.pre_t0].sort_values("q").itertuples()]
