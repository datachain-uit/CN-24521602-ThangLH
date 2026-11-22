"""P6 + VC (Review-2026-11-21 §6.3–6.5): biến thể, U2, VC giữ đợt, vòng lặp mới trùng infer_partition theo seed, shard ghi liên tục."""
import json
import types

import numpy as np
import pandas as pd
import pytest

from scripts import run_p6_vc
from src.eval import p6_variants
from src.eval.metrics import partition_metrics
from src.eval.p6_variants import COLUMNS, KEY_COLUMNS, P5_METHODS, append_rows, b1_partitions, cap_volume, io_marks, make_row, method_rows, method_units, p6_rows, p6_units, read_shard, unit_key, vc_marks, vc_stream
from src.eval.seckin_protocol import EpisodeRules, build_events, daily_coverage, evaluate_baselines, harmonize_time, make_checkpoints, make_episodes, prepare_variant
from src.eval.t7_checkpoints import TRIV_METHODS, infer_partition, load_config, observe, trivial_rows

# Fixture chép từ tests/test_seckin_protocol.py (T9), như tests/test_t7.py; không import chéo vì package "tests" trùng tên với mã tác giả.
START = pd.Timestamp("2020-01-01", tz="UTC")
RULES = EpisodeRules()
VARIANTS = ["main", "without_hashtag", "shuffle_account_day", "cap_account_day", "reset_state"]
SEEDS = [42, 43, 44]


def posts(rows):
    return harmonize_time(pd.DataFrame([dict(post_time=START + pd.Timedelta(days=day, minutes=minute, seconds=31), accountid=user, is_control=control, is_repost=True, reposted_postid=content, hashtags=["h"], urls=[], account_mentions=[]) for day, minute, user, control, content in rows]))


def campaign_rows():
    rows = [(0, 0, "old_a", False, "p"), (0, 1, "old_b", False, "q"), (0, 0, "old_c", True, "p"), (7, 0, "excluded", False, "skip"), (7, 1, "excluded2", False, "skip"), (6, 0, "midnight", True, "next_day")]
    rows += [(5, minute, f"i{i}", False, "p") for i, minute in enumerate([0, 1, 1, 4, 5])]
    rows += [(5, i + 2, f"c{i}", True, "p") for i in range(5)]
    rows += [(5, 5, "i0", False, "other"), (5, 1439, "c5", True, "day_end")]
    rows += [(9, 0, "i0", False, "p"), (9, 1, "i1", False, "r"), (9, 1, "c0", True, "p")]
    rows += [(12, 0, "i0", False, "p"), (12, 1, "i4", False, "r"), (12, 2, "i0", False, "r")]
    rows += [(12, 2, "c0", True, "r"), (12, 7, "after_io", True, "r")]
    return rows


def build_campaign():
    frame = posts(campaign_rows())
    daily = daily_coverage(frame, RULES)
    stream, episodes = make_episodes(frame, daily, RULES, 1)
    return stream, daily, episodes, make_checkpoints(stream, episodes, RULES), RULES


@pytest.fixture
def campaign():
    return build_campaign()


@pytest.fixture
def cfg():
    return load_config()


def eligible_mark(checkpoints):
    marks = io_marks(checkpoints, [0.25, 1.0])
    return marks.loc[marks["eligible"]].iloc[0].to_dict()


# --- cap_account_day / cap_volume ---
def tie_stream():
    """a có 4 hành động cùng ngày, 2 cái đầu trùng phút (thứ tự gốc p1 trước p2); b một hành động; ngày sau a một hành động."""
    rows = [(0, 5, "a", False, "p1"), (0, 5, "a", False, "p2"), (0, 7, "a", False, "p3"), (0, 9, "a", False, "p4"), (0, 6, "b", True, "q1"), (1, 0, "a", False, "p5")]
    return posts(rows).sort_values("t_min", kind="stable").reset_index(drop=True)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_cap_giu_dung_k_hanh_dong_dau_on_dinh_khi_hoa_khong_mat_tai_khoan_ngay(k):
    stream = tie_stream()
    for capped in (cap_volume(stream, k), prepare_variant(stream, stream, "cap_account_day", k, seed=0)[0]):
        assert capped.groupby(["accountid", "day"]).size().le(k).all()
        assert set(map(tuple, capped[["accountid", "day"]].drop_duplicates().to_numpy())) == set(map(tuple, stream[["accountid", "day"]].drop_duplicates().to_numpy()))
        day0 = capped.loc[capped["accountid"].eq("a") & capped["day"].eq(stream["day"].min()), "reposted_postid"].tolist()
        assert day0 == ["p1", "p2", "p3", "p4"][:k]
        assert capped["t_min"].is_monotonic_increasing
    with pytest.raises(ValueError):
        cap_volume(stream, 0)


# --- shuffle_account_day ---
@pytest.mark.parametrize("seed", SEEDS)
def test_shuffle_giu_tai_khoan_ngay_so_hanh_dong_va_nam_trong_ngay(campaign, seed):
    stream, _, _, checkpoints, _ = campaign
    mark = eligible_mark(checkpoints)
    history, current = observe(stream, mark["tau"], mark["episode"])
    shuffled, _ = prepare_variant(history, current, "shuffle_account_day", 1, seed)
    assert history.groupby(["accountid", "day"]).size().equals(shuffled.groupby(["accountid", "day"]).size())
    assert shuffled["t_min"].dt.floor("D").eq(shuffled["day"]).all()
    assert shuffled["t_min"].le(mark["tau"]).all()
    key = ["accountid", "day", "t_min"]
    assert history[key].sort_values(key, ignore_index=True).equals(shuffled[key].sort_values(key, ignore_index=True))
    again, _ = prepare_variant(history, current, "shuffle_account_day", 1, seed)
    assert again["t_min"].equals(shuffled["t_min"])


def test_shuffle_doi_thoi_diem_voi_seed_khac_nhau(campaign):
    stream, _, _, checkpoints, _ = campaign
    mark = eligible_mark(checkpoints)
    history, current = observe(stream, mark["tau"], mark["episode"])
    results = {tuple(prepare_variant(history, current, "shuffle_account_day", 1, seed)[0]["t_min"]) for seed in range(20)}
    assert len(results) > 1


# --- reset_state ---
def test_reset_state_chi_thay_du_lieu_dot_hien_tai(campaign, cfg, monkeypatch):
    stream, _, _, checkpoints, _ = campaign
    mark = eligible_mark(checkpoints)
    history, current = observe(stream, mark["tau"], mark["episode"])
    assert history["episode"].nunique() > 1
    posts_reset, _ = prepare_variant(history, current, "reset_state", 1, 42)
    assert set(posts_reset["accountid"]) == set(current["accountid"]) and len(posts_reset) == len(current)
    seen = []
    real = p6_variants.b1_partitions

    def spy(events, nodes, *args, **kwargs):
        seen.append((events["timestamp"].min(), tuple(nodes)))
        return real(events, nodes, *args, **kwargs)
    monkeypatch.setattr(p6_variants, "b1_partitions", spy)
    list(p6_rows(stream, mark, "test", ["reset_state"], [42], 1, cfg, "h", set()))
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    assert seen[0][0] >= (current["t_min"].min() - epoch) // pd.Timedelta(minutes=1)
    assert seen[0][1] == tuple(sorted(current["accountid"].unique()))


# --- U2 cho mọi biến thể ---
def test_u2_moi_bien_the_khong_doc_sau_tau(campaign, cfg, monkeypatch):
    stream, _, _, checkpoints, _ = campaign
    mark = eligible_mark(checkpoints)
    stream = pd.concat([stream, stream.iloc[[-1]].assign(t_min=stream["t_min"].max() + pd.Timedelta(days=30))], ignore_index=True)
    seen = []
    real = p6_variants.build_events

    def spy(frame, layers):
        seen.append(frame["t_min"].max())
        return real(frame, layers)
    monkeypatch.setattr(p6_variants, "build_events", spy)
    rows = list(p6_rows(stream, mark, "test", VARIANTS, SEEDS, 1, cfg, "h", set()))
    assert len(rows) == len(VARIANTS) * len(SEEDS)
    assert seen and max(seen) <= mark["tau"]


def test_u2_vi_pham_bi_bat(campaign, cfg, monkeypatch):
    stream, _, _, checkpoints, _ = campaign
    mark = eligible_mark(checkpoints)

    def leak(history, current, variant, cap_k, seed):
        posts_, layers = prepare_variant(history, current, variant, cap_k, seed)
        return posts_.assign(t_min=posts_["t_min"] + pd.Timedelta(days=365)), layers
    monkeypatch.setattr(p6_variants, "prepare_variant", leak)
    with pytest.raises(AssertionError, match="U2"):
        list(p6_rows(stream, mark, "test", ["shuffle_account_day"], [42], 1, cfg, "h", set()))


# --- Vòng lặp mới trùng infer_partition gọi theo từng seed ---
@pytest.mark.parametrize("variant", VARIANTS)
def test_b1_partitions_trung_infer_partition_theo_seed(campaign, cfg, variant):
    stream, _, _, checkpoints, _ = campaign
    mark = eligible_mark(checkpoints)
    history, current = observe(stream, mark["tau"], mark["episode"])
    population, truth = set(current["accountid"]), set(current.loc[~current["is_control"], "accountid"])
    rows = {row["seed"]: row for row in p6_rows(stream, mark, "test", [variant], SEEDS, 2, cfg, "h", set())}
    for seed in SEEDS:
        posts_, layers = prepare_variant(history, current, variant, 2, seed)
        nodes = sorted((current if variant == "reset_state" else history)["accountid"].unique())
        events = build_events(posts_, layers)
        reference = infer_partition(events, nodes, layers, seed, cfg=cfg)
        ours = [partition for _, s, partition, _, _ in b1_partitions(events, nodes, layers, mark["tau"], {"B1_chuan": SEEDS}, cfg) if s == seed]
        assert sorted(map(sorted, ours[0])) == sorted(map(sorted, reference))
        expected = partition_metrics(reference, population, truth)
        for metric in ("f1_star", "nmi", "homogeneity", "wp", "n_clusters_ge2"):
            assert rows[seed][metric] == pytest.approx(expected[metric], abs=1e-12)


def test_b1_goc_va_chuan_chung_ghep_cap_trung_tinh_rieng(campaign, cfg):
    stream, _, _, checkpoints, _ = campaign
    mark = eligible_mark(checkpoints)
    history, _ = observe(stream, mark["tau"], mark["episode"])
    nodes = sorted(history["accountid"].unique())
    events = build_events(history[p6_variants.MODEL_COLUMNS], tuple(cfg["ds1_layers"]))
    both = {(name, seed): partition for name, seed, partition, _, _ in b1_partitions(events, nodes, cfg["ds1_layers"], mark["tau"], {"B1_chuan": SEEDS, "B1_goc": SEEDS}, cfg)}
    for name in ("B1_chuan", "B1_goc"):
        for seed in SEEDS:
            alone = next(partition for _, s, partition, _, _ in b1_partitions(events, nodes, cfg["ds1_layers"], mark["tau"], {name: [seed]}, cfg))
            assert both[(name, seed)] == alone


# --- VC ---
@pytest.mark.parametrize("k", [1, 2, 5])
def test_vc_giu_ngay_phu_va_dot_ghi_tau_goc_tau_vc(campaign, k):
    stream, daily, episodes, checkpoints, rules = campaign
    capped = vc_stream(stream, daily, episodes, k, rules, 1)
    goc = io_marks(checkpoints, [0.10, 0.25, 0.50, 1.0])
    marks = vc_marks(capped, episodes, rules, [0.10, 0.25, 0.50, 1.0], goc)
    assert len(marks) == len(goc) and marks[["episode", "q"]].equals(goc[["episode", "q"]])
    assert marks["tau_goc"].equals(goc["tau"]) and marks["eligible_goc"].equals(goc["eligible"])
    assert marks["tau"].equals(marks["tau_vc"]) and marks["eligible"].equals(marks["eligible_vc"])
    assert capped.groupby(["accountid", "day"]).size().le(k).all()


def test_moc_theo_ngay_vc_giu_tau_va_eligible_khoa_khong_trung(campaign, cfg):
    stream, daily, episodes, checkpoints, rules = campaign
    goc = io_marks(checkpoints, [0.10, 0.25, 0.50, 1.0], [1, 3, 7])
    days = goc.loc[goc["checkpoint_type"].eq("day")]
    assert len(days) == len(checkpoints.loc[checkpoints["checkpoint_type"].eq("day")]) and days["q"].isna().all()
    capped = vc_stream(stream, daily, episodes, 1, rules, 1)
    marks = vc_marks(capped, episodes, rules, [0.10, 0.25, 0.50, 1.0], goc, [1, 3, 7])
    vc_days = marks.loc[marks["checkpoint_type"].eq("day")]
    assert vc_days["tau_vc"].equals(vc_days["tau_goc"]) and vc_days["eligible_vc"].equals(vc_days["eligible_goc"])
    keys = [unit_key(p6_variants.mark_key("GOC", "c", "main", method, mark, seed, "h")) for mark in marks.to_dict("records") for method, seed in method_units(mark, [] if mark["checkpoint_type"] == "day" else ["B1_chuan"], SEEDS, True, True)]
    assert len(keys) == len(set(keys))
    mark = vc_days.loc[vc_days["eligible"]].iloc[0].to_dict()
    rows = list(method_rows("VC", "vc", capped, mark, "test", [], SEEDS, True, True, cfg, "h", set()))
    assert sorted(row["method"] for row in rows) == sorted([*TRIV_METHODS, *P5_METHODS])


def test_vc_bao_loi_khi_doi_dot_hoac_ngay_phu(campaign):
    stream, daily, episodes, _, rules = campaign
    broken = daily.copy()
    broken.loc[broken.index[0], "covered_c1"] = not broken["covered_c1"].iloc[0]
    with pytest.raises(AssertionError):
        vc_stream(stream, broken, episodes, 1, rules, 1)
    moved = episodes.assign(qualified=~episodes["qualified"])
    with pytest.raises(AssertionError):
        vc_stream(stream, daily, moved, 1, rules, 1)


def test_method_rows_trivial_va_p5_trung_ham_t7_t9(campaign, cfg):
    stream, _, _, checkpoints, _ = campaign
    mark = eligible_mark(checkpoints)
    rows = list(method_rows("GOC", "main", stream, mark, "test", ["B1_chuan", "B1_goc"], [42], True, True, cfg, "h", set()))
    assert sorted(row["method"] for row in rows) == sorted(["B1_chuan", "B1_goc", *TRIV_METHODS, *P5_METHODS])
    _, current = observe(stream, mark["tau"], mark["episode"])
    triv = {row["method"]: row for row in trivial_rows(current, ~current.groupby("accountid")["is_control"].first(), mark["tau"])}
    p5 = evaluate_baselines(stream, pd.DataFrame([dict(episode=mark["episode"], tau=mark["tau"], eligible=True)]))
    for row in rows:
        if row["method"] in triv:
            assert row["f1_star"] == triv[row["method"]]["f1_star"] and row["oracle_k"] == triv[row["method"]]["oracle_k"]
        if row["method"] in P5_METHODS:
            ref = p5.loc[(("P5_" + p5["method"] + "_" + p5["direction"]) == row["method"])].iloc[0]
            assert row["auc"] == ref["auc"] and row["f1_star"] == ref["F1_star"]


def test_moc_khong_dat_ghi_dong_rong_va_dem_don_vi(campaign, cfg):
    stream, _, _, checkpoints, _ = campaign
    mark = io_marks(checkpoints, [0.25]).iloc[0].to_dict()
    assert not mark["eligible"]
    rows = list(p6_rows(stream, mark, "test", VARIANTS, SEEDS, 1, cfg, "h", set()))
    assert len(rows) == len(VARIANTS) == len(p6_units(mark, VARIANTS, SEEDS)) and all(np.isnan(row["f1_star"]) and np.isnan(row["seed"]) for row in rows)
    assert len(method_units(mark, ["B1_chuan", "B1_goc"], SEEDS, True, True)) == 2 + len(TRIV_METHODS) + len(P5_METHODS)


# --- Shard ---
def sample_row(seed, f1=0.5):
    return make_row("P6", "c", "main", "B1_chuan", dict(episode=1, checkpoint_type="io_pct", checkpoint_value=25, q=0.25, tau=START, tau_goc=START, tau_vc=pd.NaT, eligible=True, eligible_goc=True, eligible_vc=np.nan, n_io=5, n_ctrl=5, io_actions_episode=10), seed, "h", dict(f1_star=f1))


def test_shard_ghi_doc_khoa_va_cat_dong_cuoi_dang_do(tmp_path):
    path = tmp_path / "results_c.csv"
    for seed in SEEDS:
        append_rows(path, [sample_row(seed)])
    keys = read_shard(path)
    assert keys == {unit_key(sample_row(seed)) for seed in SEEDS}
    complete = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(complete.splitlines(keepends=True)[-1][:25])
    warnings = []
    assert read_shard(path, warn=warnings.append, repair=False) == keys and path.read_bytes() != complete
    assert read_shard(path, warn=warnings.append) == keys and path.read_bytes() == complete and len(warnings) == 2
    append_rows(path, [sample_row(45)])
    assert len(read_shard(path)) == 4


def test_shard_dong_hong_o_giua_hoac_khoa_trung_bao_loi(tmp_path):
    path = tmp_path / "results_c.csv"
    append_rows(path, [sample_row(42), sample_row(42, f1=0.7)])
    with pytest.raises(ValueError, match="trùng"):
        read_shard(path)
    path.unlink()
    append_rows(path, [sample_row(42), sample_row(43)])
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + b"x,y\r\n" + lines[2])
    with pytest.raises(ValueError, match="hỏng"):
        read_shard(path)
    with pytest.raises(ValueError, match="schema"):
        append_rows(path, [{**sample_row(44), "extra": 1}])


def test_cot_va_khoa(campaign):
    assert set(KEY_COLUMNS) <= set(COLUMNS) and len(COLUMNS) == len(set(COLUMNS))
    assert unit_key(sample_row(42.0)) == unit_key({**sample_row(42), "q": "0.25"})


# --- Config ---
def test_k_trong_config_khop_t9_va_lech_thi_dung():
    paths, t7, p6, vc = run_p6_vc.load_all()
    p2 = run_p6_vc.ROOT / paths["t9_output_dir"] / run_p6_vc.P2_FILE
    run_p6_vc.check_k(p6, vc, p2, t7["control_min"])
    assert p6["cap_k_p6"] == {"Catalonia": 2, "Iran_5": 1, "Thailand": 3}
    assert vc["k_c"] == {"Catalonia": 5, "Iran_5": 3, "Thailand": 14, "Ghana_Nigeria": 16, "Bangladesh": 19, "Spain": 62, "Armenia": 64}
    with pytest.raises(ValueError, match="DỪNG"):
        run_p6_vc.check_k({**p6, "cap_k_p6": {**p6["cap_k_p6"], "Iran_5": 2}}, vc, p2, t7["control_min"])
    with pytest.raises(ValueError, match="DỪNG"):
        run_p6_vc.check_k(p6, {**vc, "k_c": {**vc["k_c"], "Spain": 61}}, p2, t7["control_min"])


def test_config_hash_doi_khi_config_doi():
    paths, t7, p6, vc = run_p6_vc.load_all()
    p2 = run_p6_vc.ROOT / paths["t9_output_dir"] / run_p6_vc.P2_FILE
    base = run_p6_vc.config_hash(t7, p6, vc, p2)
    assert base == run_p6_vc.config_hash(json.loads(json.dumps(t7)), p6, vc, p2)
    assert base != run_p6_vc.config_hash(t7, {**p6, "seeds": [42]}, vc, p2)


def test_guard_tu_choi_head_config_dirty(tmp_path):
    state = dict(commit="abc", tags=[], code_dirty=False)
    args = types.SimpleNamespace(allow_dirty=False)
    run = tmp_path / "run.json"
    run_p6_vc.guard(run, state, "h", args)
    run.write_text(json.dumps(dict(commit="abc", config_hash="h", code_dirty=False)), encoding="utf-8")
    run_p6_vc.guard(run, state, "h", args)
    for bad_state, digest in ((dict(state, commit="xyz"), "h"), (state, "other"), (dict(state, code_dirty=True), "h")):
        with pytest.raises(SystemExit, match="TỪ CHỐI"):
            run_p6_vc.guard(run, bad_state, digest, args)
    run.write_text(json.dumps(dict(commit="abc", config_hash="h", code_dirty=True)), encoding="utf-8")
    with pytest.raises(SystemExit, match="TỪ CHỐI"):
        run_p6_vc.guard(run, state, "h", args)
