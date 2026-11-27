"""Test T13 (DS2-v2): tất định, v1 không đổi, ngụy trang, mô phỏng 3, bất biến chế độ seckin, nâng hành động, mốc q (S1), U2 với thời gian giây (S2), S3 nhanh."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from scripts import run_t13
from scripts.run_t13 import START, T13_COLUMNS, campaign_name, events_p1, events_seconds, load_config, make_marks, prepare_stream, read_done_keys, run_mode, seconds_stream, triv_rows_with_auc, world_jobs, scores_seconds
from src.eval.seckin_protocol import EpisodeRules, make_checkpoints
from src.eval.t7_checkpoints import checkpoint_pairs, observe
from src.simulation.synthetic import SimulationConfig, generate_authentic_events, generate_simulation1_dataset, generate_simulation1_events, generate_simulation2_dataset, generate_simulation3_dataset
from src.simulation.synthetic_v2 import DAY_S, SIM3_GROUPS, SIM3_WINDOW_SIZES, assign_simulation3_users_v2, coordinated_events, make_modes, make_world, replace_background, table2_datasets, to_posts, world_background

CFG = load_config()
SEEDS = CFG["data_seeds"][:4]
V1_HASHES = {1: "c359a15e79e77f6a7804cbad00093e248521efdc30f05f4ea8bbc0a07a16fde0", 2: "100283366654ae05d133eb50f2e34ea6647aae733981f7b2e20b81881cba0fcc", 3: "14415852fe1df37462ca682d21a17d7d8024eb625adc0745796cf9059fd004bf"}
SMALL = dict(CFG["world"], n_normal=80, mean_actions_per_day=3.0)


def coord_times(world):
    return START + pd.to_timedelta(world.loc[world["is_coord_action"], "t"].to_numpy(np.int64), unit="s")


def t0_of(meta):
    return START + pd.Timedelta(seconds=int(meta["t0"]))


@pytest.fixture(scope="module")
def jobs():
    """Thế giới chính (shape 2, mean 8) cho 4 seed × 3 kịch bản."""
    return {(seed, sim): world_jobs(seed, sim, CFG["world"]["gamma_shape"], CFG["world"], CFG) for seed in SEEDS for sim in CFG["scenarios"]}


# --- Tất định và v1 ---
def test_v1_khong_doi_so_hash_mau_co_dinh():
    config = SimulationConfig(seed=20261108)
    for sim, generator in {1: generate_simulation1_dataset, 2: generate_simulation2_dataset, 3: generate_simulation3_dataset}.items():
        assert hashlib.sha256(generator(config).to_csv(index=False).encode()).hexdigest() == V1_HASHES[sim]


def test_cung_seed_cung_du_lieu():
    a, meta_a = make_world(SEEDS[0], 2, 2.0, CFG["world"])
    b, meta_b = make_world(SEEDS[0], 2, 2.0, CFG["world"])
    assert_frame_equal(a, b)
    assert meta_a == meta_b
    modes_a, modes_b = make_modes(a, meta_a, CFG["modes"], CFG["ctrl_cap_per_day"], CFG["clean_sample_from"]), make_modes(b, meta_b, CFG["modes"], CFG["ctrl_cap_per_day"], CFG["clean_sample_from"])
    for mode in CFG["modes"]:
        assert_frame_equal(to_posts(modes_a[mode][0]), to_posts(modes_b[mode][0]))
    for variant in CFG["table2"]["variants"]:
        first, second = table2_datasets(SEEDS[0], variant, CFG["table2"]), table2_datasets(SEEDS[0], variant, CFG["table2"])
        for sim in (1, 2, 3):
            assert_frame_equal(first[0][sim], second[0][sim])
    c, _ = make_world(SEEDS[1], 2, 2.0, CFG["world"])
    assert not a[["accountid", "t"]].equals(c[["accountid", "t"]])


def test_bang2_khong_nguy_trang_giu_nen_va_mo_phong_1_2_cua_v1():
    datasets, _ = table2_datasets(SEEDS[0], "no_camouflage", CFG["table2"])
    config = SimulationConfig(seed=SEEDS[0])
    assert_frame_equal(datasets[1], generate_simulation1_dataset(config))
    assert_frame_equal(datasets[2], generate_simulation2_dataset(config))
    authentic = generate_authentic_events(config)
    assert_frame_equal(datasets[3].loc[~datasets[3]["is_inauthentic"]].sort_values(["timestamp", "user_id"]).reset_index(drop=True), authentic.sort_values(["timestamp", "user_id"]).reset_index(drop=True))
    assert_frame_equal(coordinated_events(1, config), generate_simulation1_events(config))


# --- Ngụy trang ---
def test_nguy_trang_moi_tai_khoan_co_nen_ca_7_ngay_va_cung_phan_phoi(jobs):
    for (seed, sim), (world, meta, _) in jobs.items():
        io = world.loc[world["label"].eq(1)]
        assert io["day"].nunique() == CFG["world"]["days"]
        assert io.loc[~io["is_coord_action"], "day"].nunique() >= CFG["world"]["days"] - 1
        assert world["day"].between(0, CFG["world"]["days"] - 1).all()
        assert world["t"].dtype == np.int64


def test_clean_auc_khoi_luong_va_xuat_hien_dau_tien_xap_xi_0_5(jobs):
    values = {"actions_per_active_day": [], "first_seen": []}
    for (seed, sim), (world, meta, modes) in jobs.items():
        stream = seconds_stream(modes["clean"][2]["stream"])
        marks = make_marks(stream, coord_times(world), t0_of(meta), CFG["marks"], False)
        mark = marks.loc[marks["q"].eq(CFG["marks"]["q_main"])].iloc[0]
        _, current = observe(stream, mark["tau"], mark["episode"])
        rows = {row["method"]: row for row in triv_rows_with_auc(current, scores_seconds, CFG["triv_tiebreak_seed"])}
        for score in values:
            values[score].append(rows[f"TRIV_{score}_low"]["auc"])
    for score, aucs in values.items():
        assert 0.35 <= np.mean(aucs) <= 0.65, (score, aucs)


# --- Mô phỏng 3 ---
def test_mo_phong_3_dung_hai_nhom_luan_phien():
    assert set(SIM3_GROUPS[0]).isdisjoint(SIM3_GROUPS[1]) and all(len(group) == 3 for group in SIM3_GROUPS)
    for seed in SEEDS:
        events = assign_simulation3_users_v2(SimulationConfig(seed=seed))
        assert events["timestamp"].is_monotonic_increasing
        active = [set(part["user_id"]) for _, part in events.groupby("window", sort=True)]
        assert len(active) == len(SIM3_WINDOW_SIZES) == 4
        for window, users in enumerate(active):
            assert users == set(SIM3_GROUPS[window % 2])
        assert events.groupby("window").size().tolist() == list(SIM3_WINDOW_SIZES)
        bounds = events.groupby("window")["timestamp"].agg(["min", "max"])
        assert (bounds["min"].iloc[1:].to_numpy() > bounds["max"].iloc[:-1].to_numpy()).all()


# --- Nâng hành động khi thiếu nền ---
@pytest.mark.parametrize("shape", [0.5, 2.0])
def test_nang_chi_o_tai_khoan_ngay_co_hanh_dong_phoi_hop_va_tong_bang_max(shape):
    for seed in SEEDS:
        background = world_background(seed, shape, CFG["world"])
        before = background.groupby(["account", "day"]).size()
        for sim in CFG["scenarios"]:
            events = coordinated_events(sim, SimulationConfig(seed=seed))
            t = np.rint(events["timestamp"].to_numpy(dtype=float) * 60).astype(np.int64)
            coord = pd.DataFrame({"account": events["user_id"].str[-2:].astype(int).to_numpy(), "day": t // DAY_S, "t": t, "action": "hashtag", "content": "x", "is_coord_action": True})
            merged, stats = replace_background(background, coord, np.random.default_rng(seed))
            need = coord.groupby(["account", "day"]).size()
            after = merged.groupby(["account", "day"]).size()
            cells = before.index.union(after.index)
            before_all, after_all = before.reindex(cells, fill_value=0), after.reindex(cells, fill_value=0)
            changed = cells[after_all.to_numpy() != before_all.to_numpy()]
            # (a) chỉ tài khoản–ngày có hành động phối hợp mới có thể bị nâng; ngày khác giữ nguyên tổng.
            assert set(changed) <= set(need.index)
            assert (after_all.drop(need.index, errors="ignore") == before_all.drop(need.index, errors="ignore")).all()
            # (b) tổng mỗi tài khoản–ngày có phối hợp = max(tổng rút trước, số hành động phối hợp).
            expected = np.maximum(before_all.reindex(need.index).to_numpy(), need.to_numpy())
            assert (after_all.reindex(need.index).to_numpy() == expected).all()
            assert stats["raised_account_days"] == int((need.to_numpy() > before_all.reindex(need.index).to_numpy()).sum())
            assert stats["added_actions"] == int(np.maximum(need.to_numpy() - before_all.reindex(need.index).to_numpy(), 0).sum())


def test_metadata_ghi_ba_ty_le_nang(jobs):
    for (_, _), (_, meta, _) in jobs.items():
        for key in ("raised_share_coord_days", "raised_share_all_coord_days", "added_action_share"):
            assert 0.0 <= meta[key] <= 1.0
    _, metas = table2_datasets(SEEDS[0], "camouflage", CFG["table2"])
    assert {"raised_share_coord_days", "raised_share_all_coord_days", "added_action_share"} <= set(metas.columns)


# --- Bất biến của chế độ seckin ---
@pytest.mark.parametrize("mode", ["seckin_h0", "seckin_h10", "seckin_u0", "sampling_only"])
def test_bat_bien_che_do_seckin(jobs, mode):
    offset, cap = CFG["modes"][mode]["ctrl_offset"], CFG["ctrl_cap_per_day"]
    for (seed, sim), (world, meta, modes) in jobs.items():
        posts = modes[mode][0]
        io, ctrl = posts.loc[~posts["is_control"]], posts.loc[posts["is_control"]]
        io_days = (io["t"] // DAY_S).value_counts().index
        per_day = pd.DataFrame({"io": io.groupby(io["t"] // DAY_S)["accountid"].nunique(), "ctrl": ctrl.groupby(ctrl["t"] // DAY_S)["accountid"].nunique()}).fillna(0)
        # control − IO = ctrl_offset ở 100% số ngày có IO; không có control ở ngày không có IO.
        assert (per_day.loc[io_days, "ctrl"] - per_day.loc[io_days, "io"]).eq(offset).all()
        assert set(ctrl["t"] // DAY_S) <= set(io_days)
        assert ctrl.groupby(["accountid", ctrl["t"] // DAY_S]).size().max() <= cap
        if CFG["modes"][mode]["round_io"]:
            assert io["t"].mod(60).eq(0).all() and io["post_time"].dt.second.eq(0).all()
        else:
            assert_frame_equal(io[["accountid", "t"]].reset_index(drop=True), world.loc[world["label"].eq(1), ["accountid", "t"]].reset_index(drop=True))
        assert ctrl["t"].mod(60).ne(0).any()
        source = world.set_index(["accountid", "t", "content"]).index
        assert ctrl.set_index(["accountid", "t", "content"]).index.isin(source).all()


def test_control_giu_100_hanh_dong_dau_cua_ngay():
    world, meta = make_world(SEEDS[0], 1, 2.0, dict(CFG["world"], mean_actions_per_day=150.0, n_normal=60))
    posts = make_modes(world, meta, CFG["modes"], CFG["ctrl_cap_per_day"], CFG["clean_sample_from"])["seckin_h0"][0]
    ctrl = posts.loc[posts["label"].eq(0)].assign(day=lambda f: f["t"] // DAY_S)
    full = world.loc[world["label"].eq(0)].merge(ctrl[["accountid", "day"]].drop_duplicates(), on=["accountid", "day"])
    first = full.sort_values(["t", "accountid"], kind="stable").groupby(["accountid", "day"]).head(100)
    assert ctrl.groupby(["accountid", "day"]).size().max() == 100
    assert set(map(tuple, ctrl[["accountid", "t", "content"]].to_numpy())) == set(map(tuple, first[["accountid", "t", "content"]].to_numpy()))


def test_clean_cung_kich_thuoc_voi_so_control_duy_nhat_cua_seckin(jobs):
    for (_, _), (_, _, modes) in jobs.items():
        n = modes["seckin_h0"][1]["n_ctrl_unique"]
        for mode in ("clean", "rounding_only"):
            assert modes[mode][0].loc[modes[mode][0]["is_control"], "accountid"].nunique() == n


def test_clean_va_rounding_only_trung_nhau_sau_p1(jobs):
    for (_, _), (_, _, modes) in jobs.items():
        clean, rounding = (modes[mode][2]["frame"][["accountid", "t_min", "day", "action", "content", "is_control"]].sort_values(["t_min", "accountid", "content"], kind="stable").reset_index(drop=True) for mode in ("clean", "rounding_only"))
        assert_frame_equal(clean, rounding)
        assert not modes["clean"][0]["post_time"].equals(modes["rounding_only"][0]["post_time"])


# --- Mốc (S1) ---
@pytest.mark.parametrize("p1", ["off", "on"])
def test_tau_q_sau_t0_va_tang_dan(jobs, p1):
    for (_, _), (world, meta, modes) in jobs.items():
        t0 = t0_of(meta)
        for mode in ("clean", "seckin_h0"):
            stream = modes[mode][2]["stream"] if p1 == "on" else seconds_stream(modes[mode][2]["stream"])
            marks = make_marks(stream, coord_times(world), t0, CFG["marks"], p1 == "on")
            q = marks.loc[marks["checkpoint_type"].eq("q")].sort_values("q")
            assert len(q) == len(CFG["marks"]["q_values"])
            assert (q["tau"] >= (t0.floor("min") if p1 == "on" else t0)).all()
            assert q["tau"].is_monotonic_increasing
            assert (q.loc[q["q"].eq(1.0), "tau"] == (coord_times(world).max().floor("min") if p1 == "on" else coord_times(world).max())).all()
            pre = marks.loc[marks["checkpoint_type"].eq("pre_t0")]
            assert (pre["tau"] < t0).all() and (pre["tau_rel_t0"] < 0).all()


def test_tau_q_dem_hanh_dong_phoi_hop_khong_dem_nen(jobs):
    (world, meta, modes) = jobs[(SEEDS[0], 1)]
    times = coord_times(world).sort_values()
    for q in CFG["marks"]["q_values"]:
        assert run_t13.q_tau(times, q) == times[int(np.ceil(q * len(times))) - 1]


def test_moc_ngay_trung_make_checkpoints(jobs):
    rules = EpisodeRules(min_accounts=CFG["marks"]["min_accounts"])
    for (_, _), (world, meta, modes) in jobs.items():
        prepared = modes["seckin_h0"][2]
        assert not prepared["fallback"]
        ours = make_marks(prepared["stream"], coord_times(world), t0_of(meta), CFG["marks"], True)
        theirs = make_checkpoints(prepared["stream"], prepared["episodes"], rules)
        assert sorted(ours.loc[ours["checkpoint_type"].eq("day"), "tau"]) == sorted(theirs.loc[theirs["checkpoint_type"].eq("day"), "tau"])


# --- U2 với thời gian giây (S2) ---
def tiny_posts(rows):
    frame = pd.DataFrame([dict(accountid=user, t=seconds, action="retweet", content=content, label=label, is_coord_action=False, day=seconds // DAY_S) for user, seconds, content, label in rows])
    return to_posts(frame)


def test_u2_giay_hanh_dong_10_05_30_khong_nam_trong_d_10_05_00():
    base = 10 * 3600 + 5 * 60
    rows = [(f"i{k}", base - 120 + k, "p", 1) for k in range(5)] + [(f"c{k}", base - 60 + k, "p", 0) for k in range(5)] + [("i0", base, "p", 1), ("c0", base + 30, "p", 0)]
    prepared = prepare_stream(tiny_posts(rows), EpisodeRules(), 1)
    stream = seconds_stream(prepared["stream"])
    tau = START + pd.Timedelta(seconds=base)
    history, _ = observe(stream, tau, 1)
    assert history["t_min"].max() == tau
    assert not (history["accountid"].eq("c0") & history["post_time"].eq(tau + pd.Timedelta(seconds=30))).any()
    assert len(history) == len(stream) - 1
    checkpoint_pairs(events_seconds(history), ["retweet"], tau)
    with pytest.raises(AssertionError, match="U2"):
        checkpoint_pairs(events_seconds(stream), ["retweet"], tau)
    # Với P1 (làm tròn phút), cùng hành động rơi vào D(10:05:00): đây là lý do p1_methods = off cắt theo giây.
    assert prepared["stream"]["t_min"].le(tau).all()


def test_su_kien_p1_trung_build_events():
    world, meta, modes = world_jobs(SEEDS[0], 1, 2.0, SMALL, CFG)
    stream = modes["seckin_h0"][2]["stream"]
    ours = events_seconds(stream).assign(timestamp=lambda f: f["timestamp"].round().astype(np.int64))
    theirs = events_p1(CFG["layers"])(stream)
    key = ["user_id", "timestamp", "action_type", "content_id"]
    assert_frame_equal(ours[key].sort_values(key).reset_index(drop=True), theirs[key].astype({"timestamp": np.int64, "content_id": str}).sort_values(key).reset_index(drop=True), check_dtype=False)


def test_u2_pass_va_chay_tiep_tren_the_gioi_nho(tmp_path):
    world, meta, modes = world_jobs(SEEDS[0], 3, 2.0, SMALL, CFG)
    path = tmp_path / "t13.csv"
    for mode, p1 in (("clean", "off"), ("seckin_h0", "on")):
        job = dict(world=world, world_meta=meta, prepared=modes[mode][2], mode=mode, p1=p1, scenario="sim3", data_seed=SEEDS[0], shape=2.0, mean=SMALL["mean_actions_per_day"])
        done = read_done_keys(path)
        written = run_mode(job, CFG, path, done, [42], progress=lambda _: None)
        assert written > 0
        assert run_mode(job, CFG, path, read_done_keys(path), [42], progress=lambda _: None) == 0
    frame = pd.read_csv(path)
    assert list(frame.columns) == T13_COLUMNS
    assert not frame.duplicated(["campaign", "episode", "q", "tau", "method", "seed"]).any()
    assert frame["campaign"].iloc[0] == campaign_name("sim3", "clean", "off", 2.0, SMALL["mean_actions_per_day"], SEEDS[0])
    assert {"B0_K2", "B1_goc", "B1_chuan", "TRIV_best"} <= set(frame["method"])
    assert frame.loc[frame["method"].str.startswith("TRIV_") & frame["eligible"], "auc"].notna().all()
    assert frame.loc[frame["checkpoint_type"].eq("full_7d"), "method"].str.startswith("TRIV_").all()


# --- S3 nhanh ---
def test_s3a_nhanh_triv_best_khong_bang_1_truoc_q25(jobs):
    for (seed, sim), (world, meta, modes) in jobs.items():
        stream = seconds_stream(modes["clean"][2]["stream"])
        marks = make_marks(stream, coord_times(world), t0_of(meta), CFG["marks"], False)
        tau25 = marks.loc[marks["q"].eq(0.25), "tau"].iloc[0]
        for mark in marks.loc[marks["tau"].lt(tau25) & marks["eligible"]].to_dict("records"):
            _, current = observe(stream, mark["tau"], mark["episode"])
            best = next(row for row in triv_rows_with_auc(current, scores_seconds, CFG["triv_tiebreak_seed"]) if row["method"] == "TRIV_best")
            assert best["f1_star"] < 1.0, (seed, sim, mark["checkpoint_type"])


# --- summarize_t13: bảng kết quả giả có đáp án biết trước ---
from scripts import summarize_t13 as summ  # noqa: E402

FAKE_SEEDS = CFG["data_seeds"][:6]
# sim1: mọi tiêu chí đạt. sim2: H1 (seckin 0,6), S3b (G trước t0 cao), S3c (B1 = TRIV_LOCO) không đạt. sim3: S3a không đạt (AUC khối lượng clean 0,9), kéo H1 không đạt.
FAKE = {"sim1": dict(clean=0.5, seckin=0.9, b1_pre=None, b1_q1=1.0), "sim2": dict(clean=0.5, seckin=0.6, b1_pre=0.9, b1_q1=0.5), "sim3": dict(clean=0.9, seckin=0.95, b1_pre=None, b1_q1=1.0)}
FAKE_MARKS = [("pre_t0", np.nan, -1440.0), ("q", 0.10, 1.0), ("q", 0.25, 5.0), ("q", 1.0, 20.0), ("full_7d", np.nan, 9000.0)]


def fake_results():
    floor = 2 * 6 / (2 * 6 + 40)
    t0 = START + pd.Timedelta(days=1, hours=8)
    rows = []
    for scenario, spec in FAKE.items():
        for mode in ("clean", "seckin_h0"):
            for index, seed in enumerate(FAKE_SEEDS):
                jitter = 0.001 * index
                for kind, q, rel in FAKE_MARKS:
                    base = {column: np.nan for column in T13_COLUMNS}
                    base.update(campaign=campaign_name(scenario, mode, "off", 2.0, 8.0, seed), episode=1, q=q, tau=t0 + pd.Timedelta(minutes=rel), n_io=6, n_ctrl=40, eligible=True, ds_version="v2", scenario=scenario, config=scenario[3:], mode=mode, p1_methods="off", gamma_shape=2.0, world_mean=8.0, data_seed=seed, tau_rel_t0=rel, checkpoint_type=kind)
                    volume = spec[mode if mode == "clean" else "seckin"] + jitter
                    aucs = {"actions_per_active_day": volume, "first_seen": 0.5 + jitter, "zero_delta_repost": 0.5}
                    for score, auc in aucs.items():
                        rows.append(dict(base, method=f"TRIV_{score}_low", auc=auc, f1_star=0.5))
                        rows.append(dict(base, method=f"TRIV_{score}_high", auc=1 - auc, f1_star=0.4))
                    rows.append(dict(base, method="TRIV_best", f1_star=0.6, auc=0.5))
                    b1 = spec["b1_pre"] if kind == "pre_t0" and spec["b1_pre"] is not None else spec["b1_q1"] if kind == "q" and q == 1.0 else floor
                    for method, value in (("B0_K2", floor), ("B0_K3", floor), ("B1_chuan", b1), ("B1_goc", b1)):
                        rows.append(dict(base, method=method, seed=42, leiden_seed=42, f1_star=value + (jitter if method.startswith("B1") and value != floor else 0)))
    return pd.DataFrame(rows)[T13_COLUMNS]


def fake_cfg():
    return dict(CFG, data_seeds=FAKE_SEEDS)


def test_summarize_tieu_chi_dap_an_biet_truoc():
    cfg = fake_cfg()
    frame = summ.prepare_results(fake_results(), cfg, FAKE_SEEDS)
    result = summ.evaluate_criteria(frame, cfg, len(FAKE_SEEDS))["world_mean=8"]
    assert {s: r["passed"] for s, r in result["S3a"].items()} == {"sim1": True, "sim2": True, "sim3": False}
    assert result["S3a"]["sim3"]["volume_passed"] is False
    assert {s: r["passed"] for s, r in result["S3b"].items()} == {"sim1": True, "sim2": False, "sim3": True}
    assert set(result["S3b"]["sim1"]["by_mark"]) == {"pre1440m"}
    assert "seckin_h0" in result["S3b"]["sim1"]["descriptive"] and "passed" not in result["S3b"]["sim1"]["descriptive"]["seckin_h0"]["pre1440m"]
    assert result["S3c"]["sim1"]["passed"] is True and result["S3c"]["sim1"]["main"]["verdict"] == "vượt"
    assert result["S3c"]["sim2"]["passed"] is False
    assert result["S3c"]["sim3"]["passed"] is None
    assert {s: r["main"]["passed"] for s, r in result["H1"]["by_scenario"].items()} == {"sim1": True, "sim2": False, "sim3": False}
    assert result["H1"]["aggregate"] == "ủng hộ một phần"
    assert result["fallback"]["s3a_volume_failed"] is True and result["fallback"]["eligible_to_trigger"] is True
    assert summ.evaluate_criteria(frame, cfg, len(FAKE_SEEDS) - 1)["world_mean=8"]["fallback"]["eligible_to_trigger"] is False
    h1 = result["H1"]["by_scenario"]["sim1"]["main"]
    assert h1["seckin_auc_min"] == CFG["criteria"]["h1"]["seckin_auc_min"] and h1["ci"][0] > 0


def test_summarize_h1_ung_ho_va_khong_ung_ho():
    cfg = fake_cfg()
    results = fake_results()
    for target, label in ((0.9, "ủng hộ"), (0.5, "không ủng hộ")):
        changed = results.copy()
        vol = changed["method"].eq("TRIV_actions_per_active_day_low")
        changed.loc[vol & changed["mode"].eq("clean"), "auc"] = 0.5
        changed.loc[vol & changed["mode"].eq("seckin_h0"), "auc"] = target + 0.001 * changed.loc[vol & changed["mode"].eq("seckin_h0"), "data_seed"].rank(method="dense")
        frame = summ.prepare_results(changed, cfg, FAKE_SEEDS)
        assert summ.h1(summ.scope(frame, cfg, 8.0), cfg, ["sim1", "sim2", "sim3"])["aggregate"] == label


def test_summarize_bao_loi_thieu_cot_va_thieu_seed():
    results = fake_results()
    with pytest.raises(ValueError, match="thiếu cột"):
        summ.prepare_results(results.drop(columns=["auc"]), fake_cfg(), FAKE_SEEDS)
    dropped = results.loc[~(results["scenario"].eq("sim2") & results["mode"].eq("clean") & results["data_seed"].eq(FAKE_SEEDS[3]))]
    with pytest.raises(ValueError, match="thiếu seed"):
        summ.prepare_results(dropped, fake_cfg(), FAKE_SEEDS)
    frame = summ.prepare_results(results.loc[~results["mode"].eq("seckin_h0")], fake_cfg(), FAKE_SEEDS)
    with pytest.raises(ValueError, match="Thiếu dữ liệu H1"):
        summ.h1(summ.scope(frame, fake_cfg(), 8.0), fake_cfg(), ["sim1"])


def test_summarize_main_no_criteria_khong_ghi_criteria(tmp_path, monkeypatch):
    fake_results().to_csv(tmp_path / "t13_results.csv", index=False)
    pd.DataFrame([dict(variant="no_camouflage", data_seed=FAKE_SEEDS[0], community_seed=42, pipeline="student", configuration="3", f1_star=0.67, homogeneity=1.0, weighted_precision=1.0, nmi=0.5, pattern_ok=True)]).to_csv(tmp_path / "table2_v2.csv", index=False)
    pd.DataFrame([dict(world="modes", data_seed=FAKE_SEEDS[0], scenario="sim1", gamma_shape=2.0, world_mean=8.0, mode="clean", raised_share_coord_days=0.1, raised_share_all_coord_days=0.01, added_action_share=0.001, fill_share=np.nan, dz_io_before_p1=0.0)]).to_csv(tmp_path / "t13_meta.csv", index=False)
    monkeypatch.setattr(summ, "load_config", fake_cfg)
    summ.main(["--input-dir", str(tmp_path), "--no-criteria"])
    assert not (tmp_path / "criteria.json").exists()
    assert {"table2_v2_vs_v1.csv", "auc_by_mode.csv", "models_by_mode_mark.csv", "paired_diffs.csv", "raised_shares.csv", "shape_sensitivity.csv", "leiden_check.csv"} <= {p.name for p in (tmp_path / "summary").iterdir()}
    summ.main(["--input-dir", str(tmp_path)])
    saved = json.loads((tmp_path / "criteria.json").read_text(encoding="utf-8"))
    assert saved["world_mean=8"]["H1"]["aggregate"] == "ủng hộ một phần"
