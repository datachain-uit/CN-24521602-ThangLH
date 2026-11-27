"""T13 (Review-2026-11-21 §6.7, §6.8): DS2-v2 — Bảng 2 v2 (30 × 5 × 2 pipeline × 7 cấu hình × 2 biến thể) và các chế độ × mốc × phương pháp.

Batch theo mốc, không phải cập nhật tăng dần. Chạy tiếp theo khóa như T7: dòng đã có trong CSV thì bỏ qua.
Mọi chế độ đi qua P1 (harmonize_time), P2 (daily_coverage), P3 (make_episodes) không sửa logic; mốc q tự tính trên hành động phối hợp (không dùng make_checkpoints).
B0/B1 gọi iter_campaign_rows của T7; TRIV gọi baseline_scores/oracle_top_k của T9 và _rows_from_scores của T7. Nhãn chỉ dùng cho tập đánh giá và chấm điểm.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.run_t6b_experiment import build_original_graph, build_student_graph, detect_original, detect_student, evaluate, round_timestamps  # noqa: E402
from scripts.run_t7 import git_info, package_versions  # noqa: E402
from src.eval.metrics import clustering_nmi  # noqa: E402
from src.eval.seckin_protocol import MODEL_COLUMNS, EpisodeRules, baseline_scores, build_events, daily_coverage, harmonize_time, make_episodes, oracle_top_k, zero_delta_share  # noqa: E402
from src.eval.t7_checkpoints import EPOCH, KEY_COLUMNS, RESULT_COLUMNS, TRIV_METHODS, _missing, _rows_from_scores, format_tau, iter_campaign_rows, observe, result_key  # noqa: E402
from src.simulation.synthetic_v2 import START, make_modes, make_world, table2_datasets, to_posts, world_background  # noqa: E402

PATHS = json.loads((ROOT / "configs" / "paths.json").read_text(encoding="utf-8"))
EXTRA_COLUMNS = ["ds_version", "scenario", "config", "mode", "ctrl_selection", "ctrl_offset", "p1_methods", "gamma_shape", "world_mean", "data_seed", "leiden_seed", "tau_rel_t0", "checkpoint_type", "auc"]
T13_COLUMNS = [*RESULT_COLUMNS, *EXTRA_COLUMNS]
TABLE2_COLUMNS = ["ds_version", "variant", "data_seed", "community_seed", "pipeline", "configuration", "community_seed_effective", "beta_1", "beta_2", "beta_3", "f1_star", "homogeneity", "weighted_precision", "nmi", "n_communities", "pattern_ok", "runtime_seed_s"]
META_COLUMNS = ["world", "data_seed", "scenario", "gamma_shape", "world_mean", "mode", "t0", "n_coord_actions", "raised_account_days", "coord_account_days", "added_actions", "coord_group_actions", "raised_share_coord_days", "raised_share_all_coord_days", "added_action_share", "fill_share", "n_io", "n_ctrl_unique", "n_posts", "p3_n_episodes", "p3_fallback", "dz_io_before_p1", "dz_ctrl_before_p1", "dz_io_after_p1", "dz_ctrl_after_p1"]
MARK_TYPES = ("q", "day", "pre_t0", "full_7d")


def load_config():
    return json.loads((ROOT / PATHS["t13_config"]).read_text(encoding="utf-8"))


# --- Ghi CSV và chạy tiếp ---
def _text(column, value):
    if column == "tau":
        return format_tau(value)
    if column in ("episode", "seed", "leiden_seed", "data_seed"):
        return "" if _missing(value) else str(int(float(value)))
    return "" if _missing(value) else value


def append_rows(path, rows, columns):
    """Ghi các dòng đúng schema columns (không thừa, không thiếu) rồi flush ngay."""
    path = Path(path)
    new_file = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if new_file:
            writer.writeheader()
        for row in rows:
            if set(row) != set(columns):
                raise ValueError(f"Dòng sai schema: thừa {set(row) - set(columns)}, thiếu {set(columns) - set(row)}.")
            writer.writerow({column: _text(column, row[column]) for column in columns})
        handle.flush()


def read_table(path, columns):
    """Đọc CSV dạng chuỗi; rỗng nếu chưa có; lỗi nếu sai schema."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != columns:
        raise ValueError(f"{path} không đúng schema {len(columns)} cột của T13.")
    return frame


def read_done_keys(path):
    """Khóa T7 (campaign, episode, q, tau, method, seed); tên campaign mã hóa ds_version, kịch bản, chế độ, p1_methods, shape, mean và data seed."""
    return {result_key(row) for row in read_table(path, T13_COLUMNS)[KEY_COLUMNS].to_dict("records")}


# --- P1–P3 và luồng sự kiện ---
def prepare_stream(posts, rules, control_min):
    """P1 → P2 → P3 không sửa logic. Nếu P3 không cho đúng một đợt phủ mọi ngày có dữ liệu thì dùng toàn bộ khung (đợt 1) cho mọi chế độ và ghi p3_fallback."""
    frame = harmonize_time(posts)
    daily = daily_coverage(frame, rules)
    stream, episodes = make_episodes(frame, daily, rules, control_min)
    fallback = len(episodes) != 1 or int(episodes["covered_days"].iloc[0]) != frame["day"].nunique()
    if fallback:
        stream = frame.assign(episode=1).sort_values("t_min", kind="stable")
    return dict(frame=frame, daily=daily, stream=stream, episodes=episodes, fallback=bool(fallback))


def seconds_stream(stream):
    """p1_methods = off: cùng các dòng của luồng P3 nhưng t_min = post_time (độ phân giải giây), sắp lại tăng dần; day giữ theo P1."""
    return stream.assign(t_min=stream["post_time"]).sort_values("t_min", kind="stable")


def events_seconds(history):
    """Sự kiện cho B1 ở p1_methods = off: mỗi bài một nội dung, timestamp = phút từ epoch kiểu thực (giữ giây), cùng phép tính với tau_minutes."""
    return pd.DataFrame({"user_id": history["accountid"].to_numpy(), "timestamp": ((history["t_min"] - EPOCH) / pd.Timedelta(minutes=1)).to_numpy(dtype=float), "action_type": history["action"].to_numpy(), "content_id": history["content"].astype(str).to_numpy()})


def events_p1(layers):
    """Sự kiện cho B1 ở p1_methods = on: đúng nhánh DS1 của T7 (build_events trên t_min của P1)."""
    return lambda history: build_events(history[MODEL_COLUMNS], tuple(layers))


def scores_seconds(current):
    """3 điểm tầm thường như baseline_scores (T9) nhưng Δt = 0 của retweet tính ở độ phân giải giây (p1_methods = off)."""
    users = current.groupby("accountid", sort=True)
    scores = users.agg(first_seen=("t_min", "min"), actions=("t_min", "size"), days=("day", "nunique"))
    retweets = current.loc[current["action"].eq("retweet")]
    events = pd.DataFrame({"user_id": retweets["accountid"].to_numpy(), "timestamp": ((retweets["t_min"] - EPOCH) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64), "action_type": "retweet", "content_id": retweets["content"].astype(str).to_numpy()})
    zero_ratio = zero_delta_share(events)
    scores["first_seen"] = scores["first_seen"].rank(method="dense")
    scores["actions_per_active_day"] = scores["actions"] / scores["days"]
    scores["zero_delta_repost"] = zero_ratio.reindex(scores.index).fillna(0.0)
    return scores[["first_seen", "actions_per_active_day", "zero_delta_repost"]]


def triv_rows_with_auc(current, scores_fn, tiebreak_seed):
    """TRIV_<điểm>_<chiều> và TRIV_best như T7, thêm AUC của từng điểm theo chiều (oracle_top_k của T9). TRIV_best mang AUC của điểm–chiều được chọn."""
    labels = ~current.groupby("accountid")["is_control"].first()
    scores = scores_fn(current)
    rows = _rows_from_scores(scores, labels, tiebreak_seed)
    auc = {f"TRIV_{name}_{found['direction']}": float(found["auc"]) for name, score in scores.items() for found in oracle_top_k(score, labels)}
    return [dict(row, auc=auc[row.get("best_of", row["method"])]) for row in rows]


# --- Mốc ---
def q_tau(coord_times, q):
    """tau_q = thời điểm của hành động phối hợp thứ ceil(q·n) kể từ t0 (n = tổng số hành động phối hợp của kịch bản, chỉ dùng offline để đặt mốc); cùng phép làm tròn số nguyên như make_checkpoints."""
    times = np.sort(np.asarray(coord_times))
    return times[(int(round(q * 100)) * len(times) + 99) // 100 - 1]


def make_marks(stream, coord_times, t0, mcfg, p1_on):
    """Mốc q (hành động phối hợp), mốc ngày (từ đầu đợt P3, như make_checkpoints), mốc trước t0 và full_7d (chỉ chẩn đoán AUC).

    coord_times, t0: Timestamp thật (giây) của thế giới. p1_on: tau làm tròn xuống phút để khớp t_min của P1.
    n_io, n_ctrl, eligible tính trên luồng của chế độ đến tau (>= min_accounts mỗi phía), như T7.
    """
    snap = (lambda tau: tau.floor("min")) if p1_on else (lambda tau: tau)
    episode = int(stream["episode"].iloc[0])
    if stream["episode"].nunique() != 1:
        raise ValueError("Luồng T13 phải có đúng một đợt.")
    start = stream["day"].min()
    span_days = (stream["day"].max() - start).days + 1
    schedule = [("q", q, snap(q_tau(coord_times, q))) for q in mcfg["q_values"]]
    schedule += [("day", np.nan, start + pd.Timedelta(days=d, minutes=-1)) for d in mcfg["day_marks"] if d <= span_days]
    schedule += [("pre_t0", np.nan, snap(t0 - pd.Timedelta(minutes=offset))) for offset in mcfg["pre_t0_offsets_minutes"]]
    schedule += [("full_7d", np.nan, stream["t_min"].max())]
    rows = []
    for kind, q, tau in schedule:
        seen = stream.loc[stream["t_min"].le(tau)]
        n_io = int(seen.loc[~seen["is_control"], "accountid"].nunique())
        n_ctrl = int(seen.loc[seen["is_control"], "accountid"].nunique())
        rows.append(dict(episode=episode, q=q, tau=tau, n_io=n_io, n_ctrl=n_ctrl, eligible=n_io >= mcfg["min_accounts"] and n_ctrl >= mcfg["min_accounts"], pre_t0=kind == "pre_t0", checkpoint_type=kind, tau_rel_t0=(tau - t0) / pd.Timedelta(minutes=1)))
    return pd.DataFrame(rows).sort_values("tau", kind="stable", ignore_index=True)


# --- Một chế độ của một thế giới ---
def campaign_name(scenario, mode, p1, shape, mean, data_seed):
    return f"DS2v2_{scenario}_{mode}_p1{p1}_g{shape:g}_m{mean:g}_d{data_seed}"


def delta_zero_by_label(frame):
    """Tỷ lệ Δt = 0 (trung bình theo tài khoản, mọi lớp) theo nhãn, trước P1 (giây) và sau P1 (phút)."""
    out = {}
    for stage, times in (("before_p1", (frame["post_time"] - EPOCH) // pd.Timedelta(seconds=1)), ("after_p1", (frame["t_min"] - EPOCH) // pd.Timedelta(minutes=1))):
        events = pd.DataFrame({"user_id": frame["accountid"].to_numpy(), "timestamp": times.to_numpy(dtype=np.int64), "action_type": frame["action"].to_numpy(), "content_id": frame["content"].astype(str).to_numpy()})
        share = zero_delta_share(events)
        control = frame.groupby("accountid")["is_control"].first().reindex(share.index)
        out[f"dz_io_{stage}"] = float(share[~control].mean()) if (~control).any() else np.nan
        out[f"dz_ctrl_{stage}"] = float(share[control].mean()) if control.any() else np.nan
    return out


def run_mode(job, cfg, csv_path, done, b1_seeds, progress=print):
    """Chạy B0/B1 (nếu b1_seeds) và TRIV/AUC cho một chế độ × p1_methods; trả số dòng mới. job: dict(posts, prepared, world_meta, mode, p1, scenario, data_seed, shape, mean)."""
    mode_cfg = cfg["modes"][job["mode"]]
    p1_on = job["p1"] == "on"
    prepared = job["prepared"]
    stream = prepared["stream"] if p1_on else seconds_stream(prepared["stream"])
    world = job["world"]
    coord_times = START + pd.to_timedelta(world.loc[world["is_coord_action"], "t"].to_numpy(np.int64), unit="s")
    t0 = START + pd.Timedelta(seconds=int(job["world_meta"]["t0"]))
    marks = make_marks(stream, coord_times, t0, cfg["marks"], p1_on)
    name = campaign_name(job["scenario"], job["mode"], job["p1"], job["shape"], job["mean"], job["data_seed"])
    extra = dict(ds_version=cfg["ds_version"], scenario=job["scenario"], config=job["scenario"][3:], mode=job["mode"], ctrl_selection=mode_cfg["ctrl_selection"], ctrl_offset=mode_cfg["ctrl_offset"], p1_methods=job["p1"], gamma_shape=job["shape"], world_mean=job["mean"], data_seed=job["data_seed"])
    by_tau = {format_tau(mark["tau"]): mark for mark in marks.to_dict("records")}
    written = 0

    def finish(row):
        mark = by_tau[format_tau(row["tau"])]
        return {**row, **extra, "leiden_seed": row["seed"], "tau_rel_t0": mark["tau_rel_t0"], "checkpoint_type": mark["checkpoint_type"], "auc": row.get("auc", np.nan)}

    model_marks = marks.loc[marks["checkpoint_type"].ne("full_7d")]
    if b1_seeds:
        skip = set(done) | {result_key(dict(campaign=name, episode=mark["episode"], q=mark["q"], tau=mark["tau"], method=method, seed=np.nan)) for mark in model_marks.to_dict("records") for method in TRIV_METHODS}
        events_fn = events_p1(cfg["layers"]) if p1_on else events_seconds
        for row in iter_campaign_rows(stream, model_marks, events_fn, name, cfg["t7_params"], cfg["layers"], b1_seeds, done=skip, progress=progress):
            out = finish({key: row[key] for key in RESULT_COLUMNS})
            append_rows(csv_path, [out], T13_COLUMNS)
            done.add(result_key(out))
            written += 1
    scores_fn = baseline_scores if p1_on else scores_seconds
    for mark in marks.to_dict("records"):
        base = dict(campaign=name, episode=mark["episode"], q=mark["q"], tau=mark["tau"], n_io=mark["n_io"], n_ctrl=mark["n_ctrl"], eligible=bool(mark["eligible"]))
        todo = [method for method in TRIV_METHODS if result_key({**base, "method": method, "seed": np.nan}) not in done]
        if not todo:
            continue
        found = {}
        if base["eligible"]:
            clock = perf_counter()
            _, current = observe(stream, mark["tau"], mark["episode"])
            rows = triv_rows_with_auc(current, scores_fn, cfg["triv_tiebreak_seed"])
            share = (perf_counter() - clock) / len(rows)
            found = {row["method"]: dict(row, runtime_s=share) for row in rows}
        out = []
        for method in todo:
            row = {column: np.nan for column in RESULT_COLUMNS}
            row.update(base, method=method, seed=np.nan)
            row.update({key: value for key, value in found.get(method, {}).items() if key in RESULT_COLUMNS or key == "auc"})
            out.append(finish(row))
        append_rows(csv_path, out, T13_COLUMNS)
        done.update(result_key(row) for row in out)
        written += len(out)
    return written


def world_jobs(data_seed, sim, shape, wcfg, cfg, background=None):
    """Thế giới và mọi chế độ của (data seed, kịch bản, shape): trả (world, world_meta, {mode: (posts, info, prepared)})."""
    world, meta = make_world(data_seed, sim, shape, wcfg, background)
    rules = EpisodeRules(min_accounts=cfg["marks"]["min_accounts"])
    out = {}
    for mode, (sample, info) in make_modes(world, meta, cfg["modes"], cfg["ctrl_cap_per_day"], cfg["clean_sample_from"]).items():
        posts = to_posts(sample)
        out[mode] = (posts, info, prepare_stream(posts, rules, cfg["episode_control_min"]))
    return world, meta, out


def meta_row(world_meta, mode, info, prepared, mean):
    frame = prepared["frame"]
    return dict(world="modes", data_seed=world_meta["data_seed"], scenario=world_meta["scenario"], gamma_shape=world_meta["gamma_shape"], world_mean=mean, mode=mode, t0=format_tau(START + pd.Timedelta(seconds=int(world_meta["t0"]))), n_coord_actions=world_meta["n_coord_actions"], **{key: world_meta[key] for key in ("raised_account_days", "coord_account_days", "added_actions", "coord_group_actions", "raised_share_coord_days", "raised_share_all_coord_days", "added_action_share")}, fill_share=info["fill_share"], n_io=int(frame.loc[~frame["is_control"], "accountid"].nunique()), n_ctrl_unique=info["n_ctrl_unique"], n_posts=len(frame), p3_n_episodes=len(prepared["episodes"]), p3_fallback=prepared["fallback"], **delta_zero_by_label(frame))


# --- Bảng 2 v2 (quy trình T6b) ---
def pattern_ok(configuration, metrics, tolerance):
    """Đúng mẫu: H = WP = 1 ở cấu hình 3; F1* = H = WP = 1 ở cấu hình khác."""
    keys = ["homogeneity", "weighted_precision"] if configuration == "3" else ["f1_star", "homogeneity", "weighted_precision"]
    return all(metrics[key] >= 1 - tolerance for key in keys)


def run_table2_seed(data_seed, variant, cfg):
    """Như run_one_seed của T6b (cả mã SV và mã gốc), trên bộ sinh v2; thêm NMI và cờ đúng mẫu. Trả (dòng kết quả, dòng metadata)."""
    clock = perf_counter()
    t2 = cfg["table2"]
    raw, meta = table2_datasets(data_seed, variant, t2)
    datasets = {sim: round_timestamps(events) for sim, events in raw.items()}
    truths = {sim: set(events.loc[events["is_inauthentic"], "user_id"]) for sim, events in datasets.items()}
    node_sets = {sim: set(events["user_id"]) for sim, events in datasets.items()}
    if not (truths[1] == truths[2] == truths[3] and node_sets[1] == node_sets[2] == node_sets[3]):
        raise AssertionError("Tập phối hợp hoặc tập nút khác nhau giữa các mô phỏng.")
    truth, users = truths[1], sorted(node_sets[1])
    graphs, betas = {"student": {}, "original_reference": {}}, {"student": {}, "original_reference": {}}
    for sim, events in datasets.items():
        graphs["student"][sim], betas["student"][sim], _ = build_student_graph(events, t2["community_seeds"])
        graphs["original_reference"][sim], betas["original_reference"][sim], _ = build_original_graph(events, users, data_seed, sim)
    rows = []
    for pipeline in graphs:
        for configuration in t2["configurations"]:
            selected = [graphs[pipeline][sim] for sim in configuration]
            name = "+".join(map(str, configuration))
            deterministic = pipeline == "original_reference" and len(configuration) == 1
            cached = None
            for community_seed in t2["community_seeds"]:
                if not (deterministic and cached is not None):
                    communities = detect_student(selected, community_seed) if pipeline == "student" else detect_original(selected, users, community_seed)
                    metrics = evaluate(communities, truth, users)
                    metrics["nmi"] = clustering_nmi(communities, truth)[0]
                    cached = metrics if deterministic else None
                metrics_now = cached if deterministic else metrics
                rows.append(dict(ds_version=cfg["ds_version"], variant=variant, data_seed=data_seed, community_seed=community_seed, pipeline=pipeline, configuration=name, community_seed_effective=not deterministic, **{f"beta_{sim}": betas[pipeline][sim] if sim in configuration else np.nan for sim in (1, 2, 3)}, **metrics_now, pattern_ok=pattern_ok(name, metrics_now, t2["pattern_tolerance"]), runtime_seed_s=np.nan))
    elapsed = perf_counter() - clock
    for row in rows:
        row["runtime_seed_s"] = elapsed
    metas = [dict({column: np.nan for column in META_COLUMNS}, world=f"table2_{variant}", data_seed=data_seed, scenario=f"sim{int(m['simulation'])}", gamma_shape=t2["gamma_shape"] if variant == "camouflage" else np.nan, world_mean=t2["mean_actions_per_day"] if variant == "camouflage" else np.nan, mode="table2", **{key: value for key, value in m.items() if key in META_COLUMNS}) for m in meta.to_dict("records")]
    return rows, metas


# --- run.json ---
def raised_summary(meta_path):
    """Tỷ lệ bị nâng (i), (ii), (iii) theo thế giới × shape × mean × kịch bản × seed (một dòng mỗi thế giới), chỉ báo cáo."""
    meta = read_table(meta_path, META_COLUMNS)
    if meta.empty:
        return []
    keys = ["world", "gamma_shape", "world_mean", "scenario", "data_seed"]
    columns = ["raised_share_coord_days", "raised_share_all_coord_days", "added_action_share"]
    meta = meta.loc[meta["raised_share_coord_days"].ne("")].drop_duplicates(keys)
    return meta[keys + columns].to_dict("records")


def write_run_json(path, cfg, args, runtimes, outputs, started_utc):
    old = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else {}
    merged = {**old.get("runtime_s_by_part", {}), **runtimes}
    payload = dict(task="T13", review=cfg["review"], started_utc=old.get("started_utc", started_utc), updated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), **git_info(), packages=package_versions(), args=vars(args), config=cfg, runtime_s_by_part=merged, raised_shares=raised_summary(outputs["meta"]), outputs={key: Path(value).name for key, value in outputs.items()})
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# --- Điều phối ---
def parse_args(argv, cfg):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", default=["table2", "modes", "grid"], choices=["table2", "modes", "grid"], help="Phần cần chạy.")
    parser.add_argument("--n-data-seeds", type=int, default=None, help="Chỉ chạy n data seed đầu (thử khói).")
    parser.add_argument("--world-mean", choices=["primary", "fallback"], default="primary", help="fallback = world.mean_actions_per_day đổi sang fallback_mean_actions_per_day (tối đa max_fallbacks lần, theo quy tắc chốt trước).")
    parser.add_argument("--output-dir", default=None, help="Thư mục kết quả (chạy thử); mặc định t13_output_dir.")
    return parser.parse_args(argv)


def main(argv=None):
    for console in (sys.stdout, sys.stderr):
        console.reconfigure(encoding="utf-8")
    cfg = load_config()
    args = parse_args(argv, cfg)
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / PATHS["t13_output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = dict(results=out_dir / "t13_results.csv", table2=out_dir / "table2_v2.csv", meta=out_dir / "t13_meta.csv")
    run_path = out_dir / "run.json"
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seeds = cfg["data_seeds"][: args.n_data_seeds] if args.n_data_seeds else cfg["data_seeds"]
    wcfg = dict(cfg["world"])
    if args.world_mean == "fallback":
        if cfg["max_fallbacks"] < 1:
            raise ValueError("max_fallbacks = 0: không được đổi mean.")
        wcfg["mean_actions_per_day"] = cfg["fallback_mean_actions_per_day"]
    mean, main_shape = float(wcfg["mean_actions_per_day"]), float(wcfg["gamma_shape"])
    done = read_done_keys(outputs["results"])
    meta_done = {tuple(row) for row in read_table(outputs["meta"], META_COLUMNS)[["world", "data_seed", "scenario", "gamma_shape", "world_mean", "mode"]].to_numpy()}
    print(f"Kết quả: {outputs['results']} ({len(done)} dòng đã có).", flush=True)

    if "table2" in args.parts:
        clock = perf_counter()
        table2 = read_table(outputs["table2"], TABLE2_COLUMNS)
        expected = 2 * len(cfg["table2"]["configurations"]) * len(cfg["table2"]["community_seeds"])
        for variant in cfg["table2"]["variants"]:
            for data_seed in seeds:
                if (table2["variant"].eq(variant) & table2["data_seed"].eq(str(data_seed))).sum() == expected:
                    continue
                if (table2["variant"].eq(variant) & table2["data_seed"].eq(str(data_seed))).any():
                    raise RuntimeError(f"table2_v2.csv có dữ liệu dở dang cho {variant}/{data_seed}; xóa các dòng đó rồi chạy lại.")
                rows, metas = run_table2_seed(data_seed, variant, cfg)
                append_rows(outputs["table2"], rows, TABLE2_COLUMNS)
                append_rows(outputs["meta"], [m for m in metas if (m["world"], str(m["data_seed"]), m["scenario"], "" if _missing(m["gamma_shape"]) else str(m["gamma_shape"]), "" if _missing(m["world_mean"]) else str(m["world_mean"]), m["mode"]) not in meta_done], META_COLUMNS)
                print(f"[table2] {variant} seed={data_seed} xong {rows[0]['runtime_seed_s']:.1f}s", flush=True)
        write_run_json(run_path, cfg, args, {"table2": round(perf_counter() - clock, 1)}, outputs, started_utc)

    def run_world(data_seed, shape, specs_b1, specs_triv, background):
        for sim in cfg["scenarios"]:
            world, world_meta, modes = world_jobs(data_seed, sim, shape, wcfg, cfg, background)
            for mode, (posts, info, prepared) in modes.items():
                key = ("modes", str(data_seed), f"sim{sim}", str(float(shape)), str(mean), mode)
                if key not in meta_done:
                    append_rows(outputs["meta"], [meta_row(world_meta, mode, info, prepared, mean)], META_COLUMNS)
                    meta_done.add(key)
            for (mode, p1), with_b1 in [*((spec, True) for spec in specs_b1), *((spec, False) for spec in specs_triv)]:
                check = with_b1 and data_seed in cfg["data_seeds"][: cfg["n_check_seeds"]] and [mode, p1] in cfg["check_runs"]
                b1_seeds = (cfg["leiden_seeds_check"] if check else cfg["leiden_seeds_main"]) if with_b1 else []
                job = dict(world=world, world_meta=world_meta, prepared=modes[mode][2], mode=mode, p1=p1, scenario=f"sim{sim}", data_seed=data_seed, shape=float(shape), mean=mean)
                written = run_mode(job, cfg, outputs["results"], done, b1_seeds, progress=lambda text: print(text, flush=True))
                print(f"[sim{sim}] {mode} p1={p1} shape={shape:g} seed={data_seed}: {written} dòng mới", flush=True)

    if "modes" in args.parts:
        clock = perf_counter()
        for data_seed in seeds:
            run_world(data_seed, main_shape, cfg["runs"]["b1"], cfg["runs"]["triv_only"], world_background(data_seed, main_shape, wcfg))
            write_run_json(run_path, cfg, args, {"modes": round(perf_counter() - clock, 1)}, outputs, started_utc)
    if "grid" in args.parts:
        clock = perf_counter()
        triv_specs = [[mode, "off"] for mode in cfg["modes"]]
        for shape in cfg["gamma_shape_grid"]["values"]:
            if float(shape) == main_shape:
                continue
            for data_seed in seeds:
                run_world(data_seed, float(shape), [], triv_specs, world_background(data_seed, float(shape), wcfg))
            write_run_json(run_path, cfg, args, {"grid": round(perf_counter() - clock, 1)}, outputs, started_utc)
    write_run_json(run_path, cfg, args, {}, outputs, started_utc)


if __name__ == "__main__":
    main()
