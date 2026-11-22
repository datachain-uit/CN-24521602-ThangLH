"""P6 (biến thể của giao thức gốc) và VC (P5b) — Review-2026-11-21 §6.3–6.5.

Chỉ ghép các hàm có sẵn: P1–P5 và prepare_variant (T9), ghép cặp/β/Leiden/TRIV (T7), partition_metrics (T7). Không đọc nhãn khi suy luận.
Mỗi đơn vị (protocol, campaign, variant, method, episode, q, seed) + config_hash là một dòng CSV, ghi ngay và fsync.
"""
import csv
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.eval.metrics import partition_metrics
from src.eval.seckin_protocol import BASELINES, MODEL_COLUMNS, build_events, control_cluster_summary, daily_coverage, evaluate_baselines, make_checkpoints, make_episodes, prepare_variant
from src.eval.t7_checkpoints import TRIV_METHODS, _int_text, _missing, assert_not_future, build_b1_graphs, checkpoint_pairs, format_tau, leiden_multiplex, observe, trivial_rows

COLUMNS = ["protocol", "campaign", "variant", "method", "episode", "checkpoint_type", "checkpoint_value", "q", "seed", "config_hash", "tau", "tau_goc", "tau_vc", "eligible", "eligible_goc", "eligible_vc", "n_io", "n_ctrl", "io_actions_episode", "f1_star", "precision_star", "recall_star", "wp", "homogeneity", "nmi", "nmi50", "n_clusters_ge2", "n_io_clusters_ge2", "largest_io_cluster_share", "auc", "oracle_k", "beta_retweet", "beta_hashtag", "beta_mention", "beta_url", "runtime_s"]
KEY_COLUMNS = ["protocol", "campaign", "variant", "method", "episode", "checkpoint_type", "checkpoint_value", "q", "seed", "config_hash"]
QUAL_COLUMNS = ["campaign", "episode", "q", "tau", "seed", "config_hash", "rank", "accounts", "io_accounts", "control_accounts", "control_share", "synchronized_events", "synchronized_days", "repeated_sync", "shared_retweet", "shared_hashtag", "shared_mention", "shared_url"]
QUAL_KEY_COLUMNS = ["campaign", "episode", "q", "seed", "config_hash", "rank"]
B1_KEEP_ZERO = {"B1_chuan": True, "B1_goc": False}
P5_METHODS = [f"P5_{score}_{direction}" for score in BASELINES for direction in ("high", "low")]
MARK_COLUMNS = ["episode", "checkpoint_type", "checkpoint_value", "q", "tau", "tau_goc", "tau_vc", "eligible", "eligible_goc", "eligible_vc", "n_io", "n_ctrl", "io_actions_episode"]


# --- Luồng và mốc ---
def cap_volume(stream, k):
    """VC (Review-2026-11-21 §6.3): giữ k hành động đầu tiên mỗi tài khoản–ngày, sắp t_min ổn định; kết quả vẫn sắp tăng theo t_min."""
    if int(k) < 1:
        raise ValueError("K_c phải là số nguyên dương.")
    return stream.sort_values("t_min", kind="stable").groupby(["accountid", "day"], sort=False).head(int(k))


def vc_stream(stream, daily, episodes, k, rules, control_min):
    """Áp cap_volume lên luồng sau make_episodes; assert ngày được phủ và đợt (start, end, qualified) không đổi khi tính lại trên luồng đã cap."""
    capped = cap_volume(stream, k)
    if not capped["t_min"].is_monotonic_increasing:
        raise AssertionError("Luồng VC không còn sắp tăng theo t_min.")
    column = f"covered_c{control_min}"
    daily_vc = daily_coverage(capped, rules)
    covered, covered_vc = daily.index[daily[column]], daily_vc.index[daily_vc[column]]
    if not covered.equals(covered_vc) or len(capped[["accountid", "day"]].drop_duplicates()) != len(stream[["accountid", "day"]].drop_duplicates()):
        raise AssertionError("VC làm đổi ngày được phủ hoặc mất tài khoản–ngày.")
    _, episodes_vc = make_episodes(capped.drop(columns="episode"), daily_vc, rules, control_min)
    keep = ["start", "end", "qualified"]
    if not episodes_vc[keep].equals(episodes[keep]):
        raise AssertionError("VC làm đổi đợt (start, end, qualified).")
    return capped


def io_marks(checkpoints, q_values, day_values=()):
    """Mốc io_pct có q thuộc q_values và mốc theo ngày có checkpoint_value thuộc day_values (q = NaN), thêm cột tau_goc/eligible_goc; cột VC để NaN."""
    is_q = checkpoints["checkpoint_type"].eq("io_pct") & np.isclose(checkpoints["q"].to_numpy(dtype=float)[:, None], np.asarray(q_values, dtype=float)[None, :]).any(axis=1)
    is_day = checkpoints["checkpoint_type"].eq("day") & checkpoints["checkpoint_value"].isin(list(day_values))
    marks = checkpoints.loc[is_q | is_day]
    marks = marks.assign(tau_goc=marks["tau"], eligible_goc=marks["eligible"].astype(bool), tau_vc=pd.NaT, eligible_vc=np.nan)
    return marks[MARK_COLUMNS].reset_index(drop=True)


def vc_marks(capped, episodes, rules, q_values, goc_marks, day_values=()):
    """Mốc tính lại trên luồng đã cap (tau_vc, eligible_vc; mốc theo ngày giữ tau vì chỉ phụ thuộc ngày bắt đầu đợt); ghép tau_goc, eligible_goc theo (episode, checkpoint_type, checkpoint_value). Mốc chạy theo tau_vc và eligible_vc."""
    mark = ["episode", "checkpoint_type", "checkpoint_value"]
    vc = io_marks(make_checkpoints(capped, episodes, rules), q_values, day_values)[[*mark, "q", "tau", "eligible", "n_io", "n_ctrl", "io_actions_episode"]]
    goc = goc_marks[[*mark, "tau_goc", "eligible_goc"]]
    merged = vc.merge(goc, on=mark, how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise AssertionError("Tập mốc (đợt, loại, giá trị) của VC khác giao thức gốc.")
    merged = merged.assign(tau_vc=merged["tau"], eligible_vc=merged["eligible"].astype(bool), eligible=merged["eligible"].astype(bool))
    return merged[MARK_COLUMNS].reset_index(drop=True)


# --- Khóa và dòng ---
def _q_text(value):
    return "nan" if _missing(value) else f"{float(value):.4f}"


def unit_key(row):
    """(protocol, campaign, variant, method, episode, checkpoint_type, checkpoint_value, q, seed, config_hash) chuẩn hóa thành chuỗi; seed, q thiếu = "nan"."""
    return (str(row["protocol"]), str(row["campaign"]), str(row["variant"]), str(row["method"]), _int_text(row["episode"]), str(row["checkpoint_type"]), _int_text(row["checkpoint_value"]), _q_text(row["q"]), _int_text(row["seed"]), str(row["config_hash"]))


def mark_key(protocol, campaign, variant, method, mark, seed, config_hash):
    """Dict khóa của một đơn vị tại mốc `mark`."""
    return dict(protocol=protocol, campaign=campaign, variant=variant, method=method, episode=mark["episode"], checkpoint_type=mark["checkpoint_type"], checkpoint_value=mark["checkpoint_value"], q=mark["q"], seed=seed, config_hash=config_hash)


def qual_key(row):
    return (str(row["campaign"]), _int_text(row["episode"]), _q_text(row["q"]), _int_text(row["seed"]), str(row["config_hash"]), _int_text(row["rank"]))


def make_row(protocol, campaign, variant, method, mark, seed, config_hash, metrics=None, runtime=float("nan"), betas=None):
    row = dict.fromkeys(COLUMNS, float("nan"))
    row.update({column: mark[column] for column in MARK_COLUMNS})
    row.update(protocol=protocol, campaign=campaign, variant=variant, method=method, seed=seed, config_hash=config_hash, runtime_s=runtime)
    row.update({key: value for key, value in (metrics or {}).items() if key in COLUMNS and key not in KEY_COLUMNS and key not in MARK_COLUMNS})
    row.update(betas or {})
    return row


def p6_units(mark, variants, seeds):
    """(variant, seed) của một mốc P6; mốc không đạt: một dòng mỗi biến thể, seed NaN."""
    if not bool(mark["eligible"]):
        return [(variant, float("nan")) for variant in variants]
    return [(variant, seed) for variant in variants for seed in seeds]


def method_units(mark, b1_methods, seeds, triv, p5):
    """(method, seed) của một mốc VC/GOC: B1 theo seed (mốc không đạt: seed NaN), TRIV và P5 seed NaN."""
    nan = float("nan")
    units = [(method, seed) for method in b1_methods for seed in seeds] if bool(mark["eligible"]) else [(method, nan) for method in b1_methods]
    units += [(method, nan) for method in TRIV_METHODS] if triv else []
    units += [(method, nan) for method in P5_METHODS] if p5 else []
    return units


# --- Tính toán ---
def b1_partitions(events, nodes, layers, tau, plan, cfg, prep_s=0.0):
    """plan: {tên B1: [seed]}. Ghép cặp một lần tại tau; chọn β và dựng đồ thị một lần mỗi biến thể B1 (β với community_seeds[0]); Leiden mỗi seed.

    Trùng infer_partition gọi theo từng seed (β không phụ thuộc seed Leiden). runtime_s = (chuẩn bị + ghép cặp) chia đều cho mọi đơn vị + dựng đồ thị / số seed của biến thể + Leiden của seed.
    Trả lần lượt (method, seed, partition, betas, runtime_s).
    """
    units = sum(len(seeds) for seeds in plan.values())
    if events.empty:
        for name, seeds in plan.items():
            for seed in seeds:
                yield name, seed, [{node} for node in nodes], {}, prep_s / units
        return
    clock = perf_counter()
    pairs = checkpoint_pairs(events, layers, tau)
    shared = prep_s + perf_counter() - clock
    for name, seeds in plan.items():
        clock = perf_counter()
        graphs, info = build_b1_graphs(pairs, nodes, B1_KEEP_ZERO[name], tau, tuple(cfg["beta_bounds"]), cfg["community_seeds"][0], cfg["edge_tolerance"], cfg["brent_xatol"])
        build_s = perf_counter() - clock
        betas = {f"beta_{layer}": info[layer]["beta"] for layer in layers}
        weights = [info[layer]["edges_author"] for layer in layers]
        for seed in seeds:
            clock = perf_counter()
            partition = leiden_multiplex(graphs, nodes, seed, weights)
            yield name, seed, partition, betas, shared / units + build_s / len(seeds) + perf_counter() - clock


def p6_rows(stream, mark, campaign, variants, seeds, cap_k, cfg, config_hash, done, qualitative=None):
    """Sinh dòng P6 (B1-chuẩn) của một mốc, bỏ qua khóa trong done. qualitative(dict seed, callback) — gọi callback(rows) với bảng cụm control của biến thể main ở seed đó."""
    todo = [(variant, seed) for variant, seed in p6_units(mark, variants, seeds) if unit_key(mark_key("P6", campaign, variant, "B1_chuan", mark, seed, config_hash)) not in done]
    if not todo:
        return
    if not bool(mark["eligible"]):
        yield from (make_row("P6", campaign, variant, "B1_chuan", mark, seed, config_hash) for variant, seed in todo)
        return
    tau = mark["tau"]
    history, current = observe(stream, tau, mark["episode"])
    population = set(current["accountid"])
    truth = set(current.loc[~current["is_control"], "accountid"])
    for variant in variants:
        variant_seeds = [seed for name, seed in todo if name == variant]
        groups = [[seed] for seed in variant_seeds] if variant == "shuffle_account_day" else [variant_seeds] if variant_seeds else []
        for group in groups:
            clock = perf_counter()
            posts, layers = prepare_variant(history, current, variant, cap_k, group[0])
            assert_not_future(posts["t_min"], tau, f"P6 {variant}")
            nodes = sorted((current if variant == "reset_state" else history)["accountid"].unique())
            events = build_events(posts, layers)
            for _, seed, partition, betas, runtime in b1_partitions(events, nodes, layers, tau, {"B1_chuan": group}, cfg, perf_counter() - clock):
                if set().union(*partition) != set(nodes) or sum(map(len, partition)) != len(nodes):
                    raise ValueError("Phân hoạch phải chứa đúng một lần mỗi tài khoản được phân hoạch.")
                if qualitative is not None and variant == "main" and seed == qualitative["seed"]:
                    restricted = [cluster & population for cluster in partition if cluster & population]
                    qualitative["callback"](control_cluster_summary(restricted, truth, events, limit=qualitative["top"]))
                yield make_row("P6", campaign, variant, "B1_chuan", mark, seed, config_hash, partition_metrics(partition, population, truth), runtime, betas)


def method_rows(protocol, variant, stream, mark, campaign, b1_methods, seeds, triv, p5, cfg, config_hash, done):
    """Sinh dòng VC/GOC của một mốc: B1 (b1_methods × seed), TRIV (trivial_rows của T7), P5 (evaluate_baselines của T9). Bỏ qua khóa trong done."""
    nan = float("nan")
    todo = [(method, seed) for method, seed in method_units(mark, b1_methods, seeds, triv, p5) if unit_key(mark_key(protocol, campaign, variant, method, mark, seed, config_hash)) not in done]
    if not todo:
        return
    if not bool(mark["eligible"]):
        yield from (make_row(protocol, campaign, variant, method, mark, seed, config_hash) for method, seed in todo)
        return
    tau = mark["tau"]
    history, current = observe(stream, tau, mark["episode"])
    population = set(current["accountid"])
    truth = set(current.loc[~current["is_control"], "accountid"])
    b1_todo = [(method, seed) for method, seed in todo if method in B1_KEEP_ZERO]
    if b1_todo:
        clock = perf_counter()
        nodes = sorted(history["accountid"].unique())
        events = build_events(history[MODEL_COLUMNS], tuple(cfg["ds1_layers"]))
        plan = {method: [seed for name, seed in b1_todo if name == method] for method in b1_methods if any(name == method for name, _ in b1_todo)}
        for name, seed, partition, betas, runtime in b1_partitions(events, nodes, cfg["ds1_layers"], tau, plan, cfg, perf_counter() - clock):
            yield make_row(protocol, campaign, variant, name, mark, seed, config_hash, partition_metrics(partition, population, truth), runtime, betas)
    triv_todo = {method for method, _ in todo if method in TRIV_METHODS}
    if triv_todo:
        clock = perf_counter()
        rows = trivial_rows(current, ~current.groupby("accountid")["is_control"].first(), tau)
        share = (perf_counter() - clock) / len(rows)
        yield from (make_row(protocol, campaign, variant, row["method"], mark, nan, config_hash, row, share) for row in rows if row["method"] in triv_todo)
    p5_todo = {method for method, _ in todo if method in P5_METHODS}
    if p5_todo:
        assert_not_future(current["t_min"], tau, f"{protocol} P5")
        clock = perf_counter()
        found = evaluate_baselines(stream, pd.DataFrame([dict(episode=mark["episode"], tau=tau, eligible=True)]))
        share = (perf_counter() - clock) / len(found)
        for row in found.to_dict("records"):
            method = f"P5_{row['method']}_{row['direction']}"
            if method in p5_todo:
                yield make_row(protocol, campaign, variant, method, mark, nan, config_hash, dict(auc=row["auc"], f1_star=row["F1_star"], oracle_k=row["oracle_k"]), share)


# --- Ghi liên tục và đọc lại ---
def _cell(column, value):
    if column in ("tau", "tau_goc", "tau_vc"):
        return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else format_tau(value)
    if column in ("episode", "checkpoint_value", "seed", "rank"):
        return "" if _missing(value) else _int_text(value)
    if column in ("eligible", "eligible_goc", "eligible_vc", "repeated_sync"):
        return "" if _missing(value) else str(bool(value))
    return "" if _missing(value) else value


def append_rows(path, rows, columns=COLUMNS):
    """Ghi các dòng đúng schema (mode "a"), header nếu tệp rỗng; flush + os.fsync trước khi trả về."""
    path = Path(path)
    for row in rows:
        if set(row) != set(columns):
            raise ValueError(f"Dòng sai schema: thừa {set(row) - set(columns)}, thiếu {set(columns) - set(row)}.")
    new_file = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: _cell(column, row[column]) for column in columns})
        handle.flush()
        os.fsync(handle.fileno())


def read_shard(path, columns=COLUMNS, key_fn=unit_key, warn=print, repair=True):
    """Đọc shard, trả tập khóa đã xong. Dòng cuối thiếu ký tự xuống dòng, sai số cột hoặc không đọc được khóa = cắt dở: cảnh báo và cắt khỏi tệp để lần ghi sau không dính vào.

    repair=False (chế độ kiểm tra): chỉ bỏ qua dòng cắt dở, không sửa tệp. Dòng hỏng ở giữa, sai header hoặc khóa trùng: báo lỗi.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)

    def parse(line):
        try:
            values = next(csv.reader([line.decode("utf-8")]))
            if len(values) != len(columns):
                return None
            row = dict(zip(columns, values))
            return row, key_fn(row)
        except (UnicodeDecodeError, ValueError, StopIteration):
            return None

    last = lines[-1]
    if not last.endswith(b"\n") or (len(lines) > 1 and parse(last) is None):
        warn(f"CẢNH BÁO: {path.name}: dòng cuối bị cắt dở ({len(last)} byte), bỏ{' và cắt khỏi tệp' if repair else ' (không sửa tệp)'}.")
        if repair:
            with path.open("r+b") as handle:
                handle.truncate(len(data) - len(last))
                handle.flush()
                os.fsync(handle.fileno())
        lines = lines[:-1]
        if not lines:
            return set()
    header = next(csv.reader([lines[0].decode("utf-8")]))
    if header != list(columns):
        raise ValueError(f"{path} sai header.")
    keys = set()
    for number, line in enumerate(lines[1:], 2):
        parsed = parse(line)
        if parsed is None:
            raise ValueError(f"{path}: dòng {number} hỏng (không phải dòng cuối).")
        if parsed[1] in keys:
            raise ValueError(f"{path}: khóa trùng ở dòng {number}: {parsed[1]}.")
        keys.add(parsed[1])
    return keys


def qualitative_rows(rows, mark, campaign, seed, config_hash):
    """Gắn khóa mốc vào các dòng control_cluster_summary (chỉ số đếm, không accountid)."""
    return [dict(campaign=campaign, episode=mark["episode"], q=mark["q"], tau=mark["tau"], seed=seed, config_hash=config_hash, **{key: row[key] for key in QUAL_COLUMNS[6:]}) for row in rows]


def ceil_k(value):
    """ceil có dung sai 1e-9 cho đối chiếu K với trung vị/P90 đọc từ CSV."""
    return int(math.ceil(float(value) - 1e-9))
