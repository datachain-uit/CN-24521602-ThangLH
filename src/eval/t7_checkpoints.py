"""T7: khung đánh giá theo mốc, bản batch theo checkpoint (Review 15/11, Mục 6.6). Không phải cập nhật tăng dần."""
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import pandas as pd

from src.community.modularity import select_beta_brent
from src.eval.metrics import partition_metrics
from src.eval.seckin_protocol import baseline_scores, oracle_top_k, zero_delta_share
from src.graph.time_aware import build_layer_graph_from_pairs, prepare_action_pairs

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def tau_minutes(tau):
    """Đổi tau sang phút: Timestamp (DS1) tính từ epoch UTC như build_events; số thực (DS2) giữ nguyên."""
    return (tau - EPOCH) / pd.Timedelta(minutes=1) if isinstance(tau, pd.Timestamp) else float(tau)


def assert_not_future(times, tau, where=""):
    """U2: mọi thời điểm đã đọc phải <= tau; dùng raise để không bị tắt bởi python -O."""
    values = pd.Series(times)
    if len(values) and values.max() > tau:
        raise AssertionError(f"U2 vi phạm{' tại ' + where if where else ''}: max(t)={values.max()} > tau={tau}")


def observe(stream, tau, episode=None, time_col="t_min"):
    """Trả D(tau) = mọi hành động t <= tau (gồm các đợt trước) và phần thuộc đợt hiện tại để lấy tập đánh giá."""
    times = stream[time_col]
    if not times.is_monotonic_increasing:
        raise ValueError(f"Luồng phải sắp tăng dần theo {time_col}.")
    history = stream.iloc[:times.searchsorted(tau, side="right")]
    assert_not_future(history[time_col], tau, "observe")
    current = history if episode is None else history.loc[history["episode"].eq(episode)]
    return history, current


def checkpoint_pairs(events, layers, tau):
    """Ghép cặp một lần tại tau cho mỗi lớp (keep_zero=True, có cột t); lớp là giá trị cột action_type, n_k tính trên D(tau)."""
    assert_not_future(events["timestamp"], tau_minutes(tau), "checkpoint_pairs")
    return {layer: prepare_action_pairs(events, layer, keep_zero=True, with_time=True) for layer in layers}


def variant_pairs(pairs, keep_zero):
    """B1-chuẩn giữ Δt = 0; B1-gốc bỏ Δt = 0 sau khi ghép, trùng prepare_action_pairs(keep_zero=False)."""
    return pairs if keep_zero else pairs.loc[pairs["dd"].ne(0)].reset_index(drop=True)


def layer_diagnostics(pairs, used, graph):
    """Chỉ đếm, không đổi kết quả: cặp chỉ có Δt = 0, cạnh bị ngưỡng loại, số cạnh của ta và theo cách đếm của tác giả.

    Tác giả sinh cạnh cho mọi cặp tài khoản có chung nội dung (models.py:283-315), kể cả cạnh trọng số 0 khi mọi Δt = 0;
    mỗi cặp như vậy luôn có ít nhất một lần ghép khi keep_zero=True, nên số cạnh của tác giả = số cặp (u, v) khác nhau trong pairs.
    """
    zero_only = pairs["dd"].eq(0).groupby([pairs["u"], pairs["v"]]).all() if len(pairs) else pd.Series(dtype=bool)
    before = len(used[["u", "v"]].drop_duplicates())
    return dict(edges_author=len(zero_only), zero_only_pairs=int(zero_only.sum()), edges_before_tolerance=before, dropped_by_tolerance=before - graph.number_of_edges(), edges=graph.number_of_edges())


def build_b1_graphs(pairs_by_layer, nodes, keep_zero, tau, bounds, beta_seed, tolerance=1e-9, xatol=1e-5):
    """Dựng các lớp B1 tại tau: chọn beta riêng mỗi lớp trên D(tau) rồi dựng đồ thị; trả (graphs theo thứ tự lớp, info theo lớp)."""
    graphs, info = [], {}
    for layer, pairs in pairs_by_layer.items():
        used = variant_pairs(pairs, keep_zero)
        assert_not_future(used["t"], tau_minutes(tau), "build_b1_graphs")
        started = perf_counter()
        beta, q, nfev = select_beta_brent(used, tau, bounds, beta_seed, tolerance)
        brent_s = perf_counter() - started
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        if not np.isnan(beta):
            graph = build_layer_graph_from_pairs(used, nodes, beta, tolerance)
        if graph.number_of_nodes() != len(nodes):
            raise ValueError(f"Lớp {layer} có tài khoản ngoài tập nút của D(tau).")
        graphs.append(graph)
        info[layer] = dict(beta=beta, q=q, nfev=nfev, brent_s=brent_s, at_bound=bool(beta >= bounds[1] - xatol), **layer_diagnostics(pairs, used, graph))
    return graphs, info


def leiden_multiplex(graphs, nodes, seed, layer_weights):
    """Leiden đa lớp như scores.py:1620-1675: igraph đủ nút, weights="weight", SurpriseVertexPartition, n_iterations=-1.

    layer_weights do người gọi truyền (edges_author, đếm như tác giả). Lớp không còn cạnh nào bị bỏ khỏi multiplex vì không đóng góp trọng số.
    Cụm trả về theo thứ tự mã cộng đồng tăng dần như vòng lặp comm_id của tác giả (scores.py:1694); mọi nút xuất hiện đúng một lần.
    """
    index = {node: i for i, node in enumerate(nodes)}
    layers, weights = [], []
    for graph, weight in zip(graphs, layer_weights, strict=True):
        edges = list(graph.edges(data="weight"))
        if not edges:
            continue
        layer = ig.Graph(n=len(nodes), edges=[(index[u], index[v]) for u, v, _ in edges])
        layer.es["weight"] = [float(w) for _, _, w in edges]
        layers.append(layer)
        weights.append(float(weight))
    if not layers:
        return [{node} for node in nodes]
    membership, _ = la.find_partition_multiplex(layers, la.SurpriseVertexPartition, weights="weight", layer_weights=weights, n_iterations=-1, seed=seed)
    groups = {}
    for node, community in zip(nodes, membership):
        groups.setdefault(community, set()).add(node)
    return [groups[community] for community in sorted(groups)]


def load_config():
    """Đọc configs/t7.json qua khóa t7_config của configs/paths.json."""
    root = Path(__file__).resolve().parents[2]
    paths = json.loads((root / "configs" / "paths.json").read_text(encoding="utf-8"))
    return json.loads((root / paths["t7_config"]).read_text(encoding="utf-8"))


def infer_partition(events, nodes, layers, seed, keep_zero=True, cfg=None):
    """Adapter P6 của T9 (evaluate_ablations đã kiểm t <= tau trước khi gọi, nên tau = timestamp lớn nhất của events). Không nhận nhãn."""
    cfg = load_config() if cfg is None else cfg
    if events.empty:
        return [{node} for node in nodes]
    tau = float(events["timestamp"].max())
    pairs = checkpoint_pairs(events, layers, tau)
    graphs, info = build_b1_graphs(pairs, nodes, keep_zero, tau, tuple(cfg["beta_bounds"]), cfg["community_seeds"][0], cfg["edge_tolerance"], cfg["brent_xatol"])
    return leiden_multiplex(graphs, nodes, seed, [info[layer]["edges_author"] for layer in layers])


def random_partition(nodes, n_clusters, seed):
    """B0: nhãn ngẫu nhiên đều trong {0..K-1} cho tài khoản (sắp theo ID) bằng numpy default_rng(seed); cụm theo nhãn tăng dần, bỏ nhãn rỗng. Không dùng nhãn IO."""
    ordered = sorted(nodes)
    labels = np.random.default_rng(seed).integers(0, n_clusters, len(ordered))
    return [{node for node, label in zip(ordered, labels) if label == k} for k in range(n_clusters) if (labels == k).any()]


def trivial_rows(current, labels, tau):
    """F1* tầm thường (Mục 6.5) trên tập đánh giá trong D(tau): 3 điểm số của baseline_scores (T9), mỗi điểm 2 chiều, cộng TRIV_best.

    Nhãn chỉ dùng để chấm. k* lấy từ oracle_top_k (T9: sort_index rồi sắp ổn định giảm dần, argmax = k nhỏ nhất khi hòa);
    đường TP(k) được tính lại theo đúng thứ tự đó để lấy TP chính xác, rồi kiểm k* và F1 khớp oracle_top_k (sai số < 1e-12).
    """
    assert_not_future(current["t_min"], tau, "trivial_rows")
    return _rows_from_scores(baseline_scores(current), labels, load_config()["triv_tiebreak_seed"])


def _alias_ids(index, seed):
    """Đổi accountid thành bí danh theo hoán vị ngẫu nhiên có seed (chỉ dùng cho TRIV), để phá hòa trong oracle_top_k không tương quan với nhãn qua thứ tự ID."""
    ids = sorted(index)
    order = np.random.default_rng(seed).permutation(len(ids))
    width = len(str(len(ids)))
    return {account: f"{position:0{width}d}" for account, position in zip(ids, order)}


def _rows_from_scores(scores, labels, tiebreak_seed):
    """Chấm 3 điểm số × 2 chiều bằng oracle_top_k trên ID đã đổi bí danh, tính lại đường TP(k) để lấy TP chính xác, thêm TRIV_best."""
    alias = _alias_ids(scores.index, tiebreak_seed)
    scores = scores.rename(index=alias)
    labels = labels.reindex(list(alias)).rename(index=alias)
    rows = []
    for name, score in scores.items():
        y = labels.reindex(score.index).astype(int)
        n_io = int(y.sum())
        for found in oracle_top_k(score, labels):
            sign = 1 if found["direction"] == "high" else -1
            order = (sign * score).sort_index().sort_values(ascending=False, kind="mergesort").index
            tp = y.loc[order].cumsum().to_numpy()
            f1 = 2 * tp / (np.arange(1, len(tp) + 1) + n_io)
            k = int(found["oracle_k"])
            if int(np.argmax(f1)) + 1 != k or abs(f1[k - 1] - found["F1_star"]) >= 1e-12:
                raise AssertionError(f"Đường TP(k) của {name}/{found['direction']} không khớp oracle_top_k.")
            rows.append(dict(method=f"TRIV_{name}_{found['direction']}", f1_star=float(f1[k - 1]), precision_star=float(tp[k - 1] / k), recall_star=float(tp[k - 1] / n_io), oracle_k=k))
    best = max(rows, key=lambda row: row["f1_star"])
    rows.append({**best, "method": "TRIV_best", "best_of": best["method"]})
    return rows


# --- Phần 8: vòng lặp đợt × mốc × method × seed và CSV dạng dài ---
RESULT_COLUMNS = ["campaign", "episode", "q", "tau", "method", "seed", "n_io", "n_ctrl", "eligible", "f1_star", "precision_star", "recall_star", "wp", "homogeneity", "nmi", "nmi50", "runtime_s", "beta_retweet", "beta_hashtag", "beta_mention", "beta_url", "beta_sim1", "beta_sim2", "beta_sim3", "n_clusters_ge2", "n_io_clusters_ge2", "largest_io_cluster_share"]
KEY_COLUMNS = ["campaign", "episode", "q", "tau", "method", "seed"]
B1_VARIANTS = (("B1_goc", False), ("B1_chuan", True))
TRIV_METHODS = [f"TRIV_{score}_{direction}" for score in ("first_seen", "actions_per_active_day", "zero_delta_repost") for direction in ("high", "low")] + ["TRIV_best"]


def _missing(value):
    """True với None, NaN và chuỗi rỗng hoặc "nan" (giá trị đọc lại từ CSV)."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("", "nan")
    return isinstance(value, float) and math.isnan(value)


def _int_text(value):
    """42, 42.0 và "42.0" đều thành "42"; giá trị thiếu thành "nan"."""
    return "nan" if _missing(value) else str(int(float(value)))


def format_tau(value):
    """Timestamp (DS1) thành "%Y-%m-%dT%H:%M:%SZ" theo UTC; số (DS2, phút) thành 6 chữ số thập phân; chuỗi đọc lại được đưa về một trong hai dạng đó."""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            value = pd.Timestamp(value)
    if isinstance(value, pd.Timestamp):
        value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{float(value):.6f}"


def result_key(row):
    """Khóa append (campaign, episode, q, tau, method, seed) chuẩn hóa thành chuỗi; NaN của q và seed so bằng nhau."""
    q = "nan" if _missing(row["q"]) else f"{float(row['q']):.4f}"
    return (str(row["campaign"]), _int_text(row["episode"]), q, format_tau(row["tau"]), str(row["method"]), _int_text(row["seed"]))


def read_done_keys(path):
    """Tập khóa đã có trong CSV kết quả (rỗng nếu chưa có file); báo lỗi nếu file sai schema."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != RESULT_COLUMNS:
        raise ValueError(f"{path} không đúng schema {len(RESULT_COLUMNS)} cột của T7.")
    return {result_key(row) for row in frame[KEY_COLUMNS].to_dict("records")}


def _cell(column, value):
    if column == "tau":
        return format_tau(value)
    if column in ("episode", "seed"):
        return "" if _missing(value) else _int_text(value)
    return "" if _missing(value) else value


def append_result(path, row):
    """Ghi một dòng đúng RESULT_COLUMNS (27 cột) (không cột thừa) rồi flush ngay để dừng giữa chừng không mất dữ liệu."""
    if set(row) != set(RESULT_COLUMNS):
        raise ValueError(f"Dòng kết quả sai schema: thừa {set(row) - set(RESULT_COLUMNS)}, thiếu {set(RESULT_COLUMNS) - set(row)}.")
    path = Path(path)
    new_file = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({column: _cell(column, row[column]) for column in RESULT_COLUMNS})
        handle.flush()


def _row(base, method, seed, metrics=None, runtime=float("nan"), betas=None):
    row = {column: float("nan") for column in RESULT_COLUMNS}
    row.update(base)
    row.update(method=method, seed=seed, runtime_s=runtime)
    row.update({key: value for key, value in (metrics or {}).items() if key in RESULT_COLUMNS})
    row.update(betas or {})
    return row


def iter_campaign_rows(stream, marks, events_fn, campaign, cfg, layers, seeds, done=frozenset(), trivial_fn=trivial_rows, time_col="t_min", progress=print, partition_fn=None):
    """Sinh từng dòng kết quả của một chiến dịch theo thứ tự marks; bỏ qua khóa đã có trong done.

    marks: episode, q, tau, n_io, n_ctrl, eligible và tùy chọn pre_t0 (mốc trước t0 của DS2: chạy B0/B1 để kiểm S3, TRIV ghi NaN).
    Mốc không đạt (không phải pre_t0): mỗi method một dòng, seed NaN, chỉ số NaN, không chạy mô hình.
    Suy luận chỉ dùng D(tau); nhãn (is_control) chỉ dùng cho tập đánh giá và chấm điểm.
    partition_fn(base, records) (tùy chọn, mặc định None): gọi một lần mỗi mốc đạt với danh sách dict(method, seed, accountid, cluster, in_eval) của B0/B1 trên toàn bộ nút được phân hoạch (B1: mọi tài khoản của D(tau); B0: tập đánh giá), để lưu phân hoạch; không ảnh hưởng chỉ số.
    runtime_s của B1 = ghép cặp tại tau chia đều cho 2 biến thể × len(seeds) + (Brent + dựng đồ thị của biến thể) / len(seeds) + Leiden của seed đó.
    """
    bounds, beta_seed = tuple(cfg["beta_bounds"]), cfg["community_seeds"][0]
    b0_methods = [f"B0_K{k}" for k in cfg["b0_clusters"]]
    model_methods = [*b0_methods, *(name for name, _ in B1_VARIANTS)]
    started = perf_counter()
    for mark in marks.to_dict("records"):
        base = dict(campaign=campaign, episode=mark["episode"], q=mark["q"], tau=mark["tau"], n_io=mark["n_io"], n_ctrl=mark["n_ctrl"], eligible=bool(mark["eligible"]))
        run_model = base["eligible"] or bool(mark.get("pre_t0", False))
        wanted = [(method, seed) for method in model_methods for seed in seeds] if run_model else [(method, float("nan")) for method in model_methods]
        wanted += [(method, float("nan")) for method in TRIV_METHODS]
        todo = [(method, seed) for method, seed in wanted if result_key({**base, "method": method, "seed": seed}) not in done]
        progress(f"[{campaign}] episode={base['episode']} q={base['q']} tau={format_tau(base['tau'])} eligible={base['eligible']} cần {len(todo)}/{len(wanted)} dòng | đã chạy {perf_counter() - started:.1f}s")
        if not todo:
            continue
        if not run_model:
            yield from (_row(base, method, seed) for method, seed in todo)
            continue
        history, current = observe(stream, mark["tau"], mark["episode"], time_col)
        nodes = sorted(history["accountid"].unique())
        population = set(current["accountid"])
        truth = set(current.loc[~current["is_control"], "accountid"])
        records = []
        keep = (lambda method, seed, partition: records.extend(dict(method=method, seed=seed, accountid=node, cluster=index, in_eval=node in population) for index, community in enumerate(partition) for node in sorted(community))) if partition_fn is not None else (lambda *_: None)
        for method, seed in todo:
            if method in b0_methods:
                clock = perf_counter()
                partition = random_partition(population, int(method.rsplit("K", 1)[1]), seed)
                row = _row(base, method, seed, partition_metrics(partition, population, truth), perf_counter() - clock)
                keep(method, seed, partition)
                yield row
        b1_todo = [(method, seed) for method, seed in todo if method.startswith("B1_")]
        if b1_todo:
            clock = perf_counter()
            pairs = checkpoint_pairs(events_fn(history), layers, mark["tau"])
            pair_s = perf_counter() - clock
            for method, keep_zero in B1_VARIANTS:
                variant_seeds = [seed for name, seed in b1_todo if name == method]
                if not variant_seeds:
                    continue
                clock = perf_counter()
                graphs, info = build_b1_graphs(pairs, nodes, keep_zero, mark["tau"], bounds, beta_seed, cfg["edge_tolerance"], cfg["brent_xatol"])
                build_s = perf_counter() - clock
                betas = {f"beta_{layer}": info[layer]["beta"] for layer in layers}
                weights = [info[layer]["edges_author"] for layer in layers]
                for seed in variant_seeds:
                    clock = perf_counter()
                    partition = leiden_multiplex(graphs, nodes, seed, weights)
                    row = _row(base, method, seed, partition_metrics(partition, population, truth), pair_s / (len(B1_VARIANTS) * len(seeds)) + build_s / len(seeds) + perf_counter() - clock, betas)
                    keep(method, seed, partition)
                    yield row
        if partition_fn is not None and records:
            partition_fn(base, records)
        triv_todo = {method for method, _ in todo if method.startswith("TRIV_")}
        if triv_todo and base["eligible"]:
            clock = perf_counter()
            rows = trivial_fn(current, ~current.groupby("accountid")["is_control"].first(), mark["tau"])
            share = (perf_counter() - clock) / len(rows)
            yield from (_row(base, row["method"], float("nan"), row, share) for row in rows if row["method"] in triv_todo)
        elif triv_todo:
            yield from (_row(base, method, float("nan")) for method in TRIV_METHODS if method in triv_todo)


def count_true_detections(partition, population, truth, min_coordinated, precision_threshold):
    """S3: số cụm (giới hạn vào tập đánh giá) có >= min_coordinated tài khoản phối hợp và tỷ lệ phối hợp >= precision_threshold."""
    restricted = [community & population for community in partition if community & population]
    return sum(1 for community in restricted if len(community & truth) >= min_coordinated and len(community & truth) / len(community) >= precision_threshold)


# --- Phần 9: DS2 (mô phỏng Iannucci, 3 lớp sim1–sim3) ---
def ds2_job(datasets, truth, sims, name, cfg, min_accounts):
    """Dựng một chiến dịch DS2 từ generate_data của T6b (thời gian phút làm tròn giây): luồng 1 đợt, mốc q và mốc trước t0.

    Lớp = sim{k}; nội dung = action_type:content_id trong lớp đó. Nhãn ở cấp tài khoản theo truth (6 tài khoản phối hợp);
    is_inauthentic của sự kiện chỉ dùng để đặt mốc q. Trả (name, stream, marks, events_fn, layers).
    """
    parts = [datasets[sim].assign(layer=f"sim{sim}") for sim in sims]
    events = pd.concat(parts, ignore_index=True).sort_values("timestamp", kind="stable", ignore_index=True)
    stream = pd.DataFrame({"accountid": events["user_id"].astype(str), "t_min": events["timestamp"].astype(float), "episode": 1, "is_control": ~events["user_id"].isin(truth), "day": np.floor(events["timestamp"] / 1440.0).astype(int), "layer": events["layer"], "action_type": events["action_type"], "content_id": events["content_id"].astype(str), "is_inauthentic": events["is_inauthentic"].astype(bool)})
    io_times = stream.loc[stream["is_inauthentic"], "t_min"].sort_values(kind="stable").to_numpy()
    t0 = float(cfg["ds2"]["t0_minutes"])
    schedule = [(float("nan"), t0 - offset, True) for offset in cfg["ds2"]["pre_t0_offsets_minutes"]]
    schedule += [(q, float(io_times[(round(q * 100) * len(io_times) + 99) // 100 - 1]), False) for q in cfg["q_values"]]
    rows = []
    for q, tau, pre_t0 in sorted(schedule, key=lambda item: item[1]):
        seen = stream.loc[stream["t_min"].le(tau)]
        n_io = seen.loc[~seen["is_control"], "accountid"].nunique()
        n_ctrl = seen.loc[seen["is_control"], "accountid"].nunique()
        rows.append(dict(episode=1, q=q, tau=tau, n_io=n_io, n_ctrl=n_ctrl, eligible=n_io >= min_accounts and n_ctrl >= min_accounts, pre_t0=pre_t0))
    marks = pd.DataFrame(rows, columns=["episode", "q", "tau", "n_io", "n_ctrl", "eligible", "pre_t0"])

    def events_fn(history):
        return pd.DataFrame({"user_id": history["accountid"].to_numpy(), "timestamp": history["t_min"].to_numpy(dtype=float), "action_type": history["layer"].to_numpy(), "content_id": (history["action_type"] + ":" + history["content_id"]).to_numpy()})

    return name, stream, marks, events_fn, [f"sim{sim}" for sim in sims]


def ds2_scores(current):
    """3 điểm số tầm thường cho DS2: (i) hạng thời điểm xuất hiện, (ii) số hành động/ngày hoạt động, (iii) tỷ lệ Δt = 0 trên mọi sự kiện các lớp (khóa nội dung = lớp, action_type, content_id; thời gian theo giây).

    (i) và (ii) bỏ sự kiện trùng hoàn toàn giữa các lớp (cùng tài khoản, thời điểm, action_type, content_id), vì phần authentic của bộ sinh giống hệt nhau ở 3 lớp; (iii) và đồ thị B1 giữ nguyên.
    """
    users = current.drop_duplicates(["accountid", "t_min", "action_type", "content_id"]).groupby("accountid", sort=True)
    scores = users.agg(first_seen=("t_min", "min"), actions=("t_min", "size"), days=("day", "nunique"))
    events = pd.DataFrame({"user_id": current["accountid"].to_numpy(), "timestamp": np.rint(current["t_min"].to_numpy(dtype=float) * 60).astype(np.int64), "action_type": current["layer"].to_numpy(), "content_id": (current["action_type"] + ":" + current["content_id"]).to_numpy()})
    zero_ratio = zero_delta_share(events)
    scores["first_seen"] = scores["first_seen"].rank(method="dense")
    scores["actions_per_active_day"] = scores["actions"] / scores["days"]
    scores["zero_delta_repost"] = zero_ratio.reindex(scores.index).fillna(0.0)
    return scores[["first_seen", "actions_per_active_day", "zero_delta_repost"]]


def ds2_trivial_rows(current, labels, tau):
    """F1* tầm thường cho DS2, cùng cách chấm với trivial_rows; tên cột (iii) giữ zero_delta_repost để khớp tên method TRIV."""
    assert_not_future(current["t_min"], tau, "ds2_trivial_rows")
    return _rows_from_scores(ds2_scores(current), labels, load_config()["triv_tiebreak_seed"])


def evaluate_campaign(stream, checkpoints, events_fn, campaign, cfg):
    """Bản gom thành DataFrame của iter_campaign_rows cho DS1 (lớp và seed lấy từ cfg)."""
    rows = iter_campaign_rows(stream, checkpoints, events_fn, campaign, cfg, cfg["ds1_layers"], cfg["community_seeds"], progress=lambda _: None)
    return pd.DataFrame(list(rows), columns=RESULT_COLUMNS)
