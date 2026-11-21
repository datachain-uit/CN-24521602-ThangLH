from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

PAIR_COLUMNS = ["u", "v", "dd", "w0"]
EDGE_COLUMNS = ["source_user", "target_user", "action_type", "weight"]


def matched_delta_t(t_u, t_v, keep_zero=True):
    """
    u có ID nhỏ hơn v.
    u -> v: tìm hành động sau hẳn (>)
    v -> u: tìm hành động sau hoặc bằng (>=)
    Mỗi hành động nguồn được ghép với tối đa một hành động đích.
    Đơn vị thời gian: phút. keep_zero=False loại delta bằng 0 sau khi ghép.
    """
    t_u = np.sort(np.asarray(t_u, dtype=float))
    t_v = np.sort(np.asarray(t_v, dtype=float))

    if len(t_u) == 0 or len(t_v) == 0:
        return np.array([], dtype=float)

    out = []

    for src, tgt, side in ((t_u, t_v, "right"), (t_v, t_u, "left")):
        pos = np.searchsorted(tgt, src, side=side)
        ok = pos < len(tgt)
        out.append(tgt[pos[ok]] - src[ok])

    d = np.concatenate(out)
    return d if keep_zero else d[d != 0]

def count_content_users(events: pd.DataFrame) -> pd.DataFrame:
    return events.groupby(["action_type", "content_id"])["user_id"].nunique().rename("n_users").reset_index()


def find_next_action_pairs(events: pd.DataFrame, keep_zero: bool = True) -> pd.DataFrame:
    """Ghép sự kiện theo lớp và nội dung, dùng thứ tự ID gốc của tài khoản.

    ID chuỗi được so sánh theo thứ tự từ điển, ID số nguyên theo giá trị số.
    """
    columns = [
        "action_type", "content_id", "source_user", "target_user",
        "source_time", "target_time", "delta_t",
    ]
    records = {column: [] for column in columns}

    for (action_type, content_id), group in events.groupby(["action_type", "content_id"]):
        group = group.sort_values("timestamp")

        by_user = {
            user_id: user_events["timestamp"].to_numpy()
            for user_id, user_events in group.groupby("user_id")
        }

        for source_user, source_times in by_user.items():
            for target_user, target_times in by_user.items():
                if source_user == target_user:
                    continue

                side = "right" if source_user < target_user else "left"
                positions = target_times.searchsorted(source_times, side=side)
                has_match = positions < len(target_times)
                matched_source = source_times[has_match]
                matched_target = target_times[positions[has_match]]
                deltas = matched_target - matched_source

                if not keep_zero:
                    nonzero = deltas != 0
                    matched_source = matched_source[nonzero]
                    matched_target = matched_target[nonzero]
                    deltas = deltas[nonzero]

                n_matches = len(deltas)
                records["action_type"].extend([action_type] * n_matches)
                records["content_id"].extend([content_id] * n_matches)
                records["source_user"].extend([source_user] * n_matches)
                records["target_user"].extend([target_user] * n_matches)
                records["source_time"].extend(matched_source)
                records["target_time"].extend(matched_target)
                records["delta_t"].extend(deltas)

    if not records["delta_t"]:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records)


def prepare_action_pairs(events: pd.DataFrame, action_type: str | None, keep_zero: bool = True, with_time: bool = False) -> pd.DataFrame:
    """with_time=True thêm cột t = thời điểm hành động đích của cặp (muộn hơn, cùng đơn vị với timestamp) để kiểm U2."""
    layer = events if action_type is None else events.loc[events["action_type"] == action_type]
    pairs = find_next_action_pairs(layer, keep_zero=keep_zero)

    if pairs.empty:
        return pd.DataFrame(columns=[*PAIR_COLUMNS, "t"] if with_time else PAIR_COLUMNS)

    counts = count_content_users(layer)

    pairs = pairs.merge(counts, on=["action_type", "content_id"], how="left", validate="many_to_one")

    if pairs["n_users"].isna().any() or (pairs["n_users"] < 2).any():
        raise ValueError("Each matched content must have at least two users.")

    source = pairs["source_user"]
    target = pairs["target_user"]
    canonical = source < target
    result = pd.DataFrame({
        "u": source.where(canonical, target),
        "v": target.where(canonical, source),
        "dd": pairs["delta_t"].to_numpy(dtype=np.float64) / 1440.0,
        "w0": 1.0 / (pairs["n_users"].to_numpy(dtype=np.float64) - 1.0),
    })
    if with_time:
        result["t"] = pairs["target_time"].to_numpy(dtype=np.float64)
    return result


def build_layer_edges_from_pairs(pairs: pd.DataFrame, action_type: str | None, beta: float) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    weighted = pairs[["u", "v"]].copy()

    weighted["weight"] = (
        pairs["w0"].to_numpy(dtype=np.float64)
        * np.exp(-float(beta) * pairs["dd"].to_numpy(dtype=np.float64))
    )

    edges = (
        weighted.groupby(["u", "v"], sort=True, as_index=False)["weight"]
        .sum()
        .rename(columns={"u": "source_user", "v": "target_user"})
    )

    edges["action_type"] = "all" if action_type is None else action_type

    return edges[EDGE_COLUMNS]


def build_layer_graph_from_pairs(pairs: pd.DataFrame, nodes, beta: float, tolerance: float = 1e-9) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    edges = build_layer_edges_from_pairs(pairs, action_type=None, beta=beta)
    for row in edges.itertuples(index=False):
        if row.weight > tolerance:
            graph.add_edge(row.source_user, row.target_user, weight=float(row.weight))
    return graph


def build_layer_edges(
    events: pd.DataFrame, action_type: str | None, beta: float, keep_zero: bool = True,
) -> pd.DataFrame:
    pairs = prepare_action_pairs(events, action_type, keep_zero=keep_zero)
    return build_layer_edges_from_pairs(pairs, action_type, beta)


def build_layer_graph(
    events: pd.DataFrame, action_type: str | None, beta: float,
    tolerance: float = 1e-9, keep_zero: bool = True,
) -> nx.Graph:
    pairs = prepare_action_pairs(events, action_type, keep_zero=keep_zero)

    return build_layer_graph_from_pairs(
        pairs=pairs, nodes=events["user_id"].unique(), beta=beta, tolerance=tolerance,
    )
