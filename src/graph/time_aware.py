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
    Đơn vị thời gian: phút.
    """
    t_u = np.sort(np.asarray(t_u, dtype=float))
    t_v = np.sort(np.asarray(t_v, dtype=float))

    if len(t_u) == 0 or len(t_v) == 0:
        return np.array([], dtype=float)

    out = []

    for src, tgt, side in (
        (t_u, t_v, "right"),
        (t_v, t_u, "left"),
    ):
        pos = np.searchsorted(tgt, src, side=side)
        ok = pos < len(tgt)
        out.append(tgt[pos[ok]] - src[ok])

    d = np.concatenate(out)
    return d if keep_zero else d[d != 0]

def count_content_users(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(["action_type", "content_id"])["user_id"]
        .nunique()
        .rename("n_users")
        .reset_index()
    )


def find_next_action_pairs(
    events: pd.DataFrame,
    keep_zero: bool = True,
) -> pd.DataFrame:
    records = []
    columns = [
        "action_type", "content_id",
        "source_user", "target_user",
        "source_time", "target_time",
        "delta_t",
    ]

    for (action_type, content_id), group in events.groupby(
        ["action_type", "content_id"]
    ):
        group = group.sort_values("timestamp")

        by_user = {
            user_id: user_events["timestamp"].to_numpy()
            for user_id, user_events in group.groupby("user_id")
        }

        for source_user, source_times in by_user.items():
            for target_user, target_times in by_user.items():
                if source_user == target_user:
                    continue

                # T10: strict/non-strict theo thứ tự ID.
                side = (
                    "right"
                    if source_user < target_user
                    else "left"
                )

                for source_time in source_times:
                    pos = target_times.searchsorted(
                        source_time,
                        side=side,
                    )

                    if pos >= len(target_times):
                        continue

                    target_time = target_times[pos]
                    delta_t = target_time - source_time

                    # B1-gốc: loại delta_t = 0.
                    # B1-chuẩn: giữ delta_t = 0.
                    if not keep_zero and delta_t == 0:
                        continue

                    records.append({
                        "action_type": action_type,
                        "content_id": content_id,
                        "source_user": source_user,
                        "target_user": target_user,
                        "source_time": source_time,
                        "target_time": target_time,
                        "delta_t": delta_t,
                    })

    return pd.DataFrame.from_records(
        records,
        columns=columns,
    )


def prepare_action_pairs(
    events: pd.DataFrame,
    action_type: str | None,
    keep_zero: bool = True,
) -> pd.DataFrame:
    layer = (
        events
        if action_type is None
        else events.loc[events["action_type"] == action_type]
    )

    pairs = find_next_action_pairs(
        layer,
        keep_zero=keep_zero,
    )

    if pairs.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    counts = count_content_users(layer)

    pairs = pairs.merge(
        counts,
        on=["action_type", "content_id"],
        how="left",
        validate="many_to_one",
    )

    if pairs["n_users"].isna().any() or (pairs["n_users"] < 2).any():
        raise ValueError(
            "Each matched content must have at least two users."
        )

    return pd.DataFrame({
        "u": pairs[["source_user", "target_user"]].min(axis=1),
        "v": pairs[["source_user", "target_user"]].max(axis=1),
        "dd": pairs["delta_t"].to_numpy(
            dtype=np.float64
        ) / 1440.0,
        "w0": 1.0 / (
            pairs["n_users"].to_numpy(
                dtype=np.float64
            ) - 1.0
        ),
    })


def build_layer_edges_from_pairs(
    pairs: pd.DataFrame,
    action_type: str | None,
    beta: float,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    weighted = pairs[["u", "v"]].copy()

    weighted["weight"] = (
        pairs["w0"].to_numpy(dtype=np.float64)
        * np.exp(
            -float(beta)
            * pairs["dd"].to_numpy(dtype=np.float64)
        )
    )

    edges = (
        weighted.groupby(
            ["u", "v"],
            sort=True,
            as_index=False,
        )["weight"]
        .sum()
        .rename(columns={
            "u": "source_user",
            "v": "target_user",
        })
    )

    edges["action_type"] = (
        "all" if action_type is None else action_type
    )

    return edges[EDGE_COLUMNS]


def build_layer_graph_from_pairs(
    pairs: pd.DataFrame,
    nodes,
    beta: float,
    tolerance: float = 1e-9,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    edges = build_layer_edges_from_pairs(pairs, action_type=None, beta=beta)
    for row in edges.itertuples(index=False):
        if row.weight > tolerance:
            graph.add_edge(
                row.source_user,
                row.target_user,
                weight=float(row.weight),
            )
    return graph


def build_layer_edges(
    events: pd.DataFrame,
    action_type: str | None,
    beta: float,
    keep_zero: bool = True,
) -> pd.DataFrame:
    pairs = prepare_action_pairs(
        events,
        action_type,
        keep_zero=keep_zero,
    )

    return build_layer_edges_from_pairs(
        pairs,
        action_type,
        beta,
    )


def build_layer_graph(
    events: pd.DataFrame,
    action_type: str | None,
    beta: float,
    tolerance: float = 1e-9,
    keep_zero: bool = True,
) -> nx.Graph:
    pairs = prepare_action_pairs(
        events,
        action_type,
        keep_zero=keep_zero,
    )

    return build_layer_graph_from_pairs(
        pairs=pairs,
        nodes=events["user_id"].unique(),
        beta=beta,
        tolerance=tolerance,
    )
