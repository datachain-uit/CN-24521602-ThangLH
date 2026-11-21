from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from src.eval.metrics import best_community_f1, weighted_precision


ARRAY_LAYERS = {"hashtags": "hashtag", "account_mentions": "mention", "urls": "url"}
LAYERS = ("retweet", *ARRAY_LAYERS.values())
MODEL_COLUMNS = ["accountid", "t_min", "day", "is_repost", "reposted_postid", *ARRAY_LAYERS]
INPUT_COLUMNS = ["accountid", "post_time", "is_control", "is_repost", "reposted_postid",
                 "in_reply_to_postid", *ARRAY_LAYERS]
CHECKPOINT_KEYS = ["campaign", "episode", "checkpoint_type", "checkpoint_value", "tau"]
CHECKPOINT_COLUMNS = [*CHECKPOINT_KEYS[1:], "q", "n_io", "n_ctrl", "io_seen", "ctrl_seen",
                      "io_actions_episode", "eligible"]
BASELINES = ("first_seen", "actions_per_active_day", "zero_delta_repost")
P6_VARIANTS = ("main", "without_hashtag", "shuffle_account_day", "cap_account_day")


@dataclass(frozen=True)
class EpisodeRules:
    min_io_day: int = 2
    control_minima: tuple[int, ...] = (1, 5, 10)
    max_missing_days: int = 3
    min_accounts: int = 5
    min_main_episodes: int = 3
    min_main_io_share: float = 0.20
    percentages: tuple[int, ...] = (10, 25, 50, 100)
    days: tuple[int, ...] = (1, 3, 7)


def harmonize_time(raw):
    frame = raw.copy()
    frame["post_time"] = pd.to_datetime(frame["post_time"], utc=True, format="mixed")
    if frame["post_time"].isna().any():
        raise ValueError("Có post_time bị thiếu.")
    for column in ("is_control", "is_repost"):
        if not frame[column].isin([False, True]).all():
            raise ValueError(f"Giá trị thiếu hoặc không hợp lệ ở {column}.")
        frame[column] = frame[column].astype(bool)
    if pd.api.types.is_float_dtype(frame["accountid"]):
        raise ValueError("accountid kiểu float có thể đã mất độ chính xác.")
    frame["accountid"] = frame["accountid"].astype("string")
    if frame["accountid"].fillna("").str.strip().eq("").any():
        raise ValueError("Có accountid bị thiếu hoặc rỗng.")
    if frame.groupby("accountid")["is_control"].nunique().gt(1).any():
        raise ValueError("Có tài khoản mang cả hai nhãn trong chiến dịch.")
    frame["t_min"] = frame["post_time"].dt.floor("min")
    frame["day"] = frame["t_min"].dt.floor("D")
    return frame


def label_counts(frame, by):
    grouped = frame.groupby([by, "is_control"])["accountid"]
    result = pd.DataFrame(index=frame[by].drop_duplicates().sort_values()).rename_axis(by)
    for name, values in (("accounts", grouped.nunique()), ("actions", grouped.size())):
        wide = values.unstack(fill_value=0).reindex(columns=[False, True], fill_value=0)
        wide.columns = [f"io_{name}", f"ctrl_{name}"]
        result = result.join(wide)
    return result


def daily_coverage(frame, rules):
    daily = label_counts(frame, "day")
    for c in rules.control_minima:
        daily[f"covered_c{c}"] = daily["io_accounts"].ge(rules.min_io_day) & daily["ctrl_accounts"].ge(c)
    return daily


def make_episodes(frame, daily, rules, control_min):
    covered = daily.index[daily[f"covered_c{control_min}"]].sort_values()
    episode_by_day = covered.to_series().diff().dt.days.gt(rules.max_missing_days + 1).cumsum() + 1
    stream = frame.loc[frame["day"].isin(covered)].copy()
    stream["episode"] = stream["day"].map(episode_by_day).astype("int64")
    stream = stream.sort_values("t_min", kind="stable")

    episodes = label_counts(stream, "episode").join(
        stream.groupby("episode")["day"].agg(start="min", end="max", covered_days="nunique")
    )
    episodes["span_days"] = (episodes["end"] - episodes["start"]).dt.days + 1
    episodes["qualified"] = (
        episodes["io_accounts"].ge(rules.min_accounts) & episodes["ctrl_accounts"].ge(rules.min_accounts)
    )
    return stream, episodes


def select_campaign(daily, episodes, rules, control_min):
    total = int(daily["io_actions"].sum())
    qualified = episodes.loc[episodes["qualified"]]
    qualified_io = int(qualified["io_actions"].sum())
    io_only = int(daily.loc[daily["ctrl_accounts"].eq(0), "io_actions"].sum())
    share = qualified_io / total if total else np.nan
    enough_episodes = len(qualified) >= rules.min_main_episodes
    enough_io = share >= rules.min_main_io_share
    return dict(
        control_min=control_min, observed_days=len(daily), covered_days=int(daily[f"covered_c{control_min}"].sum()),
        n_episodes=len(episodes), n_qualified=len(qualified), n_unqualified=len(episodes) - len(qualified),
        io_actions_total=total, io_actions_in_stream=int(episodes["io_actions"].sum()),
        io_actions_qualified=qualified_io, io_share_qualified=share, io_actions_without_control=io_only,
        io_share_without_control=io_only / total if total else np.nan,
        meets_episode_min=enough_episodes, meets_io_share_min=enough_io, in_main_set=enough_episodes and enough_io,
    )


def make_checkpoints(stream, episodes, rules):
    rows = []
    qualified = episodes.loc[episodes["qualified"]]
    for episode, posts in stream.loc[stream["episode"].isin(qualified.index)].groupby("episode"):
        ep = qualified.loc[episode]
        times = posts.loc[~posts["is_control"], "t_min"].reset_index(drop=True)
        schedule = [
            ("io_pct", p, times.iloc[(p * len(times) + 99) // 100 - 1]) for p in rules.percentages
        ]
        schedule.extend(
            ("day", d, ep["start"] + pd.Timedelta(days=d, minutes=-1))
            for d in rules.days if d <= ep["span_days"]
        )
        for kind, value, tau in schedule:
            seen = posts.loc[posts["t_min"].le(tau)]
            control = seen["is_control"]
            n_io = seen.loc[~control, "accountid"].nunique()
            n_ctrl = seen.loc[control, "accountid"].nunique()
            rows.append(dict(
                episode=episode, checkpoint_type=kind, checkpoint_value=value,
                q=value / 100 if kind == "io_pct" else np.nan, tau=tau, n_io=n_io, n_ctrl=n_ctrl,
                io_seen=int((~control).sum()), ctrl_seen=int(control.sum()), io_actions_episode=len(times),
                eligible=n_io >= rules.min_accounts and n_ctrl >= rules.min_accounts,
            ))
    return pd.DataFrame(rows, columns=CHECKPOINT_COLUMNS).astype({"eligible": bool})


def observed_at(stream, checkpoint):
    """Luồng đã sắp xếp. Tách prefix mang lịch sử và tập đánh giá trong đợt hiện tại."""
    stop = stream["t_min"].searchsorted(checkpoint.tau, side="right")
    history = stream.iloc[:stop]
    current = history.loc[history["episode"].eq(checkpoint.episode)]
    return history, current


def extract_items(value):
    if not isinstance(value, (list, tuple, np.ndarray)):
        return []
    return [str(item).strip() for item in value if pd.notna(item) and str(item).strip()]


def build_events(posts, layers=LAYERS):
    base = posts.rename(columns={"accountid": "user_id"})
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    base = base.assign(timestamp=(base["t_min"] - epoch) // pd.Timedelta(minutes=1))
    events = []
    for column, layer in {"reposted_postid": "retweet", **ARRAY_LAYERS}.items():
        if layer not in layers:
            continue
        part = base[["user_id", "timestamp", column]].rename(columns={column: "content_id"})
        if layer == "retweet":
            part = part.loc[base["is_repost"]]
            part["content_id"] = part["content_id"].astype("string").str.strip()
            part = part.loc[part["content_id"].notna() & part["content_id"].ne("")]
        else:
            part["content_id"] = part["content_id"].map(extract_items)
            part = part.explode("content_id", ignore_index=True).dropna(subset=["content_id"])
        events.append(part.assign(action_type=layer))
    return pd.concat(events, ignore_index=True).sort_values("timestamp", kind="stable", ignore_index=True)


def _count_smaller_ids(reference, query):
    """Với mỗi dòng query, đếm dòng reference cùng (c, t) có user_id nhỏ hơn hẳn."""
    both = pd.concat([reference.assign(is_ref=1), query.assign(is_ref=0)], ignore_index=True)
    # Cùng ID thì query đứng trước reference, nên chỉ đếm ID nhỏ hơn hẳn.
    both = both.sort_values(["c", "t", "user_id", "is_ref"], kind="stable")
    before = both.groupby(["c", "t"], sort=False)["is_ref"].cumsum() - both["is_ref"]
    picked = both["is_ref"].eq(0)
    return pd.Series(before[picked].to_numpy(), index=both.loc[picked, "row"].to_numpy()).sort_index().to_numpy()


def zero_delta_share(events):
    """Tỷ lệ cặp có Δt = 0 theo tài khoản nguồn, bằng đúng find_next_action_pairs(keep_zero=True).

    Không liệt kê từng cặp. Với sự kiện nguồn (u, s) trên nội dung c, tài khoản v ≠ u ghép được khi
    v có hành động > s (v > u) hoặc ≥ s (v < u). Cặp có Δt = 0 khi v < u và v có hành động đúng ở s.
    """
    if events.empty:
        return pd.Series(dtype=float)
    frame = pd.DataFrame({
        "c": pd.factorize(pd.MultiIndex.from_frame(events[["action_type", "content_id"]]))[0],
        "user_id": events["user_id"].to_numpy(),
        "t": events["timestamp"].to_numpy(dtype=np.int64),
    })
    frame["row"] = np.arange(len(frame))
    last = frame.groupby(["c", "user_id"], as_index=False, sort=False)["t"].max()

    # Số tài khoản khác u có hành động cuối sau hẳn s.
    span = int(frame["t"].max() - frame["t"].min()) + 2
    origin = int(frame["t"].min())
    last_keys = np.sort(last["c"].to_numpy(np.int64) * span + (last["t"].to_numpy(np.int64) - origin))
    keys = frame["c"].to_numpy(np.int64) * span + (frame["t"].to_numpy(np.int64) - origin)
    content_end = np.searchsorted(last_keys, (frame["c"].to_numpy(np.int64) + 1) * span, side="left")
    later = content_end - np.searchsorted(last_keys, keys, side="right")
    own_last = frame.merge(last, on=["c", "user_id"], suffixes=("", "_last"), how="left")["t_last"].to_numpy()
    later -= own_last > frame["t"].to_numpy()

    query = frame[["c", "t", "user_id", "row"]]
    tie_last = _count_smaller_ids(last.assign(row=-1), query)
    zero = _count_smaller_ids(frame[["c", "t", "user_id"]].drop_duplicates().assign(row=-1), query)

    totals = pd.DataFrame({"pairs": later + tie_last, "zero": zero}).groupby(frame["user_id"].to_numpy()).sum()
    totals = totals.loc[totals["pairs"].gt(0)]
    return totals["zero"] / totals["pairs"]


def baseline_scores(current):
    users = current.groupby("accountid", sort=True)
    scores = users.agg(first_seen=("t_min", "min"), actions=("t_min", "size"), days=("day", "nunique"))
    zero_ratio = zero_delta_share(build_events(current, layers=("retweet",)))
    scores["first_seen"] = scores["first_seen"].rank(method="dense")
    scores["actions_per_active_day"] = scores["actions"] / scores["days"]
    scores["zero_delta_repost"] = zero_ratio.reindex(scores.index).fillna(0.0)
    return scores[list(BASELINES)]


def oracle_top_k(score, labels):
    """Nhánh oracle ngoại tuyến. Phá hòa bằng thứ tự ID cố định, độc lập nhãn."""
    y = labels.reindex(score.index).astype(int)
    rows = []
    for direction, sign in (("high", 1), ("low", -1)):
        signed = (sign * score).sort_index()
        order = signed.sort_values(ascending=False, kind="stable").index
        tp = y.loc[order].cumsum().to_numpy()
        k = np.arange(1, len(tp) + 1)
        f1 = 2 * tp / (k + y.sum())
        best = int(np.argmax(f1))
        rows.append(dict(direction=direction, auc=roc_auc_score(y, signed.reindex(y.index)),
                         F1_star=float(f1[best]), oracle_k=int(k[best])))
    return rows


def evaluate_baselines(stream, checkpoints):
    rows = []
    eligible = checkpoints.loc[checkpoints["eligible"]]
    for _, marks in eligible.groupby(["episode", "tau"], sort=False):
        _, current = observed_at(stream, next(marks.itertuples(index=False)))
        scores = baseline_scores(current)
        labels = ~current.groupby("accountid")["is_control"].first()
        for method, score in scores.items():
            directions = oracle_top_k(score, labels)
            best = max(row["F1_star"] for row in directions)
            for mark in marks.to_dict("records"):
                rows.extend({**mark, "method": method, **row, "F1_star_best_direction": best} for row in directions)
    return pd.DataFrame(rows, columns=[*checkpoints.columns, "method", "direction", "auc", "F1_star",
                                      "oracle_k", "F1_star_best_direction"])


def per_episode_results(results, metric):
    groups = [*CHECKPOINT_KEYS, "method"]
    eligible = results.loc[results["eligible"]].dropna(subset=[metric])
    return eligible.groupby(groups, as_index=False).agg(
        estimate=(metric, "mean"), io_weight=("io_actions_episode", "first"),
    )


def aggregate_episodes(results, metric, seed, n_bootstrap):
    """Mỗi mức mốc xét riêng. Trung bình seed trước khi bootstrap cặp (đợt, trọng số)."""
    groups = ["campaign", "checkpoint_type", "checkpoint_value", "method"]
    episode = per_episode_results(results, metric).sort_values("episode")
    rng = np.random.default_rng(seed)
    rows = []
    for key, part in episode.groupby(groups):
        values = part["estimate"].to_numpy()
        weights = part["io_weight"].to_numpy()
        n = len(part)
        for weighting, w in (("equal", np.ones(n)), ("io_actions", weights)):
            lo, hi = np.nan, np.nan
            if n >= 2:
                indices = rng.integers(0, n, size=(n_bootstrap, n))
                means = np.average(values[indices], axis=1, weights=w[indices])
                lo, hi = np.quantile(means, [0.025, 0.975])
            rows.append(dict(zip(groups, key), metric=metric, unit="episode_within_campaign",
                             weighting=weighting, n_episodes=n, estimate=np.average(values, weights=w),
                             ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(rows, columns=[*groups, "metric", "unit", "weighting", "n_episodes", "estimate",
                                      "ci95_low", "ci95_high"])


def compare_campaigns(results, metric, method, reference):
    """Kiểm định dấu hai phía trên chênh lệch trung bình cùng đợt, mỗi chiến dịch một dấu."""
    episodes = per_episode_results(results, metric)
    candidates = episodes.loc[episodes["method"].eq(method), [*CHECKPOINT_KEYS, "estimate"]]
    references = episodes.loc[episodes["method"].eq(reference), [*CHECKPOINT_KEYS, "estimate"]]
    paired = candidates.merge(references, on=CHECKPOINT_KEYS, suffixes=("", "_reference"), validate="one_to_one")
    paired["difference"] = paired["estimate"] - paired["estimate_reference"]
    by_mark = ["checkpoint_type", "checkpoint_value"]
    effects = paired.groupby([*by_mark, "campaign"], as_index=False).agg(
        difference=("difference", "mean"), n_paired_episodes=("difference", "size"),
    )
    rows = []
    for key, part in effects.groupby(by_mark):
        signs = np.sign(part["difference"].to_numpy())
        signs[np.isclose(part["difference"], 0, atol=1e-12)] = 0
        wins = int((signs > 0).sum())
        losses = int((signs < 0).sum())
        rows.append(dict(zip(by_mark, key), metric=metric, method=method, reference=reference, unit="campaign",
                         n_campaigns=len(signs), wins=wins, losses=losses, ties=len(signs) - wins - losses,
                         p_value=binomtest(wins, wins + losses).pvalue if wins + losses else np.nan))
    return effects, pd.DataFrame(rows, columns=[*by_mark, "metric", "method", "reference", "unit", "n_campaigns",
                                               "wins", "losses", "ties", "p_value"])


def prepare_variant(history, current, variant, cap_k, seed):
    posts = (current if variant == "reset_state" else history)[MODEL_COLUMNS].copy()
    if variant == "shuffle_account_day":
        rng = np.random.default_rng(seed)
        posts["t_min"] = posts.groupby(["accountid", "day"], sort=True)["t_min"].transform(
            lambda times: rng.permutation(times.to_numpy())
        )
    elif variant == "cap_account_day":
        posts = posts.sort_values("t_min", kind="stable").groupby(["accountid", "day"], sort=False).head(cap_k)
    layers = tuple(layer for layer in LAYERS if variant != "without_hashtag" or layer != "hashtag")
    return posts, layers


def control_cluster_summary(communities, truth, events, limit=10):
    candidates = [members for members in communities if len(members) >= 2 and members - truth]
    candidates.sort(key=lambda members: (-len(members - truth), -len(members - truth) / len(members),
                                         -len(members), min(members)))
    rows = []
    for rank, members in enumerate(candidates[:limit], 1):
        control = members - truth
        actions = events.loc[events["user_id"].isin(control)]
        shared = actions.groupby(["action_type", "content_id"])["user_id"].nunique().ge(2)
        synced = actions.groupby(["action_type", "content_id", "timestamp"], as_index=False)["user_id"].nunique()
        synced = synced.loc[synced["user_id"].ge(2)]
        sync_days = (synced["timestamp"] // 1440).nunique()
        rows.append(dict(rank=rank, accounts=len(members), io_accounts=len(members & truth),
                         control_accounts=len(control), control_share=len(control) / len(members),
                         synchronized_events=len(synced), synchronized_days=sync_days,
                         repeated_sync=sync_days >= 2,
                         **{f"shared_{layer}": int(shared[shared.index.get_level_values(0) == layer].sum())
                            for layer in LAYERS}))
    return rows


def evaluate_ablations(stream, checkpoints, infer_partition, seeds, cap_k, reset_state=False):
    """infer_partition(events, nodes, layers, seed) trả phân hoạch đầy đủ, không nhận nhãn."""
    if cap_k < 1:
        raise ValueError("K phải là số nguyên dương.")
    eligible = checkpoints.loc[checkpoints["eligible"]]
    variants = (*P6_VARIANTS, "reset_state") if reset_state else P6_VARIANTS
    rows, qualitative = [], []
    latest = None if eligible.empty else eligible.sort_values("tau", kind="stable").iloc[-1]
    for _, marks in eligible.groupby(["episode", "tau"], sort=False):
        checkpoint = next(marks.itertuples(index=False))
        history, current = observed_at(stream, checkpoint)
        truth = set(current.loc[~current["is_control"], "accountid"])
        population = set(current["accountid"])
        for method in variants:
            for seed in seeds:
                started = perf_counter()
                posts, layers = prepare_variant(history, current, method, cap_k, seed)
                assert posts["t_min"].le(checkpoint.tau).all()
                events = build_events(posts, layers)
                # Giữ cả các tài khoản có mặt ở prefix nhưng không có sự kiện thuộc các lớp.
                nodes = sorted(current["accountid"].unique() if method == "reset_state" else history["accountid"].unique())
                partition = [set(cluster) for cluster in infer_partition(events, nodes, layers, seed)]
                if set().union(*partition) != set(nodes) or sum(map(len, partition)) != len(nodes):
                    raise ValueError("Phân hoạch phải chứa đúng một lần mỗi tài khoản prefix, kể cả nút cô lập.")
                restricted = [cluster & population for cluster in partition if cluster & population]
                f1, _ = best_community_f1(restricted, truth)
                wp = weighted_precision(restricted, truth)
                runtime = perf_counter() - started
                rows.extend({**mark, "method": method, "seed": seed, "F1_star": f1, "WP": wp,
                             "runtime_seconds": runtime} for mark in marks.to_dict("records"))
                if (method == "main" and seed == seeds[0] and checkpoint.episode == latest["episode"]
                        and checkpoint.tau == latest["tau"]):
                    identity = latest[[key for key in CHECKPOINT_KEYS if key in latest.index]].to_dict()
                    qualitative.extend({**identity, "seed": seed, **row}
                                       for row in control_cluster_summary(restricted, truth, events))
    scores = pd.DataFrame(rows, columns=[*checkpoints.columns, "method", "seed", "F1_star", "WP", "runtime_seconds"])
    control = pd.DataFrame(qualitative, columns=[*CHECKPOINT_KEYS, "seed", "rank", "accounts", "io_accounts",
                                               "control_accounts", "control_share", "synchronized_events",
                                               "synchronized_days", "repeated_sync", *[f"shared_{x}" for x in LAYERS]])
    return scores, control


def ablation_deltas(scores):
    keys = [*CHECKPOINT_KEYS, "seed"]
    metrics = ["F1_star", "WP"]
    main = scores.loc[scores["method"].eq("main"), [*keys, *metrics]]
    deltas = scores.loc[scores["method"].ne("main")].merge(main, on=keys, suffixes=("", "_main"), validate="many_to_one")
    for metric in metrics:
        deltas[f"delta_{metric}"] = deltas[metric] - deltas[f"{metric}_main"]
    return deltas


def ablation_summary(deltas):
    """Trung bình đều theo đợt trong từng seed, rồi trung bình và độ lệch chuẩn giữa các seed."""
    mark = ["campaign", "checkpoint_type", "checkpoint_value", "method"]
    values = ["F1_star_main", "F1_star", "delta_F1_star", "WP_main", "WP", "delta_WP"]
    per_seed = deltas.groupby([*mark, "seed"], as_index=False).agg(episodes=("episode", "nunique"), **{v: (v, "mean") for v in values})
    summary = per_seed.groupby(mark, as_index=False).agg(seeds=("seed", "nunique"), episodes=("episodes", "max"), **{f"{v}_mean": (v, "mean") for v in values}, sd_delta_F1_star=("delta_F1_star", "std"), sd_delta_WP=("delta_WP", "std"))
    return summary
