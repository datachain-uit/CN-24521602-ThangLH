"""DS2-v2 (T13, Review-2026-11-21 §6.7): ngụy trang, khối lượng khớp nhau, mô phỏng 3 theo bài báo, các chế độ clean / seckin / phân rã.

v1 (src/simulation/synthetic.py) không bị sửa: mô phỏng 1, 2 và các khung thời gian của mô phỏng 3 gọi lại đúng hàm v1 với cùng seed.
Thời gian của thế giới tính bằng giây nguyên từ START. Nhãn chỉ dùng để dựng chế độ giống Seckin (mô phỏng quy trình thu thập) và để đánh giá.
"""
import numpy as np
import pandas as pd

from src.simulation.synthetic import EVENT_COLUMNS, SimulationConfig, assign_coordinated_contents, assign_simulation2_users, generate_action_catalog, generate_authentic_events, generate_simulation1_events, generate_simulation3_timestamps

START = pd.Timestamp("2020-01-01", tz="UTC")
DAY_S = 86400
WORLD_COLUMNS = ["accountid", "t", "day", "action", "content", "label", "is_coord_action"]
SIM3_GROUPS = (("coord_00", "coord_01", "coord_02"), ("coord_03", "coord_04", "coord_05"))
# ASSUMPTION: số sự kiện của 4 khung mô phỏng 3 giữ như v1 (5, 6, 4, 3); Phụ lục B không cho các số này.
SIM3_WINDOW_SIZES = (5, 6, 4, 3)
TAG = 13

def _shape_code(shape):
    """Mã nguyên của Gamma shape để đưa vào khóa RNG (0,5 → 500)."""
    return int(round(float(shape) * 1000))


def assign_simulation3_users_v2(config: SimulationConfig) -> pd.DataFrame:
    """Mô phỏng 3 theo bài báo: hai nhóm ba tài khoản luân phiên hoạt động; nhóm không đến lượt im lặng trong khung đó."""
    if config.n_inauthentic != 6:
        raise ValueError("Mô phỏng 3 v2 cần đúng 6 tài khoản phối hợp.")
    rng = np.random.default_rng(config.seed + 10)
    # ASSUMPTION: 4 khung giữ đúng vị trí và độ dài của v1 (phút 1920, 4860, 7710, 8430; dài 5, 5, 3, 2 phút), lấy từ generate_simulation3_timestamps của v1.
    timestamps = generate_simulation3_timestamps(config)
    if len(timestamps) != sum(SIM3_WINDOW_SIZES):
        raise ValueError("Số sự kiện mô phỏng 3 không khớp số khung.")
    users, windows = [], []
    for window, size in enumerate(SIM3_WINDOW_SIZES):
        # ASSUMPTION: luân phiên A, B, A, B, bắt đầu bằng nhóm A (coord_00–02).
        group = list(SIM3_GROUPS[window % 2])
        # ASSUMPTION: mỗi thành viên của nhóm đang đến lượt có ít nhất 1 sự kiện trong khung; phần còn lại chọn đều trong nhóm, như cách gán của v1.
        chosen = group + rng.choice(group, size=size - len(group), replace=True).tolist()
        rng.shuffle(chosen)
        users.extend(chosen)
        windows.extend([window] * size)
    return pd.DataFrame({"user_id": users, "timestamp": timestamps, "is_inauthentic": True, "window": windows})


def coordinated_events(sim, config: SimulationConfig) -> pd.DataFrame:
    """Hành động phối hợp của mô phỏng sim (phút, như v1): 1 và 2 gọi đúng hàm v1; 3 dùng bản v2 luân phiên hai nhóm."""
    if sim == 1:
        return generate_simulation1_events(config)
    if sim == 2:
        return assign_coordinated_contents(events=assign_simulation2_users(config), config=config, seed_offset=8)
    if sim == 3:
        # ASSUMPTION: nội dung mô phỏng 3 gán như v1 (một cặp nội dung mỗi loại, xác suất [0,9; 0,1]), dùng chung cho hai nhóm.
        return assign_coordinated_contents(events=assign_simulation3_users_v2(config), config=config, seed_offset=11)
    raise ValueError(f"Không có mô phỏng {sim}.")


def daily_counts(n_accounts, days, shape, mean, rng):
    """Số hành động nền mỗi tài khoản–ngày, rút trước cho mọi tài khoản."""
    # ASSUMPTION: mức hoạt động riêng ~ Gamma(shape, mean/shape), số hành động/ngày ~ Poisson(mức) (Gamma–Poisson); tài khoản phối hợp và thường cùng phân phối.
    rates = rng.gamma(shape, mean / shape, n_accounts)
    return rng.poisson(np.repeat(rates[:, None], days, axis=1))


def background_slots(counts, rng):
    """Trả (chỉ số tài khoản, ngày, giây) cho mọi hành động nền; thời điểm đều trong ngày, độ phân giải giây."""
    days = counts.shape[1]
    cell = np.repeat(np.arange(counts.size), counts.ravel())
    account, day = cell // days, cell % days
    # ASSUMPTION: thời điểm hành động nền phân phối đều trong ngày (giây nguyên).
    return account, day, day * DAY_S + rng.integers(0, DAY_S, len(cell))


def replace_background(background, coord, rng):
    """Thay hành động nền bằng hành động phối hợp ở cùng tài khoản–ngày (không cộng thêm).

    background, coord: DataFrame có cột account, day. Khi số hành động phối hợp k lớn hơn số hành động nền n của tài khoản–ngày,
    bỏ hết n và giữ đủ k (nâng tổng lên k), nên tổng mỗi tài khoản–ngày = max(n, k). Trả (bảng gộp, thống kê nâng).
    """
    background = background.reset_index(drop=True)
    cells = background.groupby(["account", "day"], sort=True).indices
    need = coord.groupby(["account", "day"], sort=True).size()
    drop, raised, added = [], 0, 0
    for key, k in need.items():
        positions = cells.get(key, np.array([], dtype=np.int64))
        # ASSUMPTION: thiếu hành động nền thì nâng tổng của tài khoản–ngày đó lên bằng số hành động phối hợp (quyết định Bước 1, phương án A).
        # Không bù từ ngày khác của cùng tài khoản: điểm khối lượng tại tau chỉ dùng D(tau); bù từ ngày sau tau làm khối lượng tích lũy của IO cao hơn ở các mốc sớm (kể cả q = 25%),
        # bù từ ngày trước t0 làm IO hoạt động ít hơn trước t0 và tạo tín hiệu giả cho S3b. Cách bù đó chỉ khớp trên trọn 7 ngày và đưa vào sai lệch phụ thuộc tương lai; nâng tại chỗ thì cục bộ và thấy được.
        raised += int(k > len(positions))
        added += max(int(k) - len(positions), 0)
        # ASSUMPTION: hành động nền bị thay chọn ngẫu nhiên đều trong các hành động của tài khoản ở ngày đó.
        drop.extend(rng.choice(positions, size=min(int(k), len(positions)), replace=False).tolist())
    merged = pd.concat([background.drop(index=drop), coord], ignore_index=True)
    coord_total = int(merged["account"].isin(coord["account"].unique()).sum())
    return merged, dict(raised_account_days=raised, coord_account_days=len(need), added_actions=added, coord_group_actions=coord_total)


def raise_stats(stats, n_coord, days):
    """(i) trên tài khoản–ngày có hành động phối hợp, (ii) trên mọi tài khoản–ngày của nhóm phối hợp, (iii) hành động được thêm / tổng hành động của nhóm phối hợp. Chỉ báo cáo, không có ngưỡng."""
    return dict(stats, raised_share_coord_days=stats["raised_account_days"] / stats["coord_account_days"], raised_share_all_coord_days=stats["raised_account_days"] / (n_coord * days), added_action_share=stats["added_actions"] / stats["coord_group_actions"])


# --- Bảng 2 v2: lớp = mô phỏng, 6 phối hợp + 40 thường, kho nội dung v1 ---
def table2_datasets(data_seed, variant, t2cfg):
    """Ba bộ sự kiện (EVENT_COLUMNS của v1, thời gian phút có độ phân giải giây) cho Bảng 2 v2 và metadata.

    no_camouflage: hành động thường = generate_authentic_events của v1; mô phỏng 1, 2 như v1; mô phỏng 3 bản v2.
    camouflage: mọi tài khoản (thường và phối hợp) có hoạt động nền 7 ngày cùng phân phối và cùng kho nội dung v1; hành động phối hợp thay hành động nền.
    """
    config = SimulationConfig(seed=data_seed, n_inauthentic=t2cfg["n_inauthentic"], n_authentic=t2cfg["n_authentic"], duration_days=t2cfg["days"])
    coords = {sim: coordinated_events(sim, config) for sim in (1, 2, 3)}
    meta = []
    if variant == "no_camouflage":
        authentic = generate_authentic_events(config)
        datasets = {sim: pd.concat([authentic, coords[sim][EVENT_COLUMNS]], ignore_index=True).sort_values(["timestamp", "user_id"]).reset_index(drop=True)[EVENT_COLUMNS] for sim in coords}
        return datasets, pd.DataFrame([dict(simulation=sim) for sim in coords])
    if variant != "camouflage":
        raise ValueError(f"Biến thể không hợp lệ: {variant}")
    users = [f"auth_{i:02d}" for i in range(config.n_authentic)] + [f"coord_{i:02d}" for i in range(config.n_inauthentic)]
    index = {user: i for i, user in enumerate(users)}
    rng = np.random.default_rng([data_seed, TAG, 2, _shape_code(t2cfg["gamma_shape"])])
    account, day, seconds = background_slots(daily_counts(len(users), config.duration_days, t2cfg["gamma_shape"], t2cfg["mean_actions_per_day"], rng), rng)
    catalog = generate_action_catalog(config)
    # ASSUMPTION: nội dung nền rút từ kho 20 nội dung của v1 theo đúng xác suất của v1, cho mọi tài khoản.
    picked = catalog.iloc[rng.choice(len(catalog), size=len(seconds), replace=True, p=catalog["probability"].to_numpy())]
    background = pd.DataFrame({"account": account, "day": day, "t": seconds, "action_type": picked["action_type"].to_numpy(), "content_id": picked["content_id"].to_numpy(), "is_inauthentic": False})
    datasets = {}
    for sim, events in coords.items():
        t = np.rint(events["timestamp"].to_numpy(dtype=float) * 60).astype(np.int64)
        coord = pd.DataFrame({"account": events["user_id"].map(index).to_numpy(), "day": t // DAY_S, "t": t, "action_type": events["action_type"].to_numpy(), "content_id": events["content_id"].to_numpy(), "is_inauthentic": True})
        merged, stats = replace_background(background, coord, np.random.default_rng([data_seed, TAG, 2, _shape_code(t2cfg["gamma_shape"]), sim]))
        frame = pd.DataFrame({"user_id": np.asarray(users)[merged["account"].to_numpy()], "timestamp": merged["t"].to_numpy() / 60.0, "action_type": merged["action_type"].to_numpy(), "content_id": merged["content_id"].to_numpy(), "is_inauthentic": merged["is_inauthentic"].to_numpy(dtype=bool)})
        datasets[sim] = frame.sort_values(["timestamp", "user_id"]).reset_index(drop=True)[EVENT_COLUMNS]
        meta.append(dict(simulation=sim, **raise_stats(stats, config.n_inauthentic, config.duration_days)))
    return datasets, pd.DataFrame(meta)


# --- Thế giới cho các chế độ: 4 lớp theo schema Seckin, quần thể lớn ---
def content_catalog(wcfg):
    """Kho nội dung mỗi lớp: danh sách ID và xác suất Zipf (1/k^s) chung cho mọi tài khoản."""
    catalog = {}
    for layer, size in wcfg["catalog_sizes"].items():
        # ASSUMPTION: độ phổ biến nội dung trong mỗi lớp theo Zipf với số mũ zipf_s.
        weights = 1.0 / np.arange(1, size + 1) ** wcfg["zipf_s"]
        catalog[layer] = (np.array([f"{layer}_{k:04d}" for k in range(size)]), weights / weights.sum())
    return catalog


def world_ids(data_seed, n_accounts):
    """accountid dạng chuỗi theo hoán vị ngẫu nhiên, để thứ tự ID không mang thông tin nhãn (chỉ số 0..n_coord-1 là tài khoản phối hợp)."""
    # ASSUMPTION: ID tài khoản không tương quan với nhãn (dữ liệu thật dùng ID băm).
    order = np.random.default_rng([data_seed, TAG, 1]).permutation(n_accounts)
    return np.array([f"a{k:05d}" for k in order])


def world_background(data_seed, shape, wcfg):
    """Hoạt động nền của mọi tài khoản trên toàn bộ số ngày (chung cho mọi kịch bản của cùng data seed và shape)."""
    n_accounts = wcfg["n_normal"] + wcfg["n_coord"]
    rng = np.random.default_rng([data_seed, TAG, 3, _shape_code(shape)])
    account, day, seconds = background_slots(daily_counts(n_accounts, wcfg["days"], shape, wcfg["mean_actions_per_day"], rng), rng)
    layers = list(wcfg["layer_probs"])
    # ASSUMPTION: mỗi hành động (bài đăng) thuộc đúng một lớp, chọn theo layer_probs; một bài chỉ mang một nội dung.
    action = np.asarray(layers)[rng.choice(len(layers), size=len(seconds), p=np.array([wcfg["layer_probs"][layer] for layer in layers]))]
    content = np.empty(len(seconds), dtype=object)
    for layer, (ids, probs) in content_catalog(wcfg).items():
        mask = action == layer
        content[mask] = ids[rng.choice(len(ids), size=int(mask.sum()), p=probs)]
    return pd.DataFrame({"account": account, "day": day, "t": seconds, "action": action, "content": content, "is_coord_action": False})


def make_world(data_seed, sim, shape, wcfg, background=None):
    """Thế giới của một data seed × kịch bản sim × Gamma shape: WORLD_COLUMNS (t = giây từ START) và metadata (t0, số hành động phối hợp, tỷ lệ bị nâng)."""
    background = world_background(data_seed, shape, wcfg) if background is None else background
    n_coord = wcfg["n_coord"]
    events = coordinated_events(sim, SimulationConfig(seed=data_seed, n_inauthentic=n_coord))
    mapping = wcfg["sim_content_layers"][f"sim{sim}"]
    catalog = content_catalog(wcfg)
    rng_map = np.random.default_rng([data_seed, TAG, 4, sim])
    # ASSUMPTION: nội dung phối hợp i của loại v1 ánh xạ vào nội dung perm[i] của lớp tương ứng (hoán vị ngẫu nhiên có seed), tức chọn đều trong kho như v1 chọn cặp nội dung.
    perms = {v1_type: rng_map.permutation(len(catalog[layer][0])) for v1_type, layer in mapping.items()}
    content = [catalog[mapping[kind]][0][perms[kind][int(cid.rsplit("_", 1)[1])]] for kind, cid in zip(events["action_type"], events["content_id"])]
    t = np.rint(events["timestamp"].to_numpy(dtype=float) * 60).astype(np.int64)
    account = events["user_id"].str.rsplit("_", n=1).str[1].astype(int).to_numpy()
    coord = pd.DataFrame({"account": account, "day": t // DAY_S, "t": t, "action": events["action_type"].map(mapping).to_numpy(), "content": content, "is_coord_action": True})
    merged, stats = replace_background(background, coord, np.random.default_rng([data_seed, TAG, 5, _shape_code(shape), sim]))
    ids = world_ids(data_seed, wcfg["n_normal"] + n_coord)
    world = pd.DataFrame({"accountid": ids[merged["account"].to_numpy()], "t": merged["t"].to_numpy(np.int64), "day": merged["day"].to_numpy(np.int64), "action": merged["action"].to_numpy(), "content": merged["content"].to_numpy(), "label": (merged["account"] < n_coord).astype(int).to_numpy(), "is_coord_action": merged["is_coord_action"].to_numpy(dtype=bool)})
    world = world.sort_values(["t", "accountid"], kind="stable", ignore_index=True)[WORLD_COLUMNS]
    meta = dict(data_seed=data_seed, scenario=f"sim{sim}", gamma_shape=float(shape), mean_actions_per_day=float(wcfg["mean_actions_per_day"]), t0=int(t.min()), t1=None, n_coord_actions=len(t), n_world_actions=len(world), **raise_stats(stats, n_coord, wcfg["days"]))
    return world, meta


# --- Các chế độ: biến đổi tất định có seed của cùng một thế giới ---
def floor_minute(t):
    return t - t % 60


def select_controls(world, ctrl_offset, ctrl_selection, rng):
    """Chế độ giống Seckin: mỗi ngày d có tài khoản phối hợp hoạt động, chọn n_io(d) + ctrl_offset tài khoản thường hoạt động ngày d.

    Trả ({ngày: [accountid]}, số control bổ sung ngẫu nhiên, tổng số control cần). Nhãn chỉ dùng để mô phỏng quy trình thu thập.
    """
    io, normal = world.loc[world["label"].eq(1)], world.loc[world["label"].eq(0)]
    # ASSUMPTION: "tài khoản phối hợp hoạt động ngày d" tính trên mọi hành động (kể cả nền), vì quy trình thu thập không phân biệt được; hệ quả là ngày nào cũng có control.
    io_days = io.groupby("day")["accountid"].nunique()
    picks, filled, needed = {}, 0, 0
    for day, n_io in io_days.items():
        need = int(n_io) + int(ctrl_offset)
        today = normal.loc[normal["day"].eq(day)]
        active = np.array(sorted(today["accountid"].unique()))
        if ctrl_selection == "hashtag":
            # ASSUMPTION: chọn control trong số tài khoản thường dùng ít nhất một hashtag mà nhóm phối hợp dùng trong ngày d (mọi hashtag, kể cả của hoạt động nền).
            tags = set(io.loc[io["day"].eq(day) & io["action"].eq("hashtag"), "content"])
            candidates = np.array(sorted(today.loc[today["action"].eq("hashtag") & today["content"].isin(tags), "accountid"].unique()))
        elif ctrl_selection == "uniform":
            candidates = active
        else:
            raise ValueError(f"ctrl_selection không hợp lệ: {ctrl_selection}")
        if len(active) < need:
            raise ValueError(f"Ngày {day} chỉ có {len(active)} tài khoản thường hoạt động, cần {need}.")
        if len(candidates) >= need:
            chosen = rng.choice(candidates, size=need, replace=False).tolist()
        else:
            # ASSUMPTION: thiếu ứng viên hashtag thì bổ sung ngẫu nhiên đều từ các tài khoản thường hoạt động ngày d còn lại.
            rest = np.setdiff1d(active, candidates)
            chosen = candidates.tolist() + rng.choice(rest, size=need - len(candidates), replace=False).tolist()
            filled += need - len(candidates)
        needed += need
        picks[int(day)] = sorted(chosen)
    return picks, filled, needed


def seckin_sample(world, ctrl_offset, ctrl_selection, cap, rng, round_io):
    """Chế độ seckin (round_io=True) hoặc sampling_only (round_io=False): IO giữ toàn bộ thời gian; control chỉ giữ ngày được chọn, tối đa cap hành động đầu/ngày, độ phân giải giây."""
    picks, filled, needed = select_controls(world, ctrl_offset, ctrl_selection, rng)
    io = world.loc[world["label"].eq(1)].copy()
    if round_io:
        io["t"] = floor_minute(io["t"])
    chosen = pd.DataFrame([(day, account) for day, accounts in picks.items() for account in accounts], columns=["day", "accountid"])
    # ASSUMPTION: chọn control độc lập theo từng ngày; một tài khoản có thể là control ở nhiều ngày, khi đó giữ hành động của từng ngày được chọn.
    control = world.loc[world["label"].eq(0)].merge(chosen, on=["day", "accountid"], how="inner")
    control = control.sort_values(["t", "accountid"], kind="stable").groupby(["accountid", "day"], sort=False).head(cap)
    sample = pd.concat([io, control], ignore_index=True).sort_values(["t", "accountid"], kind="stable", ignore_index=True)[WORLD_COLUMNS]
    return sample, dict(fill_share=filled / needed if needed else np.nan, n_ctrl_unique=int(control["accountid"].nunique()), n_ctrl_needed=needed)


def clean_sample(world, n_ctrl, rng, round_io):
    """Chế độ clean (round_io=False) hoặc rounding_only (round_io=True): IO và n_ctrl tài khoản thường lấy ngẫu nhiên đều, giữ toàn bộ timeline."""
    io = world.loc[world["label"].eq(1)].copy()
    if round_io:
        io["t"] = floor_minute(io["t"])
    # ASSUMPTION: mẫu đều lấy trong các tài khoản thường có ít nhất một hành động trong 7 ngày (tài khoản không hoạt động không có trong dữ liệu).
    normal_ids = np.array(sorted(world.loc[world["label"].eq(0), "accountid"].unique()))
    chosen = set(rng.choice(normal_ids, size=int(n_ctrl), replace=False).tolist())
    control = world.loc[world["label"].eq(0) & world["accountid"].isin(chosen)]
    sample = pd.concat([io, control], ignore_index=True).sort_values(["t", "accountid"], kind="stable", ignore_index=True)[WORLD_COLUMNS]
    return sample, dict(fill_share=np.nan, n_ctrl_unique=len(chosen), n_ctrl_needed=int(n_ctrl))


def make_modes(world, meta, modes_cfg, cap, clean_from):
    """Mọi chế độ của một thế giới. Mẫu ngẫu nhiên có khóa RNG theo (data seed, kịch bản, shape, cách chọn, offset), nên seckin và sampling_only cùng tham số dùng đúng một tập control.

    Kích thước mẫu clean và rounding_only = số control duy nhất của chế độ clean_from cùng thế giới. Trả {tên: (mẫu, info)}.
    """
    key = [meta["data_seed"], TAG, 6, _shape_code(meta["gamma_shape"]), int(meta["scenario"][3:])]
    selection_codes = {"hashtag": 1, "uniform": 2}
    out = {}
    for name, mode in modes_cfg.items():
        if mode["kind"] == "seckin":
            rng = np.random.default_rng([*key, selection_codes[mode["ctrl_selection"]], int(mode["ctrl_offset"])])
            out[name] = seckin_sample(world, mode["ctrl_offset"], mode["ctrl_selection"], cap, rng, mode["round_io"])
    n_clean = out[clean_from][1]["n_ctrl_unique"]
    for name, mode in modes_cfg.items():
        if mode["kind"] == "clean":
            out[name] = clean_sample(world, n_clean, np.random.default_rng([*key, 0, 0]), mode["round_io"])
    return {name: out[name] for name in modes_cfg}


def to_posts(sample):
    """Đổi mẫu sang schema đầu vào của seckin_protocol (INPUT_COLUMNS) và giữ thêm các cột của thế giới; post_time độ phân giải giây (UTC)."""
    action, content = sample["action"].to_numpy(), sample["content"].to_numpy()
    posts = pd.DataFrame({
        "accountid": sample["accountid"].astype(str).to_numpy(),
        "post_time": START + pd.to_timedelta(sample["t"].to_numpy(np.int64), unit="s"),
        "is_control": sample["label"].eq(0).to_numpy(),
        "is_repost": action == "retweet",
        "reposted_postid": np.where(action == "retweet", content, None),
        "in_reply_to_postid": None,
        "hashtags": [[c] if a == "hashtag" else [] for a, c in zip(action, content)],
        "account_mentions": [[c] if a == "mention" else [] for a, c in zip(action, content)],
        "urls": [[c] if a == "url" else [] for a, c in zip(action, content)],
    })
    for column in ("t", "action", "content", "label", "is_coord_action"):
        posts[column] = sample[column].to_numpy()
    return posts
