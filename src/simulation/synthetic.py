from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    n_inauthentic: int = 6
    n_authentic: int = 40
    n_authentic_events: int = 1000
    n_inauthentic_events_min: int = 15
    n_inauthentic_events_max: int = 20
    duration_days: int = 7
    n_action_types: int = 3
    n_actions: int = 20
    seed: int = 20261108

EVENT_COLUMNS = ["user_id", "timestamp", "action_type", "content_id", "is_inauthentic",]

def generate_authentic_timestamps(
    config: SimulationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)

    duration_minutes = config.duration_days * 24 * 60

    user_ids = [
        f"auth_{i:02d}"
        for i in range(config.n_authentic)
    ]

    users = rng.choice(
        user_ids,
        size=config.n_authentic_events,
        replace=True,
    )

    timestamps = rng.uniform(
        low=0.0,
        high=duration_minutes,
        size=config.n_authentic_events,
    )

    df = pd.DataFrame({
        "user_id": users,
        "timestamp": timestamps,
        "is_inauthentic": False,
    })

    return df.sort_values(
        "timestamp"
    ).reset_index(drop=True)


def generate_action_catalog(
    config: SimulationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 1)

    action_types = ["hashtag", "retweet", "mention"]
    type_counts = [7, 7, 6]

    assert config.n_action_types == len(action_types)
    assert config.n_actions == sum(type_counts)

    rows = []

    for action_type, count in zip(action_types, type_counts):
        for i in range(count):
            rows.append({
                "action_type": action_type,
                "content_id": f"{action_type}_{i:02d}",
            })

    catalog = pd.DataFrame(rows)

    component = rng.choice(
        [0, 1],
        size=len(catalog),
        p=[0.6, 0.4],
    )

    means = np.where(component == 0, 3.0, 12.0)

    weights = rng.normal(
        loc=means,
        scale=1.0,
    )

    weights = np.clip(weights, 0.001, None)

    catalog["weight"] = weights
    catalog["probability"] = weights / weights.sum()

    return catalog

def generate_authentic_events(
    config: SimulationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 2)

    events = generate_authentic_timestamps(config)
    catalog = generate_action_catalog(config)

    selected = rng.choice(
        len(catalog),
        size=len(events),
        replace=True,
        p=catalog["probability"].to_numpy(),
    )

    sampled = catalog.iloc[selected].reset_index(drop=True)

    events["action_type"] = sampled["action_type"].to_numpy()
    events["content_id"] = sampled["content_id"].to_numpy()

    return events[EVENT_COLUMNS]

# Kịch bản 1: Mô phỏng một nhóm tài khoản cùng hoạt động trong một khoảng thời gian rất ngắn
def generate_simulation1_timestamps(
    config: SimulationConfig,
) -> np.ndarray:
    rng = np.random.default_rng(config.seed + 3)

    # Day 2, 08:00 (minutes since start of simulation)
    burst_start = 24 * 60 + 8 * 60

    burst_duration = 15.0
    rate_per_minute = 1.3

    n_events = rng.poisson(
        lam=rate_per_minute * burst_duration
    )

    timestamps = rng.uniform(
        low=burst_start,
        high=burst_start + burst_duration,
        size=n_events,
    )

    return np.sort(timestamps)

def assign_simulation1_users(
    config: SimulationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 4)

    timestamps = generate_simulation1_timestamps(config)

    if not (
        config.n_inauthentic_events_min
        <= len(timestamps)
        <= config.n_inauthentic_events_max
    ):
        raise ValueError(
            "Burst event count is outside the required range."
        )

    if len(timestamps) < config.n_inauthentic:
        raise ValueError("Not enough events for all users.")

    user_ids = [
        f"coord_{i:02d}"
        for i in range(config.n_inauthentic)
    ]

    assigned_users = list(user_ids)

    remaining = len(timestamps) - len(user_ids)

    assigned_users.extend(
        rng.choice(
            user_ids,
            size=remaining,
            replace=True,
        ).tolist()
    )

    rng.shuffle(assigned_users)

    return pd.DataFrame({
        "user_id": assigned_users,
        "timestamp": timestamps,
        "is_inauthentic": True,
    })


def generate_simulation1_events(
    config: SimulationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 5)

    events = assign_simulation1_users(config)
    catalog = generate_action_catalog(config)

    action_types = sorted(catalog["action_type"].unique())

    content_pairs = {}

    for action_type in action_types:
        candidates = catalog.loc[
            catalog["action_type"] == action_type,
            "content_id",
        ].to_numpy()

        pair = rng.choice(
            candidates,
            size=2,
            replace=False,
        )

        content_pairs[action_type] = pair

    selected_types = rng.choice(
        action_types,
        size=len(events),
        replace=True,
    )

    selected_contents = []

    for action_type in selected_types:
        content = rng.choice(
            content_pairs[action_type],
            p=[0.9, 0.1],
        )
        selected_contents.append(content)

    events["action_type"] = selected_types
    events["content_id"] = selected_contents

    return events[EVENT_COLUMNS]


def generate_simulation1_dataset(
    config: SimulationConfig,
) -> pd.DataFrame:
    authentic = generate_authentic_events(config)
    coordinated = generate_simulation1_events(config)

    dataset = pd.concat(
        [authentic, coordinated],
        ignore_index=True,
    )

    dataset = dataset.sort_values(
        ["timestamp", "user_id"]
    ).reset_index(drop=True)

    return dataset[EVENT_COLUMNS]

# Kịch bản 2: Mô phỏng một nhóm tài khoản hoạt động thành hai đợt riêng biệt, giữa hai đợt có khoảng thời gian im lặng.

def generate_simulation2_timestamps(
    config: SimulationConfig,
) -> np.ndarray:
    rng = np.random.default_rng(config.seed + 6)

    first_start = 24 * 60 + 8 * 60  # Day 2 - 08:00
    burst_duration = 6.0
    inactivity_duration = 90.0

    bursts = []
    for i in range(3):
        start = first_start + i * (
            burst_duration + inactivity_duration
        )

        bursts.append(
            rng.uniform(
                start,
                start + burst_duration,
                size=6,
            )
        )

    return np.sort(np.concatenate(bursts))



def assign_simulation2_users(
    config: SimulationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 7)

    timestamps = generate_simulation2_timestamps(config)

    user_ids = [
        f"coord_{i:02d}"
        for i in range(config.n_inauthentic)
    ]

    if len(timestamps) != 18 or len(user_ids) != 6:
        raise ValueError("Unexpected Simulation 2 configuration.")

    assigned_users = []

    for _ in range(3):  # Three bursts
        burst_users = list(user_ids)
        rng.shuffle(burst_users)
        assigned_users.extend(burst_users)

    return pd.DataFrame({
        "user_id": assigned_users,
        "timestamp": timestamps,
        "is_inauthentic": True,
    })


def assign_coordinated_contents(
    events: pd.DataFrame,
    config: SimulationConfig,
    seed_offset: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + seed_offset)

    catalog = generate_action_catalog(config)
    action_types = sorted(catalog["action_type"].unique())

    content_pairs = {}

    for action_type in action_types:
        candidates = catalog.loc[
            catalog["action_type"] == action_type,
            "content_id",
        ].to_numpy()

        content_pairs[action_type] = rng.choice(
            candidates,
            size=2,
            replace=False,
        )

    result = events.copy()

    selected_types = rng.choice(
        action_types,
        size=len(result),
        replace=True,
    )

    selected_contents = [
        rng.choice(content_pairs[action_type], p=[0.9, 0.1])
        for action_type in selected_types
    ]

    result["action_type"] = selected_types
    result["content_id"] = selected_contents

    return result[EVENT_COLUMNS]

def generate_simulation2_dataset(
    config: SimulationConfig,
) -> pd.DataFrame:
    authentic = generate_authentic_events(config)

    coordinated = assign_coordinated_contents(
        events=assign_simulation2_users(config),
        config=config,
        seed_offset=8,
    )

    dataset = pd.concat(
        [authentic, coordinated],
        ignore_index=True,
    )

    dataset = dataset.sort_values(
        ["timestamp", "user_id"]
    ).reset_index(drop=True)

    return dataset[EVENT_COLUMNS]

# Kịch bản 3: Phối hợp luân phiên giữa các nhóm tài khoản
def generate_simulation3_timestamps(
    config: SimulationConfig,
) -> np.ndarray:
    rng = np.random.default_rng(config.seed + 9)

    bursts = [
    (1920.0, 5.0, 5),   # Day 2 - 08:00
    (4860.0, 5.0, 6),   # Day 4 - 09:00
    (7710.0, 3.0, 4),   # Day 6 - 08:30
    (8430.0, 2.0, 3),   # Day 6 - 20:30
    ]
 
    timestamps = []

    for start, duration, n_events in bursts:
        burst_times = rng.uniform(
            low=start,
            high=start + duration,
            size=n_events,
        )

        timestamps.extend(burst_times)

    return np.sort(np.asarray(timestamps))


def assign_simulation3_users(
    config: SimulationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 10)

    timestamps = generate_simulation3_timestamps(config)

    burst_sizes = [5, 6, 4, 3]
    active_groups = [
        ["coord_00", "coord_01"],
        ["coord_02", "coord_03"],
        ["coord_04", "coord_05"],
        ["coord_00", "coord_02", "coord_04"],
    ]

    if config.n_inauthentic != 6:
        raise ValueError("Simulation 3 requires six coordinated users.")

    if len(timestamps) != sum(burst_sizes):
        raise ValueError("Unexpected number of events.")

    assigned_users = []

    for size, active_users in zip(burst_sizes, active_groups):
        users = list(active_users)

        remaining = size - len(users)

        users.extend(
            rng.choice(
                active_users,
                size=remaining,
                replace=True,
            ).tolist()
        )

        rng.shuffle(users)
        assigned_users.extend(users)

    return pd.DataFrame({
        "user_id": assigned_users,
        "timestamp": timestamps,
        "is_inauthentic": True,
    })

def generate_simulation3_dataset(
    config: SimulationConfig,
) -> pd.DataFrame:
    authentic = generate_authentic_events(config)

    coordinated = assign_coordinated_contents(
        events=assign_simulation3_users(config),
        config=config,
        seed_offset=11,
    )

    dataset = pd.concat(
        [authentic, coordinated],
        ignore_index=True,
    )

    dataset = dataset.sort_values(
        ["timestamp", "user_id"]
    ).reset_index(drop=True)

    return dataset[EVENT_COLUMNS]